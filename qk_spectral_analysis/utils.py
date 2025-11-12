"""Numerical helpers used across the spectral analysis pipeline."""

from __future__ import annotations

import warnings
from typing import List

import numpy as np
import torch as t


def get_stats(values: t.Tensor, eps: float = 1e-9) -> List[np.ndarray]:
    """Return descriptive statistics for the final dimension of ``values``.

    The statistics are computed along the last axis and returned as NumPy
    arrays so that they can be serialised with :mod:`pickle` without pulling
    the entire tensor graph back onto the CPU more than once.
    """

    if values.ndim == 0:
        raise ValueError("expected a tensor with at least one dimension")

    min_ = values.min(dim=-1, keepdim=True).values
    max_ = values.max(dim=-1, keepdim=True).values
    mean = values.mean(dim=-1, keepdim=True)
    std = values.std(dim=-1, keepdim=True)

    zscores = (values - mean) / std.clamp_min(eps)
    skew = (zscores**3).mean(dim=-1, keepdim=True)
    kurtosis = (zscores**4).mean(dim=-1, keepdim=True) - 3.0

    stats = (min_, max_, mean, std, skew, kurtosis)
    return [x.detach().cpu().numpy() for x in stats]


def _warn_if_masked(mask: t.Tensor, name: str) -> None:
    kept = int(mask.sum().item())
    total = mask.numel()
    if kept == total:
        return
    message = f"dropped {total - kept} {name} singular values below tolerance"
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def robust_lowrank_singular_values(A: t.Tensor, B: t.Tensor) -> t.Tensor:
    """Compute stable singular values for the product ``A^T B``.

    The helper mirrors the numerical tricks that lived in the original
    notebooks, including masking of numerically insignificant singular values
    before computing the reduced product.
    """

    if A.shape != B.shape:
        raise ValueError("expected tensors with matching shapes for A and B")

    *batch, m, n = A.shape

    # Thin SVD
    UA, SA, VA = t.linalg.svd(A.transpose(-2, -1), full_matrices=False)
    UB, SB, VB = t.linalg.svd(B.transpose(-2, -1), full_matrices=False)

    # Automatic tolerance for float64
    eps = t.finfo(A.dtype).eps
    tolA = np.sqrt(max(m, n)) * eps * SA.max(dim=-1, keepdim=True).values
    tolB = np.sqrt(max(m, n)) * eps * SB.max(dim=-1, keepdim=True).values

    # Zero out small singular values
    maskA = (SA > tolA).to(SA.dtype)
    maskB = (SB > tolB).to(SB.dtype)

    _warn_if_masked(maskA, "A")
    _warn_if_masked(maskB, "B")

    SA = SA * maskA
    SB = SB * maskB

    # Reduced product
    M = (SA.unsqueeze(-1) * (VA.transpose(-2, -1) @ VB)) * SB.unsqueeze(-2)
    svals = t.linalg.svdvals(M)
    return svals


__all__ = ["get_stats", "robust_lowrank_singular_values"]
