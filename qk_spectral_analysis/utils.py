import torch as t
import numpy as np
from typing import List

"""Computes stats """
def get_stats(values: t.Tensor, eps: float = 1e-9) -> List[np.ndarray]:
    min_ = values.min(dim=-1, keepdim=True)[0]
    max_ = values.max(dim=-1, keepdim=True)[0]
    mean = values.mean(dim=-1, keepdim=True)
    std = values.std(dim=-1, keepdim=True)

    zscores = (values - mean) / std.clamp_min(eps)
    skew = (zscores ** 3).mean(dim=-1, keepdim=True)
    kurtosis = (zscores ** 4).mean(dim=-1, keepdim=True) - 3.0

    stats = [min_, max_, mean, std, skew, kurtosis]
    return [x.cpu().numpy() for x in stats]

def robust_lowrank_singular_values(A: t.Tensor, B: t.Tensor) -> t.Tensor:

    *batch, m, n = A.shape
    assert B.shape == tuple(batch + [m, n])

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

    if not t.all(maskA):
        print(f"dropping A: {maskA.sum()} values")
    if not t.all(maskB):
        print(f"dropping B: {maskB.sum()} values")

    SA = SA * maskA
    SB = SB * maskB

    # Reduced product
    M = (SA.unsqueeze(-1) * (VA.transpose(-2, -1) @ VB)) * SB.unsqueeze(-2)
    svals = t.linalg.svdvals(M)
    return svals
