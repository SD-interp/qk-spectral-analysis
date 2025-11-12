"""Model catalogue for the spectral analysis pipeline."""

from __future__ import annotations

from typing import Mapping, Sequence

_MODEL_REGISTRY: Mapping[str, Sequence[str]] = {
    "Qwen3": (
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-1.7B",
    ),
    "Gemma2": (
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
    ),
    "Llama": (
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.1-405B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ),
    "Mistral": (
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.1",
    ),
}


def get_model_names(family: str) -> list[str]:
    """Return the list of model identifiers for a model *family*.

    Parameters
    ----------
    family:
        Canonical name of the model family (e.g. ``"Llama"``).

    Returns
    -------
    list[str]
        Ordered collection of Hugging Face model identifiers.

    Raises
    ------
    ValueError
        If the requested family is unknown.
    """

    try:
        models = _MODEL_REGISTRY[family]
    except KeyError as exc:  # pragma: no cover - defensive branch
        known = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"unknown family '{family}'. Known families: {known}") from exc
    return list(models)


def list_families() -> list[str]:
    """Return the supported model families in alphabetical order."""

    return sorted(_MODEL_REGISTRY)


__all__ = ["get_model_names", "list_families"]
