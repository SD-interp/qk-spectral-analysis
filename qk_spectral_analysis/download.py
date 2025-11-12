from __future__ import annotations

import os
import pickle
import queue
import random
import threading
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Optional, Sequence

import einops
import torch as t
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig

from .models import get_model_names
from .utils import get_stats, robust_lowrank_singular_values


def _ensure_api(api: HfApi | None) -> HfApi:
    return api or HfApi()


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if t.cuda.is_available() else "cpu"


def _list_model_files(api: HfApi, model_name: str) -> list[str]:
    return [
        filename
        for filename in api.list_repo_files(model_name)
        if filename.endswith(".safetensors") and "model" in filename.casefold()
    ]


def _extract_attention_parameters(model_name: str) -> tuple[int, int, int]:
    cfg = AutoConfig.from_pretrained(model_name)
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    return n_heads, n_kv_heads, head_dim


def _process_tensor_pair(
    tensors: dict[str, t.Tensor],
    tensor_name: str,
    n_heads: int,
    n_kv_heads: int,
    records: dict[str, dict[int, t.Tensor]],
) -> None:
    q_name = tensor_name
    k_name = tensor_name.replace("q_proj", "k_proj")

    layer = int(tensor_name.split(".")[2])
    if k_name not in tensors:
        raise KeyError(f"Key tensor {k_name} not found for layer {layer}")

    W_Q = tensors[q_name].to(t.float32)
    W_K = tensors[k_name].to(t.float32)

    W_Q = einops.rearrange(
        W_Q,
        "(q_head d_head) d_model -> q_head d_head d_model",
        q_head=n_heads,
    )
    W_K = einops.rearrange(
        W_K,
        "(k_head d_head) d_model -> k_head d_head d_model",
        k_head=n_kv_heads,
    )

    if n_heads != n_kv_heads:
        ratio = n_heads // n_kv_heads
        W_K = t.repeat_interleave(W_K, dim=0, repeats=ratio)
        if not t.equal(W_K[0], W_K[1]):
            raise RuntimeError("Unexpected repetition order when expanding KV heads")

    singular_values = robust_lowrank_singular_values(W_Q, W_K)
    eigen_values = singular_values ** 2

    records.setdefault("singular_values", {})[layer] = singular_values.cpu().numpy()
    records.setdefault("eigen_values", {})[layer] = eigen_values.cpu().numpy()

    svd_stats = get_stats(singular_values)
    records.setdefault("singular_values_stats", {})[layer] = svd_stats

    eigen_stats = get_stats(eigen_values)
    records.setdefault("eigen_values_stats", {})[layer] = eigen_stats


def _build_output_path(output_root: Path, family: str, model_name: str) -> Path:
    output_dir = output_root / family / "data"
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir / f"{model_name.split('/')[-1]}.pkl"


def _process_model(
    model_name: str,
    family: str,
    *,
    api: HfApi,
    device: str,
    output_root: Path,
    force: bool = False,
) -> Optional[Path]:
    output_path = _build_output_path(output_root, family, model_name)
    if output_path.exists() and not force:
        return None

    files_to_download = _list_model_files(api, model_name)
    if not files_to_download:
        return None

    n_heads, n_kv_heads, _ = _extract_attention_parameters(model_name)

    file_queue: queue.Queue[str | None] = queue.Queue()
    records: dict[str, dict[int, t.Tensor]] = defaultdict(dict)

    def download_worker() -> None:
        for filename in files_to_download:
            path = hf_hub_download(repo_id=model_name, filename=filename, local_dir="/dev/shm")
            file_queue.put(path)
        file_queue.put(None)

    def gpu_worker() -> None:
        while True:
            path = file_queue.get()
            if path is None:
                file_queue.put(None)
                break
            tensors = load_file(path, device=device)
            os.remove(path)
            for tensor_name in tensors:
                if "q_proj" not in tensor_name:
                    continue
                _process_tensor_pair(tensors, tensor_name, n_heads, n_kv_heads, records)

    t.manual_seed(0)
    random.seed(0)

    thread_download = threading.Thread(target=partial(download_worker), daemon=True)
    thread_gpu = threading.Thread(target=gpu_worker, daemon=True)

    thread_download.start()
    thread_gpu.start()

    thread_download.join()
    thread_gpu.join()

    with output_path.open("wb") as f:
        pickle.dump(records, f)

    time.sleep(0.5)
    return output_path


def generate_family_spectra(
    family: str,
    *,
    models: Optional[Sequence[str]] = None,
    output_root: str | os.PathLike[str] = ".",
    api: HfApi | None = None,
    device: str | None = None,
    force: bool = False,
) -> list[Path]:
    """Download models and compute spectral statistics for a model family."""

    output_root_path = Path(output_root)
    api = _ensure_api(api)
    device = _resolve_device(device)

    if models is None:
        models = get_model_names(family)

    generated_files: list[Path] = []
    for model_name in models:
        output_path = _build_output_path(output_root_path, family, model_name)
        if output_path.exists() and not force:
            continue
        result = _process_model(
            model_name,
            family,
            api=api,
            device=device,
            output_root=output_root_path,
            force=force,
        )
        if result is not None:
            generated_files.append(result)
    return generated_files


__all__ = ["generate_family_spectra"]


