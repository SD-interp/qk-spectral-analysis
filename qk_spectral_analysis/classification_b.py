from __future__ import annotations

import pickle
from collections import defaultdict
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import make_splrep
from statsmodels.tsa.stattools import bds

sns.set_theme("talk")
sns.set_style("white")


class Stats(IntEnum):
    MIN = 0
    MAX = 1
    MEAN = 2
    STD = 3
    SKEW = 4
    KURTOSIS = 5


def generate_stat_boxplots(
    data_root: str | Path = ".",
    *,
    skip: Iterable[str] | None = None,
    family: str | None = None,
) -> list[Path]:
    """Generate boxplots for eigenvalue statistics across layers.

    Parameters
    ----------
    data_root:
        Base directory that contains per-family subdirectories.
    skip:
        Iterable of statistic names to skip.
    family:
        Optional family name to limit generation to a single model family.
    """

    base = Path(data_root)
    skip = set(skip or [])
    output_paths: list[Path] = []

    if family is None:
        family_dirs: Sequence[Path] = [p for p in base.iterdir() if p.is_dir()]
    else:
        family_dir = base / family
        family_dirs = [family_dir] if family_dir.is_dir() else []

    start = defaultdict(lambda: 2)
    start.update({
        "Llama-3.1-70B-Instruct": 5,
        "Llama-3.3-70B-Instruct": 5,
    })

    for stat_name in Stats._member_names_:
        if stat_name in skip:
            continue

        for family_dir in family_dirs:
            data_dir = family_dir / "data"
            if not data_dir.is_dir():
                continue

            for pickle_path in data_dir.glob("*.pkl"):
                width = 20 if "405" not in str(pickle_path) else 35

                plt.figure(figsize=(width, 8), dpi=100)
                family_name = family_dir.name
                model_name = pickle_path.stem

                save_path = base / family_name / "plots" / stat_name
                save_path.mkdir(exist_ok=True, parents=True)

                with pickle_path.open("rb") as file:
                    data = dict(pickle.load(file))
                    if stat_name.upper() in Stats._member_names_:
                        eigen_stats = data["eigen_values_stats"]
                        series = {
                            k: v[Stats[stat_name.upper()]].squeeze() for k, v in eigen_stats.items()
                        }
                    else:
                        series = data[stat_name]

                series = {k: v for k, v in series.items() if k >= start[model_name]}
                series = dict(sorted(series.items()))
                n_layers = len(series)

                bds_stat, p_value = bds([np.median(x) for x in series.values()])

                plt.boxplot(
                    list(series.values()),
                    positions=np.arange(n_layers),
                    widths=0.5,
                    showmeans=True,
                    showfliers=False,
                )

                if "405" not in str(pickle_path):
                    plt.xticks(np.arange(n_layers), list(series.keys()))
                else:
                    plt.xticks([])
                plt.title(f"{model_name}_{stat_name} (BDS: {bds_stat:.2f} p-value: {p_value:.2e})")

                means = [x.mean() for x in series.values()]
                f_linear = make_splrep(np.arange(n_layers), means, s=20, k=1)
                x_new = np.linspace(0, n_layers - 1, 500)
                y_new = f_linear(x_new)

                plt.plot(x_new, y_new, linestyle="--")
                plt.tight_layout()

                output_file = save_path / f"{model_name}_{stat_name}_box.png"
                plt.savefig(output_file)
                plt.close()
                output_paths.append(output_file)

    return output_paths


__all__ = ["generate_stat_boxplots", "Stats"]
