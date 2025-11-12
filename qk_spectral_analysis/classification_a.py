from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import gamma
from sklearn.cluster import KMeans
import umap

sns.set_theme("talk")
sns.set_style("white")


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))

    jsd = 0.5 * (kl_pm + kl_qm)
    return float(jsd)


def _gamma_pdf(x: np.ndarray, a: float, scale: float) -> np.ndarray:
    return gamma.pdf(x, a, loc=0, scale=scale)


def _load_eigenvalues(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        data = pickle.load(f)["eigen_values"]
    eigenvals = np.concatenate(list(data.values()), axis=0)
    return eigenvals


def _build_histograms(eigenvals: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    normalized = eigenvals / eigenvals.mean(axis=-1)[..., np.newaxis]
    max_bin_edge = np.quantile(normalized, 0.96)
    bin_edges = np.linspace(0.0, max_bin_edge, n_bins + 1)

    hists = np.stack(
        [np.histogram(x, bin_edges, density=True)[0] for x in normalized],
        axis=0,
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return hists, bin_centers


def generate_cluster_figures(
    family: str,
    *,
    n_clusters: int = 6,
    n_bins: int = 24,
    data_root: str | Path = ".",
) -> list[Path]:
    """Generate spectral cluster visualisations for a family."""

    data_location = Path(data_root) / family / "data"
    save_loc = Path(data_root) / family / "ESDs"
    save_loc.mkdir(exist_ok=True, parents=True)

    figure_paths: list[Path] = []
    for eigenvalue_path in sorted(data_location.glob("*.pkl")):
        eigenvals = _load_eigenvalues(eigenvalue_path)
        hists, bin_centers = _build_histograms(eigenvals, n_bins)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        labels = kmeans.fit_predict(hists)

        reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1)
        embedding = reducer.fit_transform(hists)

        representative_hists = [hists[labels == x].mean(axis=0) for x in range(n_clusters)]
        hist_std = [hists[labels == x].std(axis=0) for x in range(n_clusters)]

        n_rows = int(3 + max((np.ceil((n_clusters - 6) / 5)), 0))
        fig, axs = plt.subplots(
            n_rows,
            5,
            figsize=(15, 2.8 * (n_rows)),
            gridspec_kw={"width_ratios": [0.65, 0.65, 0.65, 1, 1]},
        )

        gs = axs[0, 0].get_gridspec()
        for ax in axs[:3, :3].ravel():
            ax.remove()

        ax_big = fig.add_subplot(gs[:3, :3])
        ax_big.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=10, cmap="Set2", alpha=0.75)
        ax_big.set_xticks([])
        ax_big.set_yticks([])
        ax_big.set_title("UMAP features")

        axs = axs.ravel()
        skip = [0, 1, 2, 5, 6, 7, 10, 11, 12]
        axs = [ax for i, ax in enumerate(axs) if i not in skip]

        for i in range(n_clusters):
            axs[i].bar(range(n_bins), representative_hists[i])
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].errorbar(
                range(n_bins),
                representative_hists[i],
                yerr=hist_std[i],
                fmt="none",
                capsize=2.0,
                color="black",
                alpha=0.7,
                elinewidth=2,
            )

            popt, _ = curve_fit(
                _gamma_pdf,
                bin_centers[:],
                representative_hists[i][:],
                p0=(3.0, 0.5),
                sigma=representative_hists[i].std(),
                absolute_sigma=True,
            )
            fit_alpha, fit_theta = popt
            pdf_vals = _gamma_pdf(bin_centers, fit_alpha, fit_theta)

            axs[i].plot(range(n_bins), pdf_vals, color="red", linestyle="--", lw=0.8)
            axs[i].set_ylim((0, axs[i].get_ylim()[1]))

            jsd = jensen_shannon_divergence(representative_hists[i], pdf_vals)
            axs[i].text(
                0.6,
                0.9,
                f"α : {fit_alpha:.2f}\nθ : {fit_theta:.2f}",
                transform=axs[i].transAxes,
                ha="left",
                va="top",
                fontsize=16,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            )
            axs[i].set_title(f"Class_{i} (JSD:{jsd:.3f})")

        name = eigenvalue_path.stem
        fig.suptitle(name)
        plt.tight_layout()

        output_path = save_loc / f"{name}.png"
        fig.savefig(output_path)
        plt.close(fig)
        figure_paths.append(output_path)

    return figure_paths


__all__ = ["generate_cluster_figures", "jensen_shannon_divergence"]
