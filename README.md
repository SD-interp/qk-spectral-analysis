# Spectral Taxonomy of QK Circuits in Transformer Models

Spectral analysis of attention-layer weight matrices in transformer models.

---

## Overview

This repository contains code and analysis notebooks for studying the spectral properties of **QK circuits** in LLMs [(Colab link)](https://colab.research.google.com/drive/1TH_MnMAdMZlacvNmQZUK1N-Tlx7m21P7?usp=drive_link)

The accompanying research write-up is available here:
👉 [**Post: "Spectral Taxonomy of QK Circuits in Transformer Models"**](https://www.lesswrong.com/posts/Yig9fc7wAxKqG63Do/spectral-taxonomy-of-qk-circuits-in-transformer-models)
*(provides background, figures, and theoretical motivation for this codebase).*

---

## Package layout

The core logic now lives in a Python package, `qk_spectral_analysis`, which is organised as follows:

* `qk_spectral_analysis/download.py` – downloads model weights and computes spectral statistics, saving them as pickled NumPy arrays.
* `qk_spectral_analysis/classification_a.py` – clusters eigenvalue histograms and renders the figures previously produced by `Classification_A.ipynb`.
* `qk_spectral_analysis/classification_b.py` – reproduces the box-plot based diagnostics that were in `Classification_B.ipynb`.
* `qk_spectral_analysis/utils.py` – numerical helpers for SVD computation and summary statistics.
* `qk_spectral_analysis/models.py` – helper to enumerate supported Hugging Face model identifiers.

Each module can be imported independently or accessed via the package root. The repository now keeps the executable logic in these Python modules and provides a single entry-point notebook, `spectral_analysis_pipeline.ipynb`, that wires them together.

---

## Running the full pipeline

The notebook `spectral_analysis_pipeline.ipynb` orchestrates the complete workflow. Configure the family, clustering, and binning parameters in the first cell and then execute the remaining cells to:

1. Download the requested models and compute their spectral statistics.
2. Produce the clustered eigenvalue spectrum figures.
3. Render the statistical box plots.

All artefacts are saved under `<family>/data`, `<family>/ESDs`, and `<family>/plots` respectively.

---

## Citation

If you use or build upon this work, please cite or reference:

Shantanu Darveshi (2025). Spectral Taxonomy of QK Circuits in Transformer Models\
https://github.com/SD-interp/qk-spectral-analysis
