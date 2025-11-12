# Spectral Taxonomy of QK Circuits in Transformer Models

Spectral analysis of attention-layer weight matrices in transformer models.

---

## Overview

This repository contains a lightweight Python package plus a single Jupyter
notebook that orchestrates the end-to-end workflow for studying the spectral
properties of **QK circuits** in LLMs. The code has been modularised into
reusable Python modules so that the heavy lifting can run outside of a notebook
while still allowing interactive exploration from the unified entry point,
`spectral_analysis_pipeline.ipynb`.

The accompanying research write-up is available here:
👉 [**Post: "Spectral Taxonomy of QK Circuits in Transformer Models"**](https://www.lesswrong.com/posts/Yig9fc7wAxKqG63Do/spectral-taxonomy-of-qk-circuits-in-transformer-models)
*(provides background, figures, and theoretical motivation for this codebase).* A
Colab notebook mirroring the original exploratory analysis is also available
[here](https://colab.research.google.com/drive/1TH_MnMAdMZlacvNmQZUK1N-Tlx7m21P7?usp=drive_link).

---

## Package layout

The core logic lives in the `qk_spectral_analysis` package:

* `download.py` – download model weights, compute spectral statistics, and cache
  them as pickled NumPy arrays.
* `classification_a.py` – cluster eigenvalue histograms and render the figures
  produced by the original *Classification_A* notebook.
* `classification_b.py` – reproduce the statistical box plots that lived in the
  *Classification_B* notebook.
* `utils.py` – numerical helpers shared across the other modules.
* `models.py` – catalogue of supported model families and their Hugging Face
  identifiers.

All public helpers are re-exported from `qk_spectral_analysis.__init__` so they
can be imported directly, e.g. `from qk_spectral_analysis import
generate_family_spectra`.

---

## Running the pipeline notebook

The notebook `spectral_analysis_pipeline.ipynb` orchestrates the complete
workflow. Configure the model family, clustering, and binning parameters in the
first cell and then execute the remaining cells to:

1. Download the requested models and compute their spectral statistics.
2. Produce the clustered eigenvalue spectrum figures.
3. Render the statistical box plots.

All artefacts are saved under `<family>/data`, `<family>/ESDs`, and
`<family>/plots` respectively.

### Environment setup

Install the Python dependencies into your preferred environment and ensure that
`t` (PyTorch) can see a GPU if you want accelerated downloads:

```bash
pip install torch einops huggingface_hub safetensors transformers seaborn umap-learn statsmodels
```

The pipeline defaults to `cuda` when available and falls back to CPU otherwise.
Pass `device="cpu"` to `generate_family_spectra` to override this behaviour.

---

## Command line usage

While the notebook is the primary entry point, the modular design makes it easy
to drive the workflow from a script. For example:

```python
from qk_spectral_analysis import generate_family_spectra, list_families

for family in list_families():
    print(f"Processing {family} models")
    generate_family_spectra(family, output_root=".")
```

---

## Citation

If you use or build upon this work, please cite or reference:

Shantanu Darveshi (2025). Spectral Taxonomy of QK Circuits in Transformer Models\\
https://github.com/SD-interp/qk-spectral-analysis
