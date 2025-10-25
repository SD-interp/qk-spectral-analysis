# Spectral Taxonomy of QK Circuits in Transformer Models

Spectral analysis of attention-layer weight matrices in transformer models.

---

## Overview

This repository contains code and analysis notebooks for studying the spectral properties of **QK circuits** in LLMs [(Colab link)](https://colab.research.google.com/drive/1TH_MnMAdMZlacvNmQZUK1N-Tlx7m21P7?usp=drive_link)


The accompanying research write-up is available here:  
👉 [**Post: "Spectral Taxonomy of QK Circuits in Transformer Models"**](https://www.lesswrong.com/posts/Yig9fc7wAxKqG63Do/spectral-taxonomy-of-qk-circuits-in-transformer-models)  
*(provides background, figures, and theoretical motivation for this codebase).*

---
# Key Components
* `compute_qk_ESD.ipynb`\
Downloads pretrained transformer weights, extracts $W_Q$ and $W_K$, and efficiently computes the singular-value spectra of $W_{QK}=W_QW_K^T$.
Results are stored under {model_family}/data/ for later visualization.

You can specify the target model family via:

```python
family = "Qwen3"  # or "Llama", "Gemma2", "Mistral"
```

* `utils.py`\
Utility functions for statistical analysis and numerically stable SVD computations.\
`get_stats(values)` → returns `[min, max, mean, std, skew, kurtosis]`\
`robust_lowrank_singular_values(A, B)` → efficiently computes the singular values of $AB^T$

* `models.py`\
Defines model families and corresponding pretrained models.

* **Classification Notebooks**\
For details about the Classifcation_A and Classifcation_B refer to the post linked above

# Citation
If you use or build upon this work, please cite or reference:

Shantanu Darveshi (2025). Spectral Taxonomy of QK Circuits in Transformer Models\
https://github.com/SD-interp/qk-spectral-analysis
