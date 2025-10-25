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
* `Generate_ESDs_and_plots.ipynb`\
This notebook downloads pretrained transformer weights (e.g., Qwen3-4B),
extracts the Query and Key projection matrices (`W_Q`, `W_K`),
computes their singular-value spectra, and generates plots for `Classification_A` and `Classification_B`.
  * The target model family can be specified by setting the `family` variable.
Model lists for each family are defined in `models.py`.

  * You can extend support for new models or families by editing `models.py`.
Each model should follow the standard file and layer naming conventions
and be accessible as `.safetensors` files.

  * Processed results are saved to `/content/qk-spectral-analysis/{model_family}/data/`.

  * You can specify the target model family via:

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
For details about the Classifcation_A and Classifcation_B refer to the post linked above.

# Citation
If you use or build upon this work, please cite or reference:

Shantanu Darveshi (2025). Spectral Taxonomy of QK Circuits in Transformer Models\
https://github.com/SD-interp/qk-spectral-analysis
