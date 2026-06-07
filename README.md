# AI Image Denoising: Classical Filters vs. Deep Learning (DnCNN)

A comparative study of traditional image-denoising filters and a deep convolutional
neural network (DnCNN), built as a final project for **Introduction to Image Processing**
and **Artificial Intelligence and Applications**.

## Overview

Given a noisy image `y = x + n`, we recover the clean image `x`. We compare five
classical methods against a learned model under identical noise conditions, on both
grayscale and color images, and for both Gaussian and salt-and-pepper noise.

| Category | Methods |
|----------|---------|
| Classical | Gaussian filter, Median filter, Bilateral filter, Non-Local Means (NLM), BM3D |
| Deep learning | DnCNN (residual learning, blind denoising) |

**Metrics:** PSNR, SSIM, and average inference time.
**Datasets:** Set12 / BSD68 (grayscale), Kodak (color); training on Train400.

## Repository structure

```
.
├── AI_Image_Denoising.ipynb   # main notebook (run on Google Colab)
├── report/                    # IEEE LaTeX report (main.tex, references.bib)
├── results/                   # generated: model weights, CSV tables, figures
└── README.md
```

## Installation / How to run

The project is designed to run end-to-end on **Google Colab** (free T4 GPU).

1. Open `AI_Image_Denoising.ipynb` in Colab.
2. `Runtime → Change runtime type → T4 GPU`.
3. `Runtime → Run all`. The notebook will:
   - install dependencies (`bm3d`, `scikit-image`; `torch`, `opencv` preinstalled),
   - download datasets automatically,
   - train DnCNN (~30–60 min for 25 epochs),
   - run all methods and export `results.zip`.

To run locally instead:

```bash
pip install torch torchvision opencv-python scikit-image bm3d numpy matplotlib pandas
jupyter notebook AI_Image_Denoising.ipynb
```

## Dataset information

| Dataset | Use | Source |
|---------|-----|--------|
| Train400 (400 BSD images) | Training | https://github.com/cszn/DnCNN |
| Set12 | Grayscale testing | https://github.com/cszn/DnCNN |
| BSD68 | Grayscale testing | https://github.com/cszn/DnCNN |
| Kodak | Color testing | http://r0k.us/graphics/kodak/ |

Noise is synthesized: additive white Gaussian noise (sigma = 15/25/50) and
salt-and-pepper (5%).

## Results (summary)

Filled in after running the notebook (see `results/*.csv`). Typical ordering on
Set12 @ sigma=25: **DnCNN > BM3D > NLM > Bilateral > Median > Gaussian**, while the
median filter is best for salt-and-pepper noise and DnCNN (Gaussian-trained) degrades.

## Data Availability Statement

All source code, the trained model weights, evaluation tables, and figures are
available in this repository. The datasets used are publicly available at the URLs
listed above.

## References

See `report/references.bib` (DnCNN, BM3D, NLM, bilateral filtering, BSD dataset).
