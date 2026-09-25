# Enhanced Compression Model (ECM)

**A Hyperprior-Based Neural Image Compression Framework with Context Modeling, Entropy Parameter Network, Semantic Gating, and Scale Refinement**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)

ECM is a learned image compression model built on the variational autoencoder + scale hyperprior paradigm. It extends the standard hyperprior backbone with **semantic gating**, **channel attention**, **autoregressive context modeling**, an **improved entropy parameter network**, and a dedicated **scale refinement** module — achieving a substantially better rate–distortion trade-off, particularly at low bitrates.

On the **Kodak** benchmark, ECM achieves **30.882 dB PSNR** and **0.985 MS-SSIM** at only **0.176 BPP** — 5.5 dB higher PSNR than Scale Hyperprior (Q1) at a lower bitrate, and the highest MS-SSIM among all evaluated methods.

> 📄 This repository accompanies the paper *"Enhanced Compression Model: A Hyperprior-Based Neural Image Compression Framework with Context Modeling, Entropy Parameter Network, Semantic Gating, and Scale Refinement."*

---

## Table of Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Results](#results)
- [Ablation Study](#ablation-study)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Ablation Experiments](#ablation-experiments)
- [Datasets](#datasets)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)

---

## Highlights

- **Semantic gating** — a soft, end-to-end trained channel-selection gate that suppresses non-informative latent channels before entropy coding, without explicit supervision.
- **Channel attention** — applied in both the analysis and synthesis transforms to emphasize informative feature channels.
- **Autoregressive context modeling** — a masked-convolution context model that captures local spatial dependencies between latent coefficients.
- **Entropy parameter network** — jointly conditions on hyperprior statistics and autoregressive context features to predict Gaussian conditional means/scales.
- **Scale refinement** — corrects predicted hyperprior scales using direct latent content, improving entropy accuracy on complex textures.
- State-of-the-art rate–distortion performance at low bitrates on the Kodak dataset compared to classical (JPEG, PNG, WebP) and learned (AE, GAN, VAE, NIC, Scale Hyperprior, Stable Diffusion VAE) baselines.

## Architecture

ECM keeps the standard Scale Hyperprior backbone intact and augments it with additional modules for latent feature processing and entropy estimation:

```
Input Image → Analysis Transform (ReflectionPad Conv + GDN)
            → Channel Attention + Semantic Gate
            → Gaussian Conditional ← Entropy Parameter Network ← Context Model (Masked Conv)
                                   ← Scale Refinement ← Hyper Decoder h_s(ẑ) ← Entropy Bottleneck ← Hyper Encoder h_a(y)
            → Decoder → Reconstructed Image
```

- **Analysis transform (gₐ):** reflection-padded convolutions + GDN layers, with channel attention on the encoder output.
- **Synthesis transform (g_s):** transposed convolutions + inverse GDN layers, mirrored with channel attention.
- **Hyperprior branch (h_a / h_s):** models global statistical structure of the latents; predictions refined by the scale refinement module.
- **Context model:** masked convolution providing causal, autoregressive spatial context.
- **Entropy parameter network:** fuses hyperprior + context features to predict Gaussian means (μ) and scales (σ) for conditional entropy coding.

Trained end-to-end with a rate–distortion objective:

```
L = λ·D(x, x̂) + R(ŷ, ẑ) + α·L_perc + β·L_GAN
```

(current implementation optimizes BPP with MSE- or MS-SSIM-based distortion).

## Results

### Codec Comparison (Kodak dataset)

| Method | BPP | PSNR (dB) | MS-SSIM | Loss ↓ |
|---|---|---|---|---|
| PNG (lossless) | 15.528 | ∞ | 1.000 | 0.000 |
| JPEG Q=90 | 3.328 | 35.696 | 0.968 | 0.004 |
| WebP | 1.169 | 31.621 | 0.923 | 0.018 |
| NIC Q5 | 7.555 | 30.572 | 0.896 | 0.048 |
| Scale Hyperprior Q1 | 0.204 | 25.380 | 0.682 | 0.349 |
| Scale Hyperprior Q3 | 0.517 | 28.704 | 0.842 | 0.149 |
| Scale Hyperprior Q5 | 1.127 | 32.398 | 0.932 | 0.022 |
| SD FP16 / INT8 | 2.000 / 0.500 | 24.232 / 18.934 | 0.654 / 0.407 | 0.058 / 0.204 |
| **Enhanced Compression Model** | **0.176** | **30.882** | **0.985** | **0.0428** |

*(Full table with AE, GAN, VAE, and NIC baselines is in the paper / `results/`.)*

### Rate–Distortion Operating Points (λ = 0.1)

| Method | BPP | PSNR (dB) | MS-SSIM |
|---|---|---|---|
| Scale Hyperprior | 1.390 | 28.822 | 0.970 |
| **ECM** | **0.176** | **30.882** | **0.985** |

At matched λ, ECM delivers **+2.06 dB PSNR** at **less than 1/8th the bitrate** of Scale Hyperprior.

### Perceptual Quality (FID, Kodak)

ECM achieves a competitive **FID of 48.848**, comparable to JPEG Q=70 (47.110) and better than WebP (57.926), confirming reasonable perceptual fidelity despite the low bitrate. Full FID results are in [`fid_results_kodak.csv`](./fid_results_kodak.csv).

## Ablation Study

Each variant disables one component; all other training conditions are held constant.

| Variant | BPP | PSNR (dB) | MS-SSIM | Train Loss ↓ |
|---|---|---|---|---|
| **A0: Full Model** | **0.176** | **30.882** | **0.985** | **8.245** |
| A1: w/o context model & entropy parameter net | 1.865 | 28.753 | 0.968 | 12.836 |
| A2: w/o context model | 0.602 | 28.562 | 0.970 | 12.838 |
| A3: w/o semantic gate | 0.283 | 28.465 | 0.977 | 11.228 |

- **Context-aware entropy modeling** (A1 vs. A0) is the single largest contributor: a **90.6% bitrate reduction** and **+2.13 dB PSNR**.
- **Semantic gating** (A3 vs. A0) contributes an additional **0.107 BPP reduction** and **+0.417 dB PSNR**.

Reproduce these with `Ablation1.py`, `Ablation2.py`, and `Ablation3.py`.

## Repository Structure

```
.
├── agents/                     # Model/agent implementations used across experiments
├── results/                    # Saved metrics, logs, and evaluation outputs
├── Ablation1.py                # A1: remove context model + entropy parameter network
├── Ablation2.py                # A2: remove context model only
├── Ablation3.py                # A3: remove semantic gating
├── evaluate_all_agents.py      # Run evaluation across all baselines / agents
├── evaluate_fid.py             # FID computation on reconstructed images
├── fid_results_kodak.csv       # FID scores on the Kodak dataset
├── kodak_data_compression.ipynb# Notebook: Kodak dataset compression walkthrough
├── losses_eval.py              # Rate-distortion loss / metric evaluation utilities
├── model_enhanced.py           # Enhanced Compression Model (ECM) architecture
├── scale_hyperprior.py         # Baseline Scale Hyperprior implementation
├── train.py                    # Training entry point
├── webp_preset.py              # WebP baseline preset/config
├── requirements.txt            # Python dependencies
└── .gitignore
```

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Hardware used for the paper's experiments:** NVIDIA RTX 5080 GPU, PyTorch with CUDA + mixed precision training (AMP).

## Usage

### Training

Train ECM on the DIV2K dataset with patch-based rate–distortion optimization:

```bash
python train.py \
  --dataset_path /path/to/DIV2K \
  --patch_size 256 \
  --batch_size 4 \
  --epochs 50 \
  --lr 1e-4 \
  --aux_lr 1e-3 \
  --seed 42
```

Key hyperparameters (see paper, Section VI-C):

| Hyperparameter | Value |
|---|---|
| Patch size | 256 × 256 |
| Optimizer (main / aux) | AdamW / Adam |
| Learning rate | 1 × 10⁻⁴ |
| Auxiliary learning rate | 1 × 10⁻³ |
| Batch size | 4 |
| Epochs | 50 |
| Seed | 42 |
| Mixed precision | Enabled (CUDA) |

### Evaluation

Evaluate the trained model (and baselines) on the Kodak dataset:

```bash
python evaluate_all_agents.py --checkpoint /path/to/checkpoint.pth --dataset kodak
python evaluate_fid.py --checkpoint /path/to/checkpoint.pth --dataset kodak
```

Metrics reported: **BPP**, **PSNR**, **MS-SSIM**, **FID**, and **Train Loss** (see `losses_eval.py`).

You can also follow the end-to-end walkthrough in [`kodak_data_compression.ipynb`](./kodak_data_compression.ipynb).

### Ablation Experiments

```bash
python Ablation1.py   # A1 — w/o context model & entropy parameter network
python Ablation2.py   # A2 — w/o context model
python Ablation3.py   # A3 — w/o semantic gating
```

## Datasets

- **Training:** [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) — 800 high-resolution images, trained on random 256×256 crops with horizontal/vertical flip augmentation.
- **Evaluation:** [Kodak Lossless True Color Image Suite](http://r0k.us/graphics/kodak/) — 24 uncompressed natural images, the standard benchmark for compression research.

## Citation

If you use this code or build on ECM, please cite the paper:

```bibtex
@inproceedings{
  title     = {Enhanced Compression Model: A Hyperprior-Based Neural Image Compression Framework with Context Modeling, Entropy Parameter Network, Semantic Gating, and Scale Refinement},
  author    = {Valappil, Abhinand Meethele and Singh, Sanidhya and Gupta, Rishabh and Randhar, Garv and Upadhayay, Bhawna},
  booktitle = {IEEE Conference Proceedings},
  year      = {2026}
}
```

## Authors

- **Abhinand Meethele Valappil** — SRM Institute of Science and Technology
- **Sanidhya Singh** — Manipal University Jaipur
- **Rishabh Gupta** — SRM Institute of Science and Technology
- **Garv Randhar** — SRM Institute of Science and Technology
- **Bhawna Upadhayay** (Assistant Professor, Advisor) — SRM Institute of Science and Technology

## License

This project is released under the [MIT License](LICENSE).

---

*Future work: reducing autoregressive decoding complexity (e.g., checkerboard/parallel decoding) and extending ECM to video compression.*