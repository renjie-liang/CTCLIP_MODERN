# Modernizing CT-CLIP for Zero-Shot Disease Detection

A modernized implementation of CT-CLIP that integrates state-of-the-art Transformer components to improve efficiency and stability for 3D medical vision-language learning.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## Overview

This project addresses a critical gap in 3D medical vision-language models: while **CT-CLIP** has demonstrated strong performance in chest CT analysis, it still relies on early Transformer architectures from 2021 that are computationally expensive and difficult to optimize for large volumetric data.

We modernize CT-CLIP by incorporating recent advances in large-scale vision-language models, achieving:
- **23.6% reduction** in GPU memory usage
- **1.07× speedup** in training iteration time
- **More stable** image-text alignment with limited batch sizes
- **Improved performance** on zero-shot disease detection tasks

### Key Improvements

Our modernization framework integrates four major architectural innovations:

1. **FlashAttention** - Reduces memory consumption and computational overhead through tiling-based attention computation
2. **RMSNorm** - Improves numerical stability by replacing LayerNorm with root-mean-square normalization
3. **SwiGLU Activation** - Enhances feature representation through gated activation mechanisms in feed-forward networks
4. **SigLIP Loss** - Provides more stable contrastive learning under small batch sizes compared to traditional softmax-based loss

---

## Architecture

### Model Components

- **Visual Encoder**: CT-ViT (3D Vision Transformer) with modern attention mechanisms
- **Text Encoder**: BiomedBERT for processing radiology reports
- **Training Objective**: SigLIP contrastive loss for stable image-text alignment

### Technical Highlights

**Traditional CT-CLIP Architecture:**
```
Standard Attention + GELU + LayerNorm + Softmax Loss
├── Memory: 118.98 GB
├── Visual Encoder Forward: 344.52 ms
├── Backward Pass: 881.89 ms
└── Total Step Time: 1311.98 ms
```

**Modernized Architecture:**
```
FlashAttention + SwiGLU + RMSNorm + SigLIP Loss
├── Memory: 90.95 GB (-23.6%)
├── Visual Encoder Forward: 320.68 ms (1.07× faster)
├── Backward Pass: 827.22 ms (1.07× faster)
└── Total Step Time: 1234.89 ms (1.06× faster)
```

---

## Experimental Results

### Efficiency Comparison

| Architecture | Loss | GPU Memory | Visual Encoder | Backward Pass | Total Step |
|-------------|------|------------|----------------|---------------|------------|
| SA + GELU + LN | SigLIP | 118.98 GB | 344.52 ms | 881.89 ms | 1311.98 ms |
| SA + GELU + LN | Softmax | 118.98 GB | 344.53 ms | 882.98 ms | 1313.41 ms |
| **FA + SwiGLU + RMSNorm** | **SigLIP** | **90.95 GB** | **320.68 ms** | **827.22 ms** | **1234.89 ms** |
| **FA + SwiGLU + RMSNorm** | **Softmax** | **90.94 GB** | **321.12 ms** | **827.26 ms** | **1234.15 ms** |

*SA = Standard Attention, FA = FlashAttention, LN = LayerNorm*

### Performance Metrics

Trained on **15.8% of CT-RATE dataset** (7,463 training volumes):

| Model | AUROC | F1 Score | Precision | Recall | AUPRC |
|-------|-------|----------|-----------|--------|-------|
| CT-CLIP (Original Paper, 100% data) | 0.731 | 0.707 | 0.323 | N/A | N/A |
| Baseline (Softmax Loss) | 0.604 | 0.450 | 0.294 | 0.666 | 0.342 |
| **+ SigLIP Loss** | **0.620** | **0.453** | 0.308 | **0.774** | **0.344** |
| **+ FlashAttention + SwiGLU + RMSNorm** | 0.616 | **0.453** | **0.320** | 0.678 | **0.344** |

**Key Findings:**
- SigLIP loss delivers **+2.6% AUROC improvement** and **+10.8% Recall boost** even with limited data
- Structural optimizations maintain performance while significantly reducing computational costs
- The modernized model shows more stable training dynamics under small batch size constraints

---

## Dataset

### CT-RATE Dataset

The first large-scale public dataset pairing 3D chest CT volumes with radiology reports.

| Dataset Version | Training Volumes | Validation Volumes | Resolution | Voxel Spacing | Content |
|----------------|------------------|-------------------|------------|---------------|---------|
| **Full CT-RATE** | 47,319 | 3,038 | 480×480×240 | 0.75×0.75×1.5 mm | 18 abnormality labels + text reports |
| **Our Subset** | 7,463 (15.8%) | 498 (16.4%) | 480×480×240 | 0.75×0.75×1.5 mm | Same |

**Abnormality Categories (18 total):**
- Lung abnormalities: nodules, masses, ground-glass opacities, consolidation, etc.
- Cardiovascular findings: cardiomegaly, pericardial effusion, etc.
- Other findings: pleural effusion, lymphadenopathy, etc.

---

## Quick Start

### Prerequisites

```bash
# Python 3.8+
# PyTorch 2.0+
# CUDA 11.8+ (for GPU acceleration)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/renjie-liang/CTCLIP_MODERN.git
cd CTCLIP_MODERN

# Install dependencies
pip install -r run_setup/requirements.txt

# Or use conda environment
conda env create -f run_setup/b200_env.yml
conda activate ctclip_modern
```

### Training

```bash
    python train.py --config configs/subset_atten_softmax.yaml  
```

**Default Configuration:**
- **GPUs**: 1 (configurable)
- **Batch Size**: 16 per GPU 
- **Learning Rate**: 1.25e-6
- **Epochs**: 20
- **Precision**: fp16 mixed precision
- **Optimizer**: AdamW (weight decay: 0.01)
- **Scheduler**: Cosine with 300-step warmup



## Acknowledgments

- **CT-CLIP**: Original architecture and training framework
- **CT-RATE Dataset**: Large-scale chest CT dataset with radiology reports
- **FlashAttention**: Tri Dao et al., Stanford University
- **SigLIP**: Zhai et al., Google Research

