# Fingerprint Recognition Framework

A comprehensive fingerprint recognition framework implementing classical and deep learning methods for biometric verification.

## Overview

This framework provides a complete implementation of fingerprint recognition systems, from classical image-based methods to state-of-the-art deep learning approaches. All methods are implemented explicitly without relying on black-box libraries.

## Features

- **Classical Baselines**: MSE, NCC, SSIM image similarity
- **Enhancement**: Gabor filtering, orientation field estimation
- **Minutiae-Based**: Crossing number extraction, geometric matching
- **Descriptor-Based**: MCC (Minutia Cylinder Code), Local Orientation
- **Deep Learning**: CNN embedding, Patch-based CNN, Hybrid models
- **Evaluation**: ROC curves, EER computation, cross-sensor analysis

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd hcmus_biometrics

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
fingerprint_recognition/
├── data/
│   ├── raw/                    # Original fingerprint images
│   │   ├── FVC2004/           # FVC2004 dataset
│   │   └── Neurotech/         # Neurotechnology datasets
│   ├── processed/             # Enhanced/preprocessed images
│   └── pairs/                 # Generated verification pairs
│
├── src/
│   ├── baselines/             # Image-based matchers (MSE, NCC, SSIM)
│   ├── enhancement/           # Fingerprint enhancement
│   │   ├── orientation_field.py
│   │   ├── ridge_frequency.py
│   │   └── gabor_filter.py
│   ├── minutiae/              # Minutiae processing
│   │   ├── thinning.py
│   │   ├── minutiae_extraction.py
│   │   └── minutiae_matching.py
│   ├── descriptors/           # Feature descriptors
│   │   ├── mcc.py
│   │   ├── local_orientation_descriptor.py
│   │   └── descriptor_matching.py
│   ├── models/                # Deep learning models
│   │   ├── cnn_embedding.py
│   │   ├── patch_cnn.py
│   │   └── hybrid_model.py
│   ├── evaluation/            # Evaluation metrics
│   │   ├── roc.py
│   │   ├── eer.py
│   │   └── verification.py
│   ├── data/                  # Data handling
│   └── utils/                 # Utilities
│
├── experiments/               # Experiment scripts
├── configs/                   # Configuration files
└── results/                   # Output results
```

## Implemented Models

### 1. Image-Based Baselines

```python
from src.baselines.ssim import MSEMatcher, NCCMatcher, SSIMMatcher

matcher = SSIMMatcher()
score = matcher.match(image1, image2)
```

### 2. Minutiae-Based Matching

```python
from src.minutiae.minutiae_matching import MinutiaeMatcher
from src.minutiae.minutiae_extraction import MinutiaeExtractor

extractor = MinutiaeExtractor()
matcher = MinutiaeMatcher(alignment_method="ransac")

minutiae1 = extractor.extract(skeleton1)
minutiae2 = extractor.extract(skeleton2)
score = matcher.match(minutiae1, minutiae2)
```

### 3. MCC Descriptor Matching

```python
from src.descriptors.mcc import MCCDescriptor, MCCConfig
from src.descriptors.descriptor_matching import MCCMatcher

config = MCCConfig(radius=70, num_spatial_cells=16)
matcher = MCCMatcher(config=config, method="lss")

desc1 = MCCDescriptor(config)
desc1.compute(minutiae1)

desc2 = MCCDescriptor(config)
desc2.compute(minutiae2)

score = matcher.match_descriptors(desc1, desc2)
```

### 4. CNN Embedding

```python
from src.models import CNNEmbeddingMatcher, CNNConfig

config = CNNConfig(embedding_dim=128)
matcher = CNNEmbeddingMatcher(config=config, device="cuda")

score = matcher.match(image1, image2)
```

### 5. Hybrid Model

```python
from src.models import HybridMatcher, HybridConfig

config = HybridConfig(
    cnn_matcher="embedding",
    classical_matcher="mcc",
    fusion_method="weighted"
)
matcher = HybridMatcher(config=config)

score = matcher.match(image1, image2)
```

## Running Experiments

### SSIM Baseline

```bash
python experiments/exp_ssim_baseline.py \
    --data_dir data/raw/FVC2004/DB1_B \
    --output_dir results/ssim_baseline
```

### Minutiae Matching

```bash
python experiments/exp_minutiae_matching.py \
    --data_dir data/raw/FVC2004/DB1_B \
    --alignment ransac
```

### MCC Descriptor

```bash
python experiments/exp_mcc_descriptor.py \
    --data_dir data/raw/FVC2004/DB1_B \
    --method lss \
    --compare  # Compare MCC vs Local Orientation
```

### CNN Embedding

```bash
python experiments/exp_cnn_embedding.py \
    --data_dir data/raw/FVC2004/DB1_B \
    --train \
    --epochs 50 \
    --device cuda
```

### Cross-Sensor Evaluation

```bash
python experiments/exp_cross_sensor.py \
    --data_dir data/raw \
    --matcher mcc
```

## Evaluation Metrics

The framework provides comprehensive biometric evaluation:

```python
from src.evaluation import VerificationEvaluator, compute_eer

evaluator = VerificationEvaluator(matcher)
result = evaluator.evaluate(genuine_pairs, impostor_pairs)

print(f"EER: {result.eer * 100:.2f}%")
print(f"AUC: {result.auc:.4f}")
print(f"d': {result.d_prime:.2f}")
```

### Metrics Computed

- **EER** (Equal Error Rate): Operating point where FAR = FRR
- **AUC** (Area Under ROC Curve): Overall discrimination ability
- **d'** (d-prime): Separation between genuine/impostor distributions
- **FAR@FRR=0.1%**: Security-oriented operating point
- **FRR@FAR=1%**: Convenience-oriented operating point

## Configuration

YAML configuration files in `configs/`:

```yaml
# configs/mcc.yaml
model:
  name: "mcc"
  type: "descriptor"

mcc:
  radius: 70
  num_spatial_cells: 16
  num_angular_sections: 6
  sigma_s: 7.0
  min_minutiae_in_cylinder: 2

matching:
  method: "lss"
  num_top_pairs: 5
```

## Dataset Format

Expected fingerprint image format:
- Grayscale images (`.tif`, `.png`, `.jpg`)
- Resolution: 500 DPI recommended
- Organized by subject: `subject_id/sample_id.tif`

FVC2004 format:
```
FVC2004/DB1_B/
├── 101_1.tif
├── 101_2.tif
├── ...
├── 110_8.tif
```

## Mathematical Formulations

### Minutia Cylinder Code (MCC)

The MCC descriptor encodes local minutiae arrangements:

```
c(i,j,k) = Σ_t Ψ_s(d_s(m, m_t, i, j)) × Ψ_d(d_φ(m, m_t, k))
```

Where:
- `Ψ_s`: Spatial contribution (Gaussian)
- `Ψ_d`: Directional contribution (wrapped Gaussian)
- `d_s`: Distance from cell to minutia
- `d_φ`: Angular difference

### CNN Contrastive Loss

```
L = (1-y) × 0.5 × D² + y × 0.5 × max(0, m - D)²
```

Where:
- `D = ||f(x₁) - f(x₂)||`: Embedding distance
- `y = 0` for genuine, `y = 1` for impostor
- `m`: Margin parameter

## Results

Expected performance on FVC2004 DB1_B:

| Method | EER (%) | AUC |
|--------|---------|-----|
| SSIM | ~15-20 | ~0.85 |
| Minutiae | ~8-12 | ~0.92 |
| MCC | ~5-8 | ~0.95 |
| CNN | ~3-6 | ~0.97 |
| Hybrid | ~2-5 | ~0.98 |

## Requirements

- Python 3.8+
- NumPy, SciPy, OpenCV
- PyTorch (for deep learning models)
- scikit-learn, scikit-image

See `requirements.txt` for complete list.

## References

1. Maltoni, D., Maio, D., Jain, A. K., & Prabhakar, S. (2009). 
   *Handbook of Fingerprint Recognition*. Springer.

2. Cappelli, R., Ferrara, M., & Maltoni, D. (2010). 
   "Minutia Cylinder-Code: A New Representation and Matching Technique 
   for Fingerprint Recognition." IEEE TPAMI, 32(12), 2128-2141.

3. ISO/IEC 19795-1:2021 - Biometric performance testing and reporting.
