# Standalone SEM Dataset Generator Package

Generate synthetic SEM target localization datasets for training & evaluation on any machine.

## Features
- **60 Semiconductor Layout Sub-Scripts**: Complete CAD pattern generator suite.
- **Strict Data Leakage Protection**: 50 Training Generators (`train/`), 5 Validation Generators (`val/`), and 5 **Hidden Test Generators** (`test/`) isolated from training.
- **Stochastic Noise & Degradation Engine**: 17 SEM degradation models (Poisson, Gaussian, blur, vignetting, tilt, rotation, stage drift).
- **Hard-Negative Periodic Replica Injection**: Explicit periodic cell shifts to train periodicity disambiguation.
- **Memory-Safe & Resumable**: Automatically frees memory buffers and resumes instantly if interrupted.

## Requirements
```bash
pip install numpy opencv-python Pillow
```

## Quick Start Command

To generate 10,000 synthetic image pairs:
```bash
python generate_dataset.py --num_pairs 10000 --output_dir ./my_dataset_10k
```

To generate 2,500 pairs:
```bash
python generate_dataset.py --num_pairs 2500 --output_dir ./my_dataset_2k5
```
