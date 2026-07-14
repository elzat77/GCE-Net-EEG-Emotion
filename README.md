# GCE-Net: Graph Convolutional Emotion Network for EEG Few-shot Calibration

Official implementation for cross-subject EEG emotion recognition on SEED dataset with ordinal classification loss.

## Overview

GCE-Net integrates **Graph Convolutional Networks (GCN)** with domain-adversarial training and supervised contrastive learning for **Leave-One-Subject-Out (LOSO)** 3-class emotion classification (Negative / Neutral / Positive).

### Key Results

| Method | Cold-start Accuracy | Std | Notes |
|:---|:---:|:---:|:---|
| **GCE-Net (Ordinal BCE, w2=1.5)** | **0.6519** | **0.0903** | **Our best** |
| GCE-Net (CE baseline) | 0.6444 | 0.1025 | Standard cross-entropy |
| RGNN | 0.7920 | — | State-of-the-art |
| BiHDM | 0.6740 | — | Hierarchical GNN |
| BiDANN | 0.6220 | — | Domain adaptation |
| DGCNN | 0.5880 | — | GNN-based |
| STRNN | 0.5650 | — | RNN-based |
| DBN | 0.5250 | — | Early deep learning |
| SVM (DE features) | 0.4180 | — | Classical ML baseline |
| Random Chance | 0.3333 | — | Theoretical lower bound |

- **Inference**: 56 μs/sample (RTX 5090), 357K parameters
- **Few-shot calibration**: Not beneficial at 6-12 samples (degrades by 1.9pp)
- **SEED benchmark comparison**: Outperforms DGCNN (0.588), BiDANN (0.622); competing with BiHDM (0.674)

## Project Structure

```
├── configs/              # YAML experiment configs
│   ├── default.yaml      # Main config (ordinal BCE, best settings)
│   └── ablate_*.yaml     # Ablation study configs
├── scripts/
│   ├── cross_validate.py # Main 15-fold LOSO experiment (cold-start + calibration)
│   ├── train.py           # Single-fold training script
│   └── analyze.py         # Statistical comparison tool
├── src/
│   ├── models/           # GCE_Net, EEGNet, layers (GRL, GCN, SpatialPool, SEBlock)
│   ├── training/         # Trainer, evaluator, losses (CombinedLoss, SupCon), calibrator
│   ├── data/             # SEED_DEDataset, data sampler
│   ├── analysis/         # Error analysis, domain analysis, statistics
│   ├── utils/            # Config loader, checkpoint I/O, metrics tracker
│   └── visualization/    # Confusion matrices, learning curves, attention plots
├── static/
│   └── adjacency_62.pt  # 62-electrode pearson correlation graph
└── requirements.txt
```

## Quick Start

### 1. Prepare SEED Dataset

Place the SEED DE features (*.mat files) under `SEED/ExtractedFeatures/`. Each file should be named `{subject_id}_{date}.mat` containing keys `de_LDS1` through `de_LDS15`.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Build Adjacency Matrix

```bash
python -m src.utils.build_graph
```

### 4. Run Experiment

```bash
# Full 15-fold LOSO (cold-start only, no calibration)
python scripts/cross_validate.py \
  --config configs/default.yaml \
  --run_name my_experiment \
  --cold_only

# With calibration
python scripts/cross_validate.py \
  --config configs/default.yaml \
  --run_name my_experiment_with_cal
```

### 5. Statistical Comparison

```bash
python scripts/analyze.py results/exp_a results/exp_b
```

## Configuration

Key settings in `configs/default.yaml`:

```yaml
training:
  use_ordinal: true       # Enable ordinal classification loss
  ordinal_w2: 1.5          # Asymmetric weight for Pos-vs-Rest task

loss:
  lambda_dann1: 0.5        # Domain adversarial weight (shallow)
  lambda_dann2: 0.5        # Domain adversarial weight (deep)
  lambda_supcon: 0.3       # Supervised contrastive weight

ablation:
  use_gcn: true            # Graph convolution
  use_dann1: true           # Shallow domain classifier
  use_dann2: true           # Deep domain classifier
  use_supcon: true          # SupCon head
  use_spatial_pool: true    # Spatial attention pooling
```

## Citation

```bibtex
@article{gcenet2025,
  title={GCE-Net: Graph Convolutional Emotion Network with Ordinal Classification Loss for Cross-Subject EEG Emotion Recognition},
  author={},
  journal={},
  year={2025}
}
```

## License

MIT License
