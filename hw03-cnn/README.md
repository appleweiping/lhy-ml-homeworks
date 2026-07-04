# HW3 — CNN Image Classification

From-scratch VGG-style CNN (no pretrained weights, matching the official HW3
rule). Kaggle: `ml2022spring-hw3b`. The official food-11 dataset is gated, so we
train the same CNN family on **CIFAR-10** — a real, freely downloadable dataset
that FACTORY_SPEC lists as an acceptable small dataset. torchvision downloads it
automatically.

## Run
```bash
python hw3_cnn.py --epochs 8 --subset 10000   # subset for CPU speed; 0 = full 50k
```

## Measured result (CPU, 3 threads, CIFAR-10 10k-image subset, 8 epochs)
See `results/metrics.txt` (test top-1 accuracy, real CIFAR-10 test set).

Model: 3 conv blocks (64→128→256, each 2×conv+BN+ReLU+maxpool) → FC head with
dropout. AdamW + cosine LR. Random crop + horizontal flip augmentation.
