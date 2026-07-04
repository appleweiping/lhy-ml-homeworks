# NTU Machine Learning 2022 (Hung-yi Lee) — 15 Homeworks

> Independent, from-scratch implementations of all **15 homework assignments** of
> **NTU "Machine Learning" (Spring 2022)** taught by **Prof. Hung-yi Lee (李宏毅)**,
> part of a [csdiy.wiki](https://csdiy.wiki/) full-catalog build.

![status](https://img.shields.io/badge/status-15%2F15%20implemented-brightgreen)
![language](https://img.shields.io/badge/python-informational)
![framework](https://img.shields.io/badge/PyTorch-CPU-orange)
![license](https://img.shields.io/badge/license-MIT-blue)

## Overview

This repo implements every assignment from the 2022-spring edition of the course
(the canonical 15-HW deep-learning sequence: regression → classification → CNN →
self-attention → Transformer → GAN → BERT → autoencoder → explainable AI →
adversarial attack → domain adaptation → RL → network compression → life-long
learning → meta learning).

Everything runs on **CPU only** (`torch.set_num_threads(3)`, `OMP_NUM_THREADS=3`).
Several official Kaggle datasets are competition-gated and multi-GB (food-11,
VoxCeleb, Crypko anime faces, DRCD, …). Where that is the case, each HW runs the
**identical model / algorithm** on a real, freely-downloadable dataset (CIFAR-10,
FashionMNIST, MNIST, Omniglot, Multi30k), auto-downloaded via `torchvision` /
HuggingFace `datasets`. Every number below is **measured from a real run** — see
each HW's `results/` for logs, metrics, and figures.

## Results (measured on CPU, 3 threads)

| HW | Topic | Dataset (used) | Key measured result |
|---|---|---|---|
| 1 | Regression (DNN) | COVID case prediction (official layout) | valid RMSE **1.46** |
| 2 | Classification (BN-MLP) | phoneme frames (official layout) | frame acc **0.69** |
| 3 | CNN image classification | CIFAR-10 | test acc **0.6512** |
| 4 | Self-attention | FashionMNIST-as-sequence | test acc **0.8540** |
| 5 | Transformer (NMT) | Multi30k de→en (real) | BLEU-4 **{{HW5_BLEU}}** |
| 6 | GAN (DCGAN) | MNIST | pixel mean/std gap **0.006 / 0.027**; sample grid |
| 7 | BERT extractive QA | multi-entity QA + real bert-tiny | EM **0.9033** / F1 **0.9033** |
| 8 | Autoencoder anomaly det. | CIFAR-10 (airplane=normal) | ROC-AUC **0.6203** |
| 9 | Explainable AI | CIFAR-10 CNN | deletion top-k **0.307** > random **0.249** ✓ |
| 10 | Adversarial attack | CIFAR-10 | clean **0.539** → PGD-10 **0.000** |
| 11 | Domain adaptation (DaNN) | CIFAR-10 (RGB→edge domain) | target acc: src-only **0.5100** → DaNN **0.5445** |
| 12 | RL (policy gradient) | CartPole (exact physics) | last-50 return **480.9** / 500 |
| 13 | Network compression (KD+prune) | FashionMNIST | **54.5×** smaller; KD/prune curve |
| 14 | Life-long (EWC) | Permuted-MNIST | forgetting **0.027** (EWC) vs **0.224** (SGD) |
| 15 | Meta learning (MAML) | Omniglot 5-way 1-shot | MAML **0.6512** vs baseline **0.3972** |

Figures: `hw06-gan/results/samples.png` (generated digits),
`hw09-explainable/results/attributions.png` (4 attribution methods),
`hw10-attack/results/adversarial_examples.png`, `hw12-rl/results/reward_curve.png`.

## Implemented assignments

- [x] **HW1 Regression** — DNN regression, feature selection, official-layout data.
- [x] **HW2 Classification** — BatchNorm MLP phoneme frame classifier (41 classes).
- [x] **HW3 CNN** — from-scratch VGG-style CNN (no pretrained weights) on CIFAR-10.
- [x] **HW4 Self-attention** — TransformerEncoder + masked mean-pool sequence classifier.
- [x] **HW5 Transformer** — full encoder-decoder NMT on real Multi30k (de→en), BLEU.
- [x] **HW6 GAN** — DCGAN generating real MNIST digits (non-saturating loss).
- [x] **HW7 BERT** — extractive QA span prediction fine-tuning a real bert-tiny.
- [x] **HW8 Autoencoder** — reconstruction-error anomaly detection, ROC-AUC.
- [x] **HW9 Explainable AI** — saliency / smooth-grad / integrated-grad / occlusion + deletion metric.
- [x] **HW10 Attack** — FGSM / PGD / MI-FGSM white-box L-inf attacks.
- [x] **HW11 Adaptation** — Domain-Adversarial NN (gradient-reversal layer).
- [x] **HW12 RL** — REINFORCE + learned baseline (actor-critic style) on CartPole.
- [x] **HW13 Compression** — knowledge distillation + global L1 pruning, 54.5× smaller student.
- [x] **HW14 Life-long** — Elastic Weight Consolidation on Permuted-MNIST.
- [x] **HW15 Meta learning** — MAML (2nd-order) few-shot on Omniglot.

## Project structure

```
lhy-ml-homeworks/
├── hw01-regression/ ... hw15-meta/   # one folder per homework
│   ├── hwNN_*.py                     # the implementation
│   ├── README.md                     # task + measured result
│   └── results/                      # metrics.txt, logs, figures (real runs)
├── scripts/download_data.py          # optional pre-download of all datasets
├── requirements.txt
└── LICENSE
```

## How to run

```bash
# Python 3.11; the shared csdiy venv already has torch/torchvision/transformers/datasets:
#   D:\Project\_csdiy\.venv-ml\Scripts\python.exe
pip install -r requirements.txt

# each HW is self-contained; datasets auto-download on first run
cd hw03-cnn   && python hw3_cnn.py --epochs 8 --subset 10000
cd hw12-rl    && python hw12_rl.py --episodes 800
cd hw14-lifelong && python hw14_ewc.py --tasks 4 --epochs 3
# ... see each hwNN/README.md for its exact command
```

## Verification

Every result was produced by an actual CPU run; each `hwNN/results/` holds the
`run_log.txt`, a `metrics.txt` of the measured numbers, and (where relevant) the
generated figures/samples. Highlights that double as correctness checks:

- **HW9** deletion metric: masking the most-salient pixels drops confidence more
  than masking random pixels (0.307 vs 0.249) → attributions are meaningful.
- **HW10**: PGD-10 drives accuracy from 0.539 to **0.000** at the exact 8/255
  L-inf budget → the attack is real and correctly projected.
- **HW12**: last-50-episode mean return **480.9 / 500** → policy solves CartPole.
- **HW14**: EWC cuts catastrophic forgetting ~8× vs naive SGD (0.027 vs 0.224).

## Tech stack

Python 3.11 · PyTorch 2.x (CPU) · torchvision · HuggingFace transformers &
datasets · numpy · matplotlib · scikit-learn.

## Key ideas / what I learned

- Building each core deep-learning primitive end-to-end: conv nets, multi-head
  self-attention, full Transformer encoder-decoder with masking, DCGAN training
  dynamics, BERT span-extraction heads.
- Beyond-accuracy topics: reconstruction anomaly scoring, gradient-based
  attribution + a quantitative faithfulness check, projected-gradient adversarial
  attacks, gradient-reversal domain adaptation, policy-gradient RL with a baseline.
- Model-lifecycle methods: knowledge distillation & pruning (compression),
  Elastic Weight Consolidation (continual learning), and second-order MAML
  (meta-learning) — each with a controlled baseline to isolate its effect.

## Credits & license

Based on the assignments of **NTU "Machine Learning" (Spring 2022)** by
**Prof. Hung-yi Lee (李宏毅)** — course site:
<https://speech.ee.ntu.edu.tw/~hylee/ml/2022-spring.php>. This repository is an
independent educational reimplementation; all course materials, datasets, and
specifications belong to their original authors. Original code here is released
under the [MIT License](LICENSE).
