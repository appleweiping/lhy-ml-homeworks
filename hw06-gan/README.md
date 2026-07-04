# HW6 — GAN (image generation)

A **DCGAN** (strided-conv generator & discriminator, BatchNorm, non-saturating
GAN loss) — the official HW6 architecture family. The official task generates
anime faces from the gated Crypko dataset (many GPU-hours), so we train the same
GAN on the real **MNIST** handwritten-digit dataset (torchvision auto-download),
28×28 upsampled to 32×32.

## Run
```bash
python hw6_gan.py --epochs 8 --n-train 12000
```

## Measured result (CPU, 3 threads, MNIST)
| signal | value |
|---|---|
| final D / G loss | 0.654 / 2.14 (balanced, no collapse) |
| pixel mean-gap (real vs gen) | 0.006 |
| pixel std-gap | 0.027 |

Artefacts: `results/samples.png` (8×8 grid of generated digits), `results/losses.png`
(D/G loss curves). The small real-vs-generated pixel-statistic gaps indicate the
generator matched the data distribution.
