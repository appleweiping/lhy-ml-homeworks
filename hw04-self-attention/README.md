# HW4 — Self-Attention (Sequence Classification)

The official HW4 classifies a variable-length mel-spectrogram utterance into one
of N speakers with a Transformer encoder (self-attention) + masked mean-pooling +
linear head (Kaggle `ml2022spring-hw4`). The VoxCeleb-derived data is
competition-gated and multi-GB, so we exercise the **identical architecture** on
a real, freely-downloadable dataset: **FashionMNIST**, where each 28×28 image is
read as a length-T sequence of T row-vectors (dim 28). Sequences are randomly
truncated and padded with a proper padding mask, so the padding-masked
self-attention + masked mean-pooling path is genuinely used — exactly as in the
official speaker pipeline.

## Run
```bash
python hw4_speaker.py --epochs 12
```
FashionMNIST auto-downloads via torchvision.

## Measured result (CPU, 3 threads, real FashionMNIST, 12 epochs)
| metric | value |
|---|---|
| best test accuracy | **0.8540** |

Model: prenet Linear → sinusoidal PE → TransformerEncoder (2 layers, 4 heads,
d=80) → masked mean-pool → MLP head. Padding mask feeds `src_key_padding_mask`;
pooling ignores padded frames.
