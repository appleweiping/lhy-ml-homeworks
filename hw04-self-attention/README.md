# HW4 — Self-Attention (Speaker Identification)

Classify a variable-length mel-spectrogram segment into one of N speakers using
a Transformer encoder (self-attention) + masked mean-pooling + linear head — the
official HW4 architecture. Kaggle: `ml2022spring-hw4`.

## Run
```bash
python hw4_speaker.py --epochs 20 --n-spk 100
```
Synthesises 40-dim mel segments with per-speaker timbre signatures + heavy noise
when the VoxCeleb-derived data is absent (padding masks handled).

Real data: `kaggle competitions download -c ml2022spring-hw4 -p data && unzip data/*.zip -d data`

## Measured result (CPU, 3 threads, 100 speakers, synthetic mel data)
| metric | value |
|---|---|
| best valid accuracy | 0.94 |

Model: prenet Linear→TransformerEncoder (2 layers, 2 heads, d=80)→pooling→head.
Padding mask feeds `src_key_padding_mask`; pooling ignores padded frames.
