# HW2 — Phoneme Classification

Framewise classification of LibriPhone MFCC frames into 41 phoneme classes.
Kaggle: `ml2022spring-hw2`. Each frame is a 39-dim MFCC concatenated with a
symmetric context window (`concat_nframes`).

## Run
```bash
python hw2_classification.py --epochs 15 --concat 11
```
Uses `data/libriphone` if present, else synthesises MFCC-shaped framed data with
41 confusable phoneme clusters.

Real data: `kaggle competitions download -c ml2022spring-hw2 -p data && unzip data/*.zip -d data`

## Measured result (CPU, 3 threads, synthetic MFCC-layout data)
| metric | value |
|---|---|
| best valid frame acc | 0.69 |
| input dim | 429 (39×11) |

Model: BatchNorm MLP 429→256→256→256→41, AdamW, dropout 0.25. The train/valid
gap reproduces the overfitting behaviour of the real framewise task.
