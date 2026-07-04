# HW8 — Autoencoder (Anomaly Detection)

Reconstruction-based anomaly detection — the official HW8 method. A convolutional
autoencoder is trained to reconstruct only "normal" images; at test time a high
reconstruction error flags anomalies. The official curated face/anomaly set is
gated, so we use the real **CIFAR-10** dataset (torchvision auto-download): one
class (airplane) is treated as *normal* and trained on; the other nine classes
are *anomalies* at test time.

## Run
```bash
python hw8_anomaly.py --epochs 15 --normal-class 0
```

## Measured result (CPU, 3 threads, CIFAR-10, airplane = normal)
| metric | value |
|---|---|
| test ROC-AUC (MSE recon error as anomaly score) | **0.6203** |

Balanced test set (500 normal / 500 anomaly). ROC-AUC computed from scratch. The
autoencoder learns airplane structure, so airplanes reconstruct with lower error
than the other classes, giving a > 0.5 AUC separation.
