"""HW8 - Autoencoder (Anomaly Detection).

Course: 李宏毅 ML 2022 Spring, HW8 (Kaggle: ml2022spring-hw8).

Reconstruction-based anomaly detection: train an autoencoder to reconstruct
"normal" images; at test time, high reconstruction error flags anomalies —
exactly the official method. The official data is a curated human-face / anomaly
set (gated). We use the real CIFAR-10 dataset (torchvision auto-download): one
class is treated as *normal* (train the AE only on it) and the other 9 classes
are *anomalies* at test time. Reported metric: ROC-AUC of the reconstruction
error as an anomaly score on a balanced normal/anomaly test set.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(3)


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),   # 16
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),  # 8
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),  # 4
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),  # 8
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),   # 16
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh(),    # 32
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def load_split(data_dir, normal_class):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
    train = torchvision.datasets.CIFAR10(data_dir, True, tf, download=True)
    test = torchvision.datasets.CIFAR10(data_dir, False, tf, download=True)
    tr_x = torch.stack([train[i][0] for i in range(len(train))
                        if train.targets[i] == normal_class])
    # balanced test: normal vs anomaly
    test_targets = np.array(test.targets)
    norm_idx = np.where(test_targets == normal_class)[0][:500]
    anom_idx = np.where(test_targets != normal_class)[0][:500]
    te_x = torch.stack([test[i][0] for i in np.concatenate([norm_idx, anom_idx])])
    te_y = np.concatenate([np.zeros(len(norm_idx)), np.ones(len(anom_idx))])  # 1=anomaly
    return tr_x, te_x, te_y


def roc_auc(scores, labels):
    order = np.argsort(-scores)
    labels = labels[order]
    P, N = labels.sum(), (1 - labels).sum()
    tp = fp = 0
    tpr_prev = fpr_prev = 0.0
    auc = 0.0
    for lab in labels:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        tpr, fpr = tp / P, fp / N
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev, fpr_prev = tpr, fpr
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--normal-class", type=int, default=0)  # airplane
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    torch.manual_seed(0)
    device = "cpu"
    here = os.path.dirname(__file__)
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    tr_x, te_x, te_y = load_split(os.path.join(here, "..", "hw03-cnn", "data"),
                                  args.normal_class)
    print(f"train(normal) {tr_x.shape} test {te_x.shape} anomaly_rate {te_y.mean():.2f}")

    model = ConvAutoEncoder().to(device)
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tl = DataLoader(TensorDataset(tr_x), args.batch, shuffle=True)

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for (x,) in tl:
            x = x.to(device)
            rec = model(x)
            loss = criterion(rec, x)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(x)
        # eval AUC each epoch
        model.eval()
        with torch.no_grad():
            rec = model(te_x.to(device))
            err = ((rec - te_x.to(device)) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
        auc = roc_auc(err, te_y)
        print(f"epoch {ep:2d} | train MSE {tot/len(tr_x):.5f} | test ROC-AUC {auc:.4f}")

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"test_ROC_AUC {auc:.4f}\nnormal_class {args.normal_class}\n"
                f"n_train_normal {len(tr_x)}\nn_test {len(te_x)}\n")
    np.save(os.path.join(out_dir, "recon_errors.npy"), err)
    print(f"final test ROC-AUC {auc:.4f} -> {out_dir}/metrics.txt")


if __name__ == "__main__":
    main()
