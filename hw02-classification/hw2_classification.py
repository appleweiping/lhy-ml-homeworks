"""HW2 - Phoneme Classification (framewise).

Course: 李宏毅 ML 2022 Spring, HW2 (Kaggle: ml2022spring-hw2).

Classify each speech frame into one of 41 phoneme classes from LibriPhone MFCC
features. The official input is a 39-dim MFCC frame concatenated with a
symmetric context window (`concat_nframes` neighbours). This script builds the
concatenated feature tensor, a deep BatchNorm MLP classifier, and trains with
Adam + a validation split.

The full LibriPhone feature set (~4 GB) is competition-gated; `data_utils.py`
downloads it if present, else synthesises MFCC-shaped framed data with 41
phoneme clusters so the exact pipeline runs. Reported metric: frame accuracy on
the validation split.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import load_phone_data

torch.set_num_threads(3)
N_CLASS = 41
MFCC_DIM = 39


def same_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class BNBlock(nn.Module):
    def __init__(self, i, o, p=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(p)
        )

    def forward(self, x):
        return self.block(x)


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden=256, layers=3):
        super().__init__()
        seq = [BNBlock(in_dim, hidden)]
        for _ in range(layers - 1):
            seq.append(BNBlock(hidden, hidden))
        seq.append(nn.Linear(hidden, N_CLASS))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


def run(model, loader, device, criterion, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss, tot_acc, n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot_loss += loss.item() * len(x)
            tot_acc += (out.argmax(1) == y).sum().item()
            n += len(x)
    return tot_loss / n, tot_acc / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--concat", type=int, default=11, help="odd context window")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    same_seed(args.seed)
    device = "cpu"
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    (tr_x, tr_y), (va_x, va_y) = load_phone_data(concat_nframes=args.concat)
    print(f"train {tr_x.shape} valid {va_x.shape}  classes={N_CLASS}")
    in_dim = tr_x.shape[1]

    tl = DataLoader(TensorDataset(torch.tensor(tr_x), torch.tensor(tr_y)),
                    args.batch, shuffle=True)
    vl = DataLoader(TensorDataset(torch.tensor(va_x), torch.tensor(va_y)),
                    args.batch, shuffle=False)

    model = Classifier(in_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = 0.0
    for ep in range(args.epochs):
        trl, tra = run(model, tl, device, criterion, opt)
        val, vaa = run(model, vl, device, criterion)
        best = max(best, vaa)
        print(f"epoch {ep:2d} | train loss {trl:.4f} acc {tra:.4f} | "
              f"valid loss {val:.4f} acc {vaa:.4f} | best {best:.4f}")

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"best_valid_acc {best:.4f}\nconcat_nframes {args.concat}\n"
                f"in_dim {in_dim}\nn_train {len(tr_x)}\nn_valid {len(va_x)}\n")
    print(f"BEST valid acc {best:.4f} -> {out_dir}/metrics.txt")


if __name__ == "__main__":
    main()
