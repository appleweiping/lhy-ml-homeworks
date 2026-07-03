"""HW1 - Regression (COVID-19 daily positive-case prediction).

Course: 李宏毅 (Hung-yi Lee) ML 2022 Spring, HW1 (Kaggle: ml2022spring-hw1).

A deep neural network regresses the 4th-day tested-positive rate from a 3-day
window of US-state COVID survey features (117 input features in the official
data). This script implements the full official pipeline: feature selection,
train/valid split, a configurable MLP, training with early stopping, and a
submission CSV.

The official Kaggle CSV is competition-gated. `data_utils.py` downloads it if a
kaggle token is present; otherwise it synthesises a dataset with the *exact*
column layout (40 state one-hot + 3x survey blocks) and a realistic non-linear
target so the model trains on real-shaped data. The reported metric is the
mean-squared error on a held-out validation split.
"""
import argparse
import csv
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data_utils import load_covid_data

torch.set_num_threads(3)


def same_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_valid_split(x, y, valid_ratio, seed):
    n_valid = int(len(x) * valid_ratio)
    idx = np.arange(len(x))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    v, t = idx[:n_valid], idx[n_valid:]
    return x[t], y[t], x[v], y[v]


def select_features(train_x, valid_x, test_x, select_all=True):
    """Keep the 40 state one-hots + all survey features (official baseline)."""
    if select_all:
        feat_idx = list(range(train_x.shape[1]))
    else:
        # states (0..39) + the two most-recent-day tested_positive columns
        feat_idx = list(range(40)) + [53, 69, 85, 101]
        feat_idx = [i for i in feat_idx if i < train_x.shape[1]]
    return train_x[:, feat_idx], valid_x[:, feat_idx], test_x[:, feat_idx]


class CovidDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i] if self.y is None else (self.x[i], self.y[i])


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def trainer(train_loader, valid_loader, model, cfg, device, out_dir):
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=1e-5
    )
    best_loss, best_state, early = math.inf, None, 0
    history = []
    for epoch in range(cfg["n_epochs"]):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        vlosses = []
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                vlosses.append(criterion(model(x), y).item())
        tr, va = float(np.mean(losses)), float(np.mean(vlosses))
        history.append((epoch, tr, va))
        if va < best_loss:
            best_loss, best_state, early = va, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            early += 1
        if epoch % 20 == 0 or epoch == cfg["n_epochs"] - 1:
            print(f"epoch {epoch:4d} | train {tr:.4f} | valid {va:.4f} | best {best_loss:.4f}")
        if early >= cfg["early_stop"]:
            print(f"early stop at epoch {epoch}")
            break
    model.load_state_dict(best_state)
    return best_loss, history


def predict(loader, model, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for x in loader:
            preds.append(model(x.to(device)).cpu())
    return torch.cat(preds).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1314520)
    ap.add_argument("--select-all", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    device = "cpu"
    same_seed(args.seed)

    train_data, test_data = load_covid_data()
    print(f"train shape {train_data.shape}  test shape {test_data.shape}")

    tr_x, tr_y, va_x, va_y = train_valid_split(
        train_data[:, :-1], train_data[:, -1], 0.2, args.seed
    )
    tr_x, va_x, te_x = select_features(tr_x, va_x, test_data, args.select_all)

    # standardise features using train stats
    mu, sd = tr_x.mean(0), tr_x.std(0) + 1e-8
    tr_x, va_x, te_x = (tr_x - mu) / sd, (va_x - mu) / sd, (te_x - mu) / sd

    cfg = {"n_epochs": args.epochs, "lr": args.lr, "early_stop": 60}
    tl = DataLoader(CovidDataset(tr_x, tr_y), args.batch, shuffle=True)
    vl = DataLoader(CovidDataset(va_x, va_y), args.batch, shuffle=False)
    pl = DataLoader(CovidDataset(te_x), args.batch, shuffle=False)

    model = MLP(tr_x.shape[1]).to(device)
    best, history = trainer(tl, vl, model, cfg, device, out_dir)
    rmse = math.sqrt(best)
    print(f"BEST valid MSE {best:.4f}  RMSE {rmse:.4f}")

    preds = predict(pl, model, device)
    with open(os.path.join(out_dir, "submission.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "tested_positive"])
        for i, p in enumerate(preds):
            w.writerow([i, float(p)])

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"valid_MSE {best:.6f}\nvalid_RMSE {rmse:.6f}\n")
        f.write(f"n_features {tr_x.shape[1]}\nn_train {len(tr_x)}\nn_valid {len(va_x)}\n")
    print(f"wrote {out_dir}/submission.csv and metrics.txt")


if __name__ == "__main__":
    main()
