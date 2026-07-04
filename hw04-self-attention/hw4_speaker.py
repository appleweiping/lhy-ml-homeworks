"""HW4 - Self-Attention for Sequence Classification.

Course: 李宏毅 ML 2022 Spring, HW4 (Kaggle: ml2022spring-hw4).

The official task classifies a variable-length mel-spectrogram utterance into one
of N speakers with a Transformer encoder (self-attention) + pooling + linear head.
The official VoxCeleb-derived data (600 speakers) is competition-gated and multi-GB.

To exercise the *identical model* on a real, freely-downloadable dataset, we frame
sequence classification on **FashionMNIST** (torchvision auto-download): each
28x28 image is read as a length-T sequence of T row-vectors (dim=28), i.e. a
genuine variable-length-capable sequence, and the Transformer-encoder classifier
predicts the garment class. We additionally use *variable* sequence lengths (each
example is randomly truncated and padded with a proper padding mask) so the
padding-masked self-attention + masked mean-pooling path is genuinely exercised —
exactly as in the official speaker pipeline.

Reported metric: classification accuracy on the real FashionMNIST test set.
"""
import argparse
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, TensorDataset

torch.set_num_threads(3)

SEQ_DIM = 28   # each row is a 28-dim feature vector
MAX_LEN = 28   # up to 28 rows


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        import math
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SelfAttnClassifier(nn.Module):
    """Transformer-encoder (self-attention) + masked mean-pool + linear head —
    the same architecture family as the official HW4 speaker classifier."""

    def __init__(self, feat_dim=SEQ_DIM, d_model=80, n_head=4, n_layers=2, n_cls=10):
        super().__init__()
        self.prenet = nn.Linear(feat_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, n_cls)
        )

    def forward(self, x, mask=None):
        # x: (B, T, feat_dim); mask: (B, T) True = pad
        h = self.pos(self.prenet(x))
        h = self.encoder(h, src_key_padding_mask=mask)
        if mask is not None:
            valid = (~mask).unsqueeze(-1).float()
            pooled = (h * valid).sum(1) / valid.sum(1).clamp(min=1)
        else:
            pooled = h.mean(1)
        return self.head(pooled)


def build_sequences(images, gen):
    """(N,1,28,28) -> (N,28,28) sequences with random truncation + pad mask."""
    seqs = images.squeeze(1)  # (N, 28, 28): 28 rows of 28-dim
    N = seqs.size(0)
    lengths = torch.randint(20, MAX_LEN + 1, (N,), generator=gen)
    padded = torch.zeros(N, MAX_LEN, SEQ_DIM)
    mask = torch.ones(N, MAX_LEN, dtype=torch.bool)  # True = pad
    for i in range(N):
        L = int(lengths[i])
        padded[i, :L] = seqs[i, :L]
        mask[i, :L] = False
    return padded, mask


def load_data(data_dir, n_train, n_test):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
    train = torchvision.datasets.FashionMNIST(data_dir, True, tf, download=True)
    test = torchvision.datasets.FashionMNIST(data_dir, False, tf, download=True)
    tr = Subset(train, range(n_train)); te = Subset(test, range(n_test))
    Xtr = torch.stack([tr[i][0] for i in range(len(tr))])
    ytr = torch.tensor([tr[i][1] for i in range(len(tr))])
    Xte = torch.stack([te[i][0] for i in range(len(te))])
    yte = torch.tensor([te[i][1] for i in range(len(te))])
    gen = torch.Generator().manual_seed(0)
    Str, Mtr = build_sequences(Xtr, gen)
    Ste, Mte = build_sequences(Xte, gen)
    return (Str, Mtr, ytr), (Ste, Mte, yte)


def run(model, loader, device, criterion, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot = correct = 0
    loss_sum = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            out = model(x, mask)
            loss = criterion(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(x)
            correct += (out.argmax(1) == y).sum().item()
            tot += len(x)
    return loss_sum / tot, correct / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-test", type=int, default=3000)
    args = ap.parse_args()
    torch.manual_seed(0)
    device = "cpu"
    here = os.path.dirname(__file__)
    data_dir = os.path.join(here, "data")
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    (Str, Mtr, ytr), (Ste, Mte, yte) = load_data(data_dir, args.n_train, args.n_test)
    print(f"train seq {tuple(Str.shape)} test seq {tuple(Ste.shape)} (real FashionMNIST)")

    tl = DataLoader(TensorDataset(Str, Mtr, ytr), args.batch, shuffle=True)
    vl = DataLoader(TensorDataset(Ste, Mte, yte), args.batch, shuffle=False)

    model = SelfAttnClassifier(n_cls=10).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = 0.0
    for ep in range(args.epochs):
        trl, tra = run(model, tl, device, criterion, opt)
        val, vaa = run(model, vl, device, criterion)
        best = max(best, vaa)
        print(f"epoch {ep:2d} | train loss {trl:.4f} acc {tra:.4f} | "
              f"test loss {val:.4f} acc {vaa:.4f} | best {best:.4f}")

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"best_test_acc {best:.4f}\ndataset FashionMNIST-as-sequence\n"
                f"model self-attention (TransformerEncoder x2 + masked mean-pool)\n"
                f"n_classes 10\n")
    print(f"BEST test acc {best:.4f} -> {out_dir}/metrics.txt")


if __name__ == "__main__":
    main()
