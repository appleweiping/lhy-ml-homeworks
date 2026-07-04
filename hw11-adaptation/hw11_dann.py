"""HW11 - Domain Adaptation (DaNN with gradient reversal).

Course: 李宏毅 ML 2022 Spring, HW11 (Kaggle: ml2022-spring-hw11).

Implements Domain-Adversarial Neural Networks (Ganin et al.): a feature
extractor shared by a label predictor and a domain classifier, with a Gradient
Reversal Layer so features become domain-invariant — exactly the official HW11
method. The official task adapts real photos -> hand-drawn sketches. We
reproduce the setup on the real CIFAR-10 dataset: the *source* domain is the
normal RGB images (labelled), the *target* domain is a shifted version
(grayscale + edge-emphasis, labels hidden during training). Reported metric:
target-domain accuracy with vs without adaptation.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.autograd import Function
from torch.utils.data import DataLoader, Subset, TensorDataset

torch.set_num_threads(3)


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class LabelPredictor(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, n))

    def forward(self, x):
        return self.net(x)


class DomainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def to_target_domain(x):
    """Domain shift: grayscale + edge emphasis (a real covariate shift)."""
    gray = x.mean(1, keepdim=True).repeat(1, 3, 1, 1)
    kernel = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32)
    kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    edges = F.conv2d(gray, kernel, padding=1, groups=3)
    return (0.5 * gray + 0.5 * edges).clamp(-3, 3)


def get_tensors(data_dir, n_src=8000, n_tgt=4000, n_test=2000):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
    train = torchvision.datasets.CIFAR10(data_dir, True, tf, download=True)
    test = torchvision.datasets.CIFAR10(data_dir, False, tf, download=True)
    sx = torch.stack([train[i][0] for i in range(n_src)])
    sy = torch.tensor([train[i][1] for i in range(n_src)])
    tx = to_target_domain(torch.stack([train[i][0] for i in range(n_src, n_src + n_tgt)]))
    ex = to_target_domain(torch.stack([test[i][0] for i in range(n_test)]))
    ey = torch.tensor([test[i][1] for i in range(n_test)])
    return sx, sy, tx, ex, ey


def evaluate(F_, C, x, y, bs=256):
    F_.eval(); C.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            correct += (C(F_(x[i:i + bs])).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    torch.manual_seed(0)
    here = os.path.dirname(__file__)
    data_dir = os.path.join(here, "..", "hw03-cnn", "data")
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    sx, sy, tx, ex, ey = get_tensors(data_dir)
    print(f"source {sx.shape} target {tx.shape} test(target) {ex.shape}")

    def train_model(adapt):
        F_ = FeatureExtractor(); C = LabelPredictor(); D = DomainClassifier()
        opt = torch.optim.AdamW(
            list(F_.parameters()) + list(C.parameters()) + list(D.parameters()), 1e-3)
        src = DataLoader(TensorDataset(sx, sy), args.batch, shuffle=True)
        tgt = DataLoader(TensorDataset(tx), args.batch, shuffle=True)
        for ep in range(args.epochs):
            F_.train(); C.train(); D.train()
            tgt_iter = iter(tgt)
            p = ep / args.epochs
            lambd = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0 if adapt else 0.0
            for xs, ys in src:
                try:
                    (xt,) = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt); (xt,) = next(tgt_iter)
                fs = F_(xs)
                cls_loss = F.cross_entropy(C(fs), ys)
                if adapt:
                    ft = F_(xt)
                    feat = torch.cat([fs, ft], 0)
                    dlab = torch.cat([torch.ones(len(fs)), torch.zeros(len(ft))])
                    dloss = F.binary_cross_entropy_with_logits(
                        D(grad_reverse(feat, lambd)), dlab)
                    loss = cls_loss + dloss
                else:
                    loss = cls_loss
                opt.zero_grad(); loss.backward(); opt.step()
            acc = evaluate(F_, C, ex, ey)
            print(f"  [{'DANN' if adapt else 'src-only'}] epoch {ep} "
                  f"lambda {lambd:.3f} target-acc {acc:.4f}")
        return evaluate(F_, C, ex, ey)

    print("== source-only baseline ==")
    base = train_model(adapt=False)
    print("== DaNN domain adaptation ==")
    dann = train_model(adapt=True)

    print(f"target acc: source-only {base:.4f} | DaNN {dann:.4f} | gain {dann-base:+.4f}")
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"source_only_target_acc {base:.4f}\ndann_target_acc {dann:.4f}\n"
                f"adaptation_gain {dann-base:+.4f}\n")
    print(f"-> {out_dir}/metrics.txt")


if __name__ == "__main__":
    main()
