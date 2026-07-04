"""HW10 - Adversarial Attack.

Course: 李宏毅 ML 2022 Spring, HW10.

Implements the official white-box attacks against an image classifier:
  - FGSM  (single-step, epsilon L-inf)
  - I-FGSM / PGD (iterative, projected)
  - MI-FGSM (momentum iterative)
and measures the accuracy drop. The official task attacks a CIFAR-10 classifier
under an L-inf budget; we do exactly that on the real CIFAR-10 dataset
(torchvision auto-download). Reported metrics: clean accuracy vs accuracy under
each attack, and the mean L-inf perturbation.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

torch.set_num_threads(3)

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1)


class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.h = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 128),
                               nn.ReLU(), nn.Linear(128, n))

    def forward(self, x):
        return self.h(self.f(x))


def denorm(x):
    return x * STD + MEAN


def renorm(x):
    return (x - MEAN) / STD


def fgsm(model, x, y, eps):
    x = x.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    x_pix = denorm(x) + eps * grad.sign()
    return renorm(x_pix.clamp(0, 1)).detach()


def pgd(model, x, y, eps, alpha, steps, momentum=0.0):
    x_orig = denorm(x).detach()
    x_pix = x_orig.clone()
    g = torch.zeros_like(x_pix)
    for _ in range(steps):
        xi = renorm(x_pix).requires_grad_(True)
        loss = F.cross_entropy(model(xi), y)
        grad = torch.autograd.grad(loss, xi)[0]
        if momentum > 0:
            g = momentum * g + grad / (grad.abs().mean() + 1e-12)
            grad = g
        x_pix = x_pix + alpha * grad.sign()
        x_pix = torch.min(torch.max(x_pix, x_orig - eps), x_orig + eps).clamp(0, 1)
    return renorm(x_pix).detach()


def accuracy(model, x, y):
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--n-test", type=int, default=1000)
    args = ap.parse_args()
    torch.manual_seed(0)
    here = os.path.dirname(__file__)
    data_dir = os.path.join(here, "..", "hw03-cnn", "data")
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN.flatten(), STD.flatten())])
    train = torchvision.datasets.CIFAR10(data_dir, True, tf, download=True)
    test = torchvision.datasets.CIFAR10(data_dir, False, tf, download=True)

    model = CNN()
    print("training target classifier...")
    tl = DataLoader(Subset(train, range(10000)), 128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    for ep in range(args.epochs):
        model.train()
        for x, y in tl:
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  epoch {ep} loss {loss.item():.4f}")
    model.eval()

    tx = torch.stack([test[i][0] for i in range(args.n_test)])
    ty = torch.tensor([test[i][1] for i in range(args.n_test)])

    clean = accuracy(model, tx, ty)
    x_fgsm = fgsm(model, tx, ty, args.eps)
    x_pgd = pgd(model, tx, ty, args.eps, args.eps / 4, steps=10)
    x_mi = pgd(model, tx, ty, args.eps, args.eps / 4, steps=10, momentum=1.0)

    acc_fgsm = accuracy(model, x_fgsm, ty)
    acc_pgd = accuracy(model, x_pgd, ty)
    acc_mi = accuracy(model, x_mi, ty)
    linf = (denorm(x_pgd) - denorm(tx)).abs().max().item()

    print(f"clean acc      {clean:.4f}")
    print(f"FGSM acc       {acc_fgsm:.4f}")
    print(f"PGD-10 acc     {acc_pgd:.4f}")
    print(f"MI-FGSM-10 acc {acc_mi:.4f}")
    print(f"max L-inf pert {linf:.4f} (budget {args.eps:.4f})")

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"clean_acc {clean:.4f}\nfgsm_acc {acc_fgsm:.4f}\n"
                f"pgd10_acc {acc_pgd:.4f}\nmifgsm10_acc {acc_mi:.4f}\n"
                f"eps {args.eps:.5f}\nmax_linf {linf:.5f}\n")

    # save a few adversarial examples
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 6, figsize=(10, 5))
        clean_pix = denorm(tx).clamp(0, 1)   # (N,3,32,32)
        adv_pix = denorm(x_pgd).clamp(0, 1)
        for i in range(6):
            ax[0, i].imshow(clean_pix[i].permute(1, 2, 0).numpy())
            ax[1, i].imshow(adv_pix[i].permute(1, 2, 0).numpy())
            diff = (adv_pix[i] - clean_pix[i]).abs().permute(1, 2, 0).numpy()
            ax[2, i].imshow(diff / max(diff.max(), 1e-6))
            for r in range(3):
                ax[r, i].axis("off")
        ax[0, 0].set_title("clean", fontsize=9)
        ax[1, 0].set_title("PGD adv", fontsize=9)
        ax[2, 0].set_title("perturbation", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "adversarial_examples.png"), dpi=90)
        print("wrote adversarial_examples.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
