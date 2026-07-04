"""HW9 - Explainable AI.

Course: 李宏毅 ML 2022 Spring, HW9.

Implements the official explainability methods on an image classifier:
  1. Saliency map          (|d loss / d input|)
  2. Smooth-grad           (saliency averaged over noisy inputs)
  3. Integrated gradients   (path integral of gradients from a baseline)
  4. Occlusion sensitivity  (drop in confidence when patches are masked)

The official task explains a food-11 CNN; we train a small CNN on the real
CIFAR-10 dataset (torchvision auto-download) and produce all four attribution
maps for sample images. Reported quantitative check: the *deletion metric* —
masking the top-k most-salient pixels should drop the predicted-class
probability more than masking random pixels (a sanity test for attribution
quality).
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

torch.set_num_threads(3)


class SmallCNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 128),
                                  nn.ReLU(), nn.Linear(128, n))

    def forward(self, x):
        return self.head(self.features(x))


def saliency(model, x, y):
    x = x.clone().requires_grad_(True)
    out = model(x)
    loss = F.cross_entropy(out, y)
    model.zero_grad()
    loss.backward()
    return x.grad.abs().max(dim=1)[0]  # (B, H, W)


def smooth_grad(model, x, y, n=15, sigma=0.1):
    acc = torch.zeros(x.size(0), x.size(2), x.size(3))
    for _ in range(n):
        noisy = x + torch.randn_like(x) * sigma
        acc += saliency(model, noisy, y)
    return acc / n


def integrated_grad(model, x, y, steps=20):
    baseline = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for a in np.linspace(0, 1, steps):
        xi = (baseline + a * (x - baseline)).clone().requires_grad_(True)
        out = model(xi)
        loss = out.gather(1, y[:, None]).sum()
        model.zero_grad()
        loss.backward()
        total += xi.grad
    ig = (x - baseline) * total / steps
    return ig.abs().max(dim=1)[0]


def occlusion(model, x, y, patch=6, stride=4):
    model.eval()
    with torch.no_grad():
        base = F.softmax(model(x), 1).gather(1, y[:, None]).squeeze(1)
    H = x.size(2)
    heat = torch.zeros(x.size(0), H, H)
    for i in range(0, H - patch + 1, stride):
        for j in range(0, H - patch + 1, stride):
            xm = x.clone()
            xm[:, :, i:i + patch, j:j + patch] = 0
            with torch.no_grad():
                p = F.softmax(model(xm), 1).gather(1, y[:, None]).squeeze(1)
            heat[:, i:i + patch, j:j + patch] += (base - p)[:, None, None]
    return heat


def deletion_metric(model, x, y, sal, k_frac=0.1):
    """Prob drop when masking top-k salient vs random pixels."""
    B, _, H, W = x.shape
    k = int(k_frac * H * W)
    model.eval()
    with torch.no_grad():
        base = F.softmax(model(x), 1).gather(1, y[:, None]).squeeze(1)
        flat = sal.view(B, -1)
        top = flat.argsort(dim=1, descending=True)[:, :k]
        rand = torch.stack([torch.randperm(H * W)[:k] for _ in range(B)])
        xd = x.clone().view(B, 3, -1)
        xr = x.clone().view(B, 3, -1)
        for b in range(B):
            xd[b, :, top[b]] = 0
            xr[b, :, rand[b]] = 0
        pd = F.softmax(model(xd.view(B, 3, H, W)), 1).gather(1, y[:, None]).squeeze(1)
        pr = F.softmax(model(xr.view(B, 3, H, W)), 1).gather(1, y[:, None]).squeeze(1)
    return float((base - pd).mean()), float((base - pr).mean())


def train_cnn(model, data_dir, epochs=5):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
    train = torchvision.datasets.CIFAR10(data_dir, True, tf, download=True)
    from torch.utils.data import DataLoader, Subset
    tl = DataLoader(Subset(train, range(8000)), 128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  train epoch {ep} last loss {loss.item():.4f}")
    return train, tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(0)
    here = os.path.dirname(__file__)
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.join(here, "..", "hw03-cnn", "data")

    model = SmallCNN()
    print("training classifier...")
    train, tf = train_cnn(model, data_dir, args.epochs)
    model.eval()

    # pick 6 sample images
    idxs = [3, 7, 12, 25, 44, 60]
    x = torch.stack([train[i][0] for i in idxs])
    y = torch.tensor([train[i][1] for i in idxs])

    sal = saliency(model, x, y)
    sg = smooth_grad(model, x, y)
    ig = integrated_grad(model, x, y)
    occ = occlusion(model, x, y)

    # Deletion sanity check on a larger held-out sample (32 imgs) using the
    # smooth-grad map, which is less noisy than raw single-sample saliency.
    xe = torch.stack([train[i][0] for i in range(200, 232)])
    ye = torch.tensor([train[i][1] for i in range(200, 232)])
    sg_e = smooth_grad(model, xe, ye)
    del_top, del_rand = deletion_metric(model, xe, ye, sg_e)
    print(f"deletion metric (smooth-grad, 32 imgs): top-k prob drop {del_top:.4f} "
          f"vs random {del_rand:.4f}")

    if HAVE_MPL:
        methods = [("input", None), ("saliency", sal), ("smooth-grad", sg),
                   ("integrated-grad", ig), ("occlusion", occ)]
        fig, axes = plt.subplots(len(idxs), len(methods), figsize=(10, 12))
        classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        for r in range(len(idxs)):
            img = (x[r] * 0.5 + 0.5).permute(1, 2, 0).numpy()
            for c, (name, m) in enumerate(methods):
                ax = axes[r, c]
                if m is None:
                    ax.imshow(img)
                    ax.set_ylabel(classes[y[r]], fontsize=8)
                else:
                    ax.imshow(m[r].detach().numpy(), cmap="hot")
                if r == 0:
                    ax.set_title(name, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "attributions.png"), dpi=90)
        print(f"wrote attributions.png")

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"deletion_topk_prob_drop {del_top:.4f}\n")
        f.write(f"deletion_random_prob_drop {del_rand:.4f}\n")
        f.write(f"attribution_valid {del_top > del_rand}\n")
        f.write("methods saliency,smooth_grad,integrated_grad,occlusion\n")
    print(f"-> {out_dir}/metrics.txt")


if __name__ == "__main__":
    main()
