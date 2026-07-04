"""HW6 - GAN (image generation).

Course: 李宏毅 ML 2022 Spring, HW6.

Implements a DCGAN (the official architecture family: strided-conv generator &
discriminator, BatchNorm, the non-saturating GAN loss) and trains it to generate
images. The official task generates anime faces from the Crypko dataset, which is
gated and needs many GPU-hours.

To train a *real* GAN on real data at CPU scale, we learn the distribution of the
real **MNIST** handwritten-digit dataset (torchvision auto-download), 28x28
grayscale. We save generated sample grids across training and track the
discriminator / generator losses. As a quantitative signal we report the
Frechet-like feature statistic gap between real and generated batches (mean/std
of pixel intensities and of a fixed random-projection feature) — a proxy that
shrinks as the generator matches the data distribution.

Reported artefacts: generated digit grids (results/samples.png), loss curves,
and the real-vs-generated distribution-gap metrics.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

torch.set_num_threads(3)
IMG = 32  # upsample MNIST to 32 for clean strided-conv arithmetic


class Generator(nn.Module):
    def __init__(self, z=64, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z, ngf * 4, 4, 1, 0), nn.BatchNorm2d(ngf * 4), nn.ReLU(True),   # 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),  # 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1), nn.BatchNorm2d(ngf), nn.ReLU(True),  # 16
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1), nn.Tanh(),  # 32
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),  # 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),  # 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),  # 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0),  # 1
        )

    def forward(self, x):
        return self.net(x).view(-1)


def get_real_loader(data_dir, batch, n):
    tf = T.Compose([T.Resize(IMG), T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    ds = torchvision.datasets.MNIST(data_dir, True, tf, download=True)
    if n > 0:
        ds = Subset(ds, range(min(n, len(ds))))
    return DataLoader(ds, batch, shuffle=True, drop_last=True)


def dist_gap(real, gen, proj):
    """|mean| and |std| gap on pixels + on a fixed random projection feature."""
    r = real.reshape(real.shape[0], -1)
    g = gen.reshape(gen.shape[0], -1)
    rf = r @ proj; gf = g @ proj
    return (float(abs(r.mean() - g.mean())), float(abs(r.std() - g.std())),
            float(abs(rf.mean() - gf.mean())), float(abs(rf.std() - gf.std())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--z", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-train", type=int, default=12000, help="0 = full 60k")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = "cpu"
    here = os.path.dirname(__file__)
    data_dir = os.path.join(here, "data")
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    loader = get_real_loader(data_dir, args.batch, args.n_train)
    G = Generator(args.z).to(device)
    D = Discriminator().to(device)
    bce = nn.BCEWithLogitsLoss()
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    fixed_z = torch.randn(64, args.z, 1, 1)

    d_losses, g_losses = [], []
    step = 0
    for ep in range(args.epochs):
        for real, _ in loader:
            real = real.to(device)
            bs = real.size(0)
            # --- train D ---
            z = torch.randn(bs, args.z, 1, 1, device=device)
            fake = G(z)
            d_real = D(real)
            d_fake = D(fake.detach())
            lossD = bce(d_real, torch.ones_like(d_real) * 0.9) + bce(d_fake, torch.zeros_like(d_fake))
            optD.zero_grad(); lossD.backward(); optD.step()
            # --- train G ---
            d_fake2 = D(fake)
            lossG = bce(d_fake2, torch.ones_like(d_fake2))
            optG.zero_grad(); lossG.backward(); optG.step()
            d_losses.append(lossD.item()); g_losses.append(lossG.item())
            step += 1
            if step % 100 == 0:
                print(f"epoch {ep} step {step:4d} | D {lossD.item():.4f} | G {lossG.item():.4f}")

    # ---- quantitative signal: distribution stat gap ----
    G.eval()
    rng = np.random.default_rng(0)
    proj = torch.tensor(rng.standard_normal((IMG * IMG, 32)).astype(np.float32))
    with torch.no_grad():
        gen = G(torch.randn(512, args.z, 1, 1, device=device)).cpu()
    real_big = torch.cat([b for b, _ in
                          [next(iter(loader))]], 0)[:512].cpu()
    gm, gs, fm, fs = dist_gap(real_big, gen, proj)
    print(f"pixel |mean gap| {gm:.4f} |std gap| {gs:.4f} | "
          f"feat |mean gap| {fm:.4f} |std gap| {fs:.4f}")

    # ---- save sample grid ----
    with torch.no_grad():
        samples = ((G(fixed_z).cpu() + 1) / 2).clamp(0, 1).numpy()
    if HAVE_MPL:
        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i, 0], cmap="gray")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "samples.png"), dpi=80)
        plt.figure(figsize=(6, 4))
        plt.plot(d_losses, label="D", alpha=0.6)
        plt.plot(g_losses, label="G", alpha=0.6)
        plt.legend(); plt.xlabel("step"); plt.ylabel("loss"); plt.title("DCGAN losses (MNIST)")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "losses.png"), dpi=80)

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"dataset MNIST\nfinal_D_loss {d_losses[-1]:.4f}\nfinal_G_loss {g_losses[-1]:.4f}\n")
        f.write(f"pixel_mean_gap {gm:.4f}\npixel_std_gap {gs:.4f}\n")
        f.write(f"feat_mean_gap {fm:.4f}\nfeat_std_gap {fs:.4f}\n")
        f.write(f"train_steps {step}\nepochs {args.epochs}\n")
    print(f"wrote samples.png, losses.png, metrics.txt -> {out_dir}")


if __name__ == "__main__":
    main()
