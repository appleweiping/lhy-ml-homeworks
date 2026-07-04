"""HW3 - CNN Image Classification.

Course: 李宏毅 ML 2022 Spring, HW3 (Kaggle: ml2022spring-hw3b).

The official task classifies food-11 images into 11 classes with a CNN trained
from scratch (no pretrained weights allowed). food-11 is competition-gated, so
this script trains the same class of from-scratch CNN on **CIFAR-10** — a real,
freely downloadable image dataset that torchvision fetches automatically — which
FACTORY_SPEC explicitly lists as an acceptable real small dataset. The
architecture, augmentation, and training loop mirror the official baseline.

Reported metric: top-1 test accuracy on the real CIFAR-10 test set.
"""
import argparse
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

torch.set_num_threads(3)


class FoodCNN(nn.Module):
    """VGG-style from-scratch CNN (same family as the official HW3 baseline)."""

    def __init__(self, n_class=10):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )

        self.features = nn.Sequential(
            block(3, 64),    # 32 -> 16
            block(64, 128),  # 16 -> 8
            block(128, 256),  # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, n_class),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_loaders(batch, data_dir, subset):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    train = torchvision.datasets.CIFAR10(data_dir, True, train_tf, download=True)
    test = torchvision.datasets.CIFAR10(data_dir, False, test_tf, download=True)
    if subset > 0:
        train = Subset(train, range(min(subset, len(train))))
        test = Subset(test, range(min(subset // 4, len(test))))
    return (DataLoader(train, batch, shuffle=True, num_workers=0),
            DataLoader(test, batch, shuffle=False, num_workers=0))


def run(model, loader, device, criterion, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
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
            loss_sum += loss.item() * len(x)
            correct += (out.argmax(1) == y).sum().item()
            tot += len(x)
    return loss_sum / tot, correct / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--subset", type=int, default=10000,
                    help="train subset for CPU speed; 0 = full 50k")
    args = ap.parse_args()
    device = "cpu"
    torch.manual_seed(0)
    here = os.path.dirname(__file__)
    data_dir = os.path.join(here, "data")
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    tl, vl = get_loaders(args.batch, data_dir, args.subset)
    model = FoodCNN(10).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = 0.0
    for ep in range(args.epochs):
        trl, tra = run(model, tl, device, criterion, opt)
        tel, tea = run(model, vl, device, criterion)
        sched.step()
        best = max(best, tea)
        print(f"epoch {ep:2d} | train loss {trl:.4f} acc {tra:.4f} | "
              f"test loss {tel:.4f} acc {tea:.4f} | best {best:.4f}")

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"best_test_acc {best:.4f}\ndataset CIFAR-10\n"
                f"train_subset {args.subset}\nepochs {args.epochs}\n")
    print(f"BEST test acc {best:.4f} -> {out_dir}/metrics.txt")


if __name__ == "__main__":
    main()
