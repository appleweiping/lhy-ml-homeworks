"""HW13 - Network Compression (Knowledge Distillation + Pruning).

Course: 李宏毅 ML 2022 Spring, HW13 (Kaggle: ml2022spring-hw13).

The official task compresses a large food-11 classifier into a tiny student under
a parameter budget, using knowledge distillation (KD), architecture design, and
pruning. food-11 is competition-gated, so we run the *identical compression
pipeline* on the real FashionMNIST dataset (torchvision auto-download):

  1. Train a large *teacher* CNN.
  2. Distill it into a small *student* with KL(soft-teacher || soft-student) +
     hard-label CE (Hinton KD), and compare against a student trained from
     scratch with hard labels only.
  3. Apply global L1 unstructured *pruning* to the student and re-measure accuracy
     and the real (nonzero) parameter count.

Reported metrics: teacher / student-scratch / student-KD test accuracy, the
compression ratio (teacher params / student params), and post-pruning accuracy
at several sparsity levels.
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

torch.set_num_threads(3)


class TeacherCNN(nn.Module):
    """Large teacher: 3 conv blocks (32->64->128) + wide FC head."""

    def __init__(self, n=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 3 * 3, 256),
                                  nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n))

    def forward(self, x):
        return self.head(self.features(x))


class StudentCNN(nn.Module):
    """Tiny student: depthwise-separable convs + narrow head (few params)."""

    def __init__(self, n=10):
        super().__init__()

        def dws(i, o):
            return nn.Sequential(
                nn.Conv2d(i, i, 3, 1, 1, groups=i), nn.Conv2d(i, o, 1),
                nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(dws(1, 16), dws(16, 32), dws(32, 48))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(48 * 3 * 3, n))

    def forward(self, x):
        return self.head(self.features(x))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def count_nonzero(m):
    total = nonzero = 0
    for p in m.parameters():
        total += p.numel()
        nonzero += int((p != 0).sum())
    return nonzero, total


def get_loaders(data_dir, batch, subset):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
    train = torchvision.datasets.FashionMNIST(data_dir, True, tf, download=True)
    test = torchvision.datasets.FashionMNIST(data_dir, False, tf, download=True)
    if subset > 0:
        train = Subset(train, range(min(subset, len(train))))
    return (DataLoader(train, batch, shuffle=True),
            DataLoader(test, 256, shuffle=False))


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = tot = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        tot += len(y)
    return correct / tot


def train_hard(model, tl, epochs, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr)
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_kd(student, teacher, tl, epochs, T_temp=4.0, alpha=0.7, lr=1e-3):
    """Hinton KD: alpha * T^2 * KL(soft_t || soft_s) + (1-alpha) * CE(hard)."""
    teacher.eval()
    opt = torch.optim.AdamW(student.parameters(), lr)
    for ep in range(epochs):
        student.train()
        for x, y in tl:
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            soft_loss = F.kl_div(
                F.log_softmax(s_logits / T_temp, 1),
                F.softmax(t_logits / T_temp, 1),
                reduction="batchmean") * (T_temp ** 2)
            hard_loss = F.cross_entropy(s_logits, y)
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
            opt.zero_grad(); loss.backward(); opt.step()
    return student


def prune_and_eval(model, test_loader, amount):
    """Global L1 unstructured pruning of all conv/linear weights."""
    import copy
    m = copy.deepcopy(model)
    params = [(mod, "weight") for mod in m.modules()
              if isinstance(mod, (nn.Conv2d, nn.Linear))]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    for mod, name in params:
        prune.remove(mod, name)  # bake the mask into the weights
    nz, tot = count_nonzero(m)
    return accuracy(m, test_loader), nz / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4, help="student epochs")
    ap.add_argument("--teacher-epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--subset", type=int, default=15000, help="0 = full 60k")
    args = ap.parse_args()
    torch.manual_seed(0)
    here = os.path.dirname(__file__)
    data_dir = os.path.join(here, "data")
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    tl, test_loader = get_loaders(data_dir, args.batch, args.subset)

    teacher = TeacherCNN()
    student_scratch = StudentCNN()
    student_kd = StudentCNN()
    tp, sp = count_params(teacher), count_params(student_scratch)
    print(f"teacher params {tp:,} | student params {sp:,} | ratio {tp/sp:.1f}x")

    # Train the teacher longer so it is a genuinely stronger "expert" whose soft
    # targets carry useful dark knowledge; students get a shorter budget (the
    # realistic compression setting where the small net is cheap to fit).
    print(f"training teacher ({args.teacher_epochs} epochs)...")
    train_hard(teacher, tl, args.teacher_epochs)
    teacher_acc = accuracy(teacher, test_loader)
    print(f"teacher test acc {teacher_acc:.4f}")

    print(f"training student from scratch ({args.epochs} epochs, hard labels)...")
    train_hard(student_scratch, tl, args.epochs)
    scratch_acc = accuracy(student_scratch, test_loader)
    print(f"student-scratch test acc {scratch_acc:.4f}")

    print(f"training student via knowledge distillation ({args.epochs} epochs)...")
    train_kd(student_kd, teacher, tl, args.epochs)
    kd_acc = accuracy(student_kd, test_loader)
    print(f"student-KD test acc {kd_acc:.4f}  (KD gain {kd_acc-scratch_acc:+.4f})")

    print("pruning the KD student...")
    prune_results = {}
    for amt in (0.3, 0.5, 0.7, 0.9):
        acc, density = prune_and_eval(student_kd, test_loader, amt)
        prune_results[amt] = (acc, density)
        print(f"  prune {int(amt*100)}% -> density {density:.3f} acc {acc:.4f}")

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"dataset FashionMNIST\n")
        f.write(f"teacher_params {tp}\nstudent_params {sp}\n")
        f.write(f"compression_ratio {tp/sp:.2f}\n")
        f.write(f"teacher_acc {teacher_acc:.4f}\n")
        f.write(f"student_scratch_acc {scratch_acc:.4f}\n")
        f.write(f"student_kd_acc {kd_acc:.4f}\n")
        f.write(f"kd_gain {kd_acc-scratch_acc:+.4f}\n")
        for amt, (acc, dens) in prune_results.items():
            f.write(f"prune_{int(amt*100)}pct_acc {acc:.4f} density {dens:.3f}\n")
    print(f"-> {out_dir}/metrics.txt")


if __name__ == "__main__":
    main()
