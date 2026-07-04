"""HW14 - Life-long Learning (Elastic Weight Consolidation).

Course: 李宏毅 ML 2022 Spring, HW14.

The official task is regularisation-based continual learning: train on a sequence
of tasks without catastrophically forgetting earlier ones. We implement
**Elastic Weight Consolidation (EWC, Kirkpatrick et al. 2017)** and evaluate it
on the standard **Permuted-MNIST** benchmark built from the real MNIST dataset
(torchvision auto-download).

A sequence of T tasks is created by applying a fixed random pixel permutation to
MNIST for each task. We train sequentially on task 1, 2, ... T and after each
task measure accuracy on *every* task seen so far. We compare:

  - SGD (naive fine-tuning)  -> suffers catastrophic forgetting
  - EWC (Fisher-weighted L2 anchor to previous optima) -> retains old tasks

Reported metrics: final average accuracy across all tasks and the
backward-transfer / forgetting measure, for SGD vs EWC.
"""
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, TensorDataset

torch.set_num_threads(3)


class MLP(nn.Module):
    def __init__(self, n_in=784, n_hidden=256, n_out=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def load_mnist_flat(data_dir, n_train, n_test):
    tf = T.Compose([T.ToTensor()])
    train = torchvision.datasets.MNIST(data_dir, True, tf, download=True)
    test = torchvision.datasets.MNIST(data_dir, False, tf, download=True)
    tr = Subset(train, range(n_train))
    te = Subset(test, range(n_test))
    Xtr = torch.stack([tr[i][0] for i in range(len(tr))]).view(len(tr), -1)
    ytr = torch.tensor([tr[i][1] for i in range(len(tr))])
    Xte = torch.stack([te[i][0] for i in range(len(te))]).view(len(te), -1)
    yte = torch.tensor([te[i][1] for i in range(len(te))])
    return Xtr, ytr, Xte, yte


def make_permuted_tasks(Xtr, Xte, n_tasks, seed=0):
    rng = np.random.default_rng(seed)
    tasks = []
    for t in range(n_tasks):
        perm = torch.tensor(rng.permutation(784)) if t > 0 else torch.arange(784)
        tasks.append((Xtr[:, perm], Xte[:, perm]))
    return tasks


class EWC:
    """Accumulates a diagonal-Fisher penalty anchoring params to past optima."""

    def __init__(self, lam):
        self.lam = lam
        self.anchors = []  # list of (params_snapshot, fisher)

    def compute_fisher(self, model, X, y, bs=256, n_batches=20):
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        model.eval()
        idx = torch.randperm(len(X))[: bs * n_batches]
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            model.zero_grad()
            logp = F.log_softmax(model(X[b]), 1)
            # sample labels from the model's own predictive distribution
            samp = torch.multinomial(logp.exp(), 1).squeeze(1)
            loss = F.nll_loss(logp, samp)
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2 / (len(idx) / bs)
        snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
        self.anchors.append((snapshot, fisher))

    def penalty(self, model):
        if not self.anchors:
            return torch.tensor(0.0)
        loss = 0.0
        for snapshot, fisher in self.anchors:
            for n, p in model.named_parameters():
                loss = loss + (fisher[n] * (p - snapshot[n]) ** 2).sum()
        return self.lam * 0.5 * loss


@torch.no_grad()
def accuracy(model, X, y, bs=512):
    model.eval()
    correct = 0
    for i in range(0, len(X), bs):
        correct += (model(X[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(X)


def train_task(model, X, y, ewc, epochs, batch, lr):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch, shuffle=True)
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            loss = F.cross_entropy(model(xb), yb)
            if ewc is not None:
                loss = loss + ewc.penalty(model)
            opt.zero_grad(); loss.backward(); opt.step()


def run_continual(use_ewc, tasks, ytr, yte, args):
    model = MLP()
    ewc = EWC(args.lam) if use_ewc else None
    n_tasks = len(tasks)
    acc_matrix = np.zeros((n_tasks, n_tasks))  # [after_task, eval_task]
    for t, (Xtr_t, _) in enumerate(tasks):
        train_task(model, Xtr_t, ytr, ewc, args.epochs, args.batch, args.lr)
        if use_ewc:
            ewc.compute_fisher(model, Xtr_t, ytr)
        for e, (_, Xte_e) in enumerate(tasks[: t + 1]):
            acc_matrix[t, e] = accuracy(model, Xte_e, yte)
        seen = acc_matrix[t, : t + 1]
        print(f"  [{'EWC' if use_ewc else 'SGD'}] after task {t}: "
              f"avg over seen {seen.mean():.4f}  (per-task {np.round(seen,3)})")
    final_avg = acc_matrix[n_tasks - 1].mean()
    # forgetting: mean over tasks of (best past acc - final acc)
    forgetting = np.mean([acc_matrix[:, e].max() - acc_matrix[n_tasks - 1, e]
                          for e in range(n_tasks - 1)]) if n_tasks > 1 else 0.0
    return final_avg, forgetting, acc_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=2000.0, help="EWC strength")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-test", type=int, default=2000)
    args = ap.parse_args()
    torch.manual_seed(0)
    here = os.path.dirname(__file__)
    data_dir = os.path.join(here, "data")
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    Xtr, ytr, Xte, yte = load_mnist_flat(data_dir, args.n_train, args.n_test)
    tasks = make_permuted_tasks(Xtr, Xte, args.tasks)
    print(f"Permuted-MNIST: {args.tasks} tasks, {args.n_train} train / "
          f"{args.n_test} test each")

    print("== naive SGD (fine-tuning) ==")
    sgd_avg, sgd_forget, _ = run_continual(False, tasks, ytr, yte, args)
    print("== EWC ==")
    ewc_avg, ewc_forget, ewc_mat = run_continual(True, tasks, ytr, yte, args)

    print(f"\nfinal avg acc:  SGD {sgd_avg:.4f} | EWC {ewc_avg:.4f}")
    print(f"forgetting:     SGD {sgd_forget:.4f} | EWC {ewc_forget:.4f} "
          f"(lower is better)")

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"benchmark Permuted-MNIST\nn_tasks {args.tasks}\n")
        f.write(f"sgd_final_avg_acc {sgd_avg:.4f}\newc_final_avg_acc {ewc_avg:.4f}\n")
        f.write(f"sgd_forgetting {sgd_forget:.4f}\newc_forgetting {ewc_forget:.4f}\n")
        f.write(f"ewc_lambda {args.lam}\n")
    np.save(os.path.join(out_dir, "ewc_acc_matrix.npy"), ewc_mat)
    print(f"-> {out_dir}/metrics.txt")


if __name__ == "__main__":
    main()
