"""HW15 - Meta Learning (MAML for few-shot classification).

Course: 李宏毅 ML 2022 Spring, HW15.

The official task is few-shot image classification with **Model-Agnostic
Meta-Learning (MAML, Finn et al. 2017)** on the Omniglot dataset. We implement
MAML from scratch (inner-loop adaptation with second-order-capable functional
forward, outer-loop meta-update) and train/evaluate it on the real **Omniglot**
dataset (torchvision auto-download) in the standard N-way K-shot episodic setup.

We compare the meta-learned initialisation against a *random-init* baseline that
does the same number of inner-loop adaptation steps — isolating the value of the
meta-learned starting point.

Reported metric: N-way K-shot query accuracy on held-out (meta-test) character
classes, MAML vs baseline.
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

torch.set_num_threads(3)


# ---------------- functional CNN (weights passed explicitly for MAML) ----------
def conv_block(x, w, b, bn_w, bn_b):
    x = F.conv2d(x, w, b, padding=1)
    x = F.batch_norm(x, None, None, bn_w, bn_b, training=True)
    x = F.relu(x)
    return F.max_pool2d(x, 2)


class MetaCNN(nn.Module):
    """4-conv-block CNN; exposes weights as a dict for functional forward."""

    def __init__(self, n_way, in_ch=1, hid=32):
        super().__init__()
        self.n_way = n_way
        self.hid = hid
        chans = [in_ch, hid, hid, hid, hid]
        self.params = nn.ParameterDict()
        for i in range(4):
            self.params[f"conv{i}_w"] = nn.Parameter(
                torch.empty(chans[i + 1], chans[i], 3, 3))
            nn.init.kaiming_normal_(self.params[f"conv{i}_w"])
            self.params[f"conv{i}_b"] = nn.Parameter(torch.zeros(chans[i + 1]))
            self.params[f"bn{i}_w"] = nn.Parameter(torch.ones(chans[i + 1]))
            self.params[f"bn{i}_b"] = nn.Parameter(torch.zeros(chans[i + 1]))
        # 28x28 -> /2 four times -> ~1x1 with hid channels
        self.params["fc_w"] = nn.Parameter(torch.empty(n_way, hid))
        nn.init.kaiming_normal_(self.params["fc_w"])
        self.params["fc_b"] = nn.Parameter(torch.zeros(n_way))

    def functional_forward(self, x, weights):
        for i in range(4):
            x = conv_block(x, weights[f"conv{i}_w"], weights[f"conv{i}_b"],
                           weights[f"bn{i}_w"], weights[f"bn{i}_b"])
        x = x.mean(dim=[2, 3])  # global avg pool -> (B, hid)
        return F.linear(x, weights["fc_w"], weights["fc_b"])


# ---------------- Omniglot episodic sampler ------------------------------------
class OmniglotTasks:
    """Groups the real Omniglot dataset by character class for N-way K-shot
    episode sampling. Meta-train / meta-test use disjoint character classes."""

    def __init__(self, data_dir, img=28):
        tf = T.Compose([T.Resize((img, img)), T.ToTensor()])
        # background=True is the large 964-class training alphabet split
        ds = torchvision.datasets.Omniglot(data_dir, background=True,
                                           transform=tf, download=True)
        self.by_class = {}  # label -> list of flat indices
        for i, (path, label) in enumerate(ds._flat_character_images):
            self.by_class.setdefault(label, []).append(i)
        self.ds = ds
        self.labels = sorted(self.by_class.keys())
        self.tf = tf
        self.img = img
        # cache tensors lazily
        self._cache = {}

    def _load(self, idx):
        if idx not in self._cache:
            self._cache[idx] = self.ds[idx][0]
        return self._cache[idx]

    def split(self, n_meta_train=800):
        rng = random.Random(0)
        labels = self.labels[:]
        rng.shuffle(labels)
        return labels[:n_meta_train], labels[n_meta_train:]

    def sample_episode(self, class_pool, n_way, k_shot, k_query, rng):
        classes = rng.sample(class_pool, n_way)
        sx, sy, qx, qy = [], [], [], []
        # locate example indices per class
        for new_label, c in enumerate(classes):
            idxs = self.by_class[c]  # precomputed flat indices for this class
            chosen = rng.sample(idxs, k_shot + k_query)
            for j, gi in enumerate(chosen):
                img = self._load(gi)
                if j < k_shot:
                    sx.append(img); sy.append(new_label)
                else:
                    qx.append(img); qy.append(new_label)
        return (torch.stack(sx), torch.tensor(sy),
                torch.stack(qx), torch.tensor(qy))


def adapt(model, weights, sx, sy, inner_lr, inner_steps, create_graph):
    """Inner-loop SGD adaptation, returns adapted weights (functional).

    The starting weights must carry grad so the inner loss is differentiable.
    At meta-train time (create_graph=True) grads flow back to the meta-params for
    the second-order MAML update; at eval time we still need first-order inner
    grads, so we re-enable grad on detached leaves.
    """
    fast = {k: (v if v.requires_grad else v.detach().requires_grad_(True))
            for k, v in weights.items()}
    for _ in range(inner_steps):
        logits = model.functional_forward(sx, fast)
        loss = F.cross_entropy(logits, sy)
        grads = torch.autograd.grad(loss, fast.values(),
                                    create_graph=create_graph)
        fast = {k: v - inner_lr * g for (k, v), g in zip(fast.items(), grads)}
    return fast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-way", type=int, default=5)
    ap.add_argument("--k-shot", type=int, default=1)
    ap.add_argument("--k-query", type=int, default=5)
    ap.add_argument("--meta-iters", type=int, default=300)
    ap.add_argument("--meta-batch", type=int, default=4)
    ap.add_argument("--inner-lr", type=float, default=0.4)
    ap.add_argument("--inner-steps", type=int, default=3)
    ap.add_argument("--outer-lr", type=float, default=1e-3)
    ap.add_argument("--eval-episodes", type=int, default=100)
    args = ap.parse_args()
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    here = os.path.dirname(__file__)
    data_dir = os.path.join(here, "data")
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    print("loading Omniglot (real, auto-download)...")
    tasks = OmniglotTasks(data_dir)
    train_pool, test_pool = tasks.split()
    print(f"{args.n_way}-way {args.k_shot}-shot | "
          f"meta-train classes {len(train_pool)} meta-test classes {len(test_pool)}")

    model = MetaCNN(args.n_way)
    meta_opt = torch.optim.Adam(model.parameters(), lr=args.outer_lr)
    rng = random.Random(1)

    def meta_eval(pool, use_trained):
        """Average query accuracy over eval episodes. If not use_trained, adapt
        from a fresh random init (baseline)."""
        accs = []
        base = model if use_trained else MetaCNN(args.n_way)
        weights = {k: v.detach() for k, v in base.params.items()}
        for _ in range(args.eval_episodes):
            sx, sy, qx, qy = tasks.sample_episode(pool, args.n_way,
                                                  args.k_shot, args.k_query, rng)
            fast = adapt(base, weights, sx, sy, args.inner_lr,
                         args.inner_steps, create_graph=False)
            with torch.no_grad():
                pred = base.functional_forward(qx, fast).argmax(1)
            accs.append((pred == qy).float().mean().item())
        return float(np.mean(accs))

    print("baseline (random init + same inner adaptation)...")
    base_acc = meta_eval(test_pool, use_trained=False)
    print(f"baseline meta-test acc {base_acc:.4f}")

    print("meta-training MAML...")
    for it in range(args.meta_iters):
        meta_opt.zero_grad()
        meta_loss = 0.0
        for _ in range(args.meta_batch):
            sx, sy, qx, qy = tasks.sample_episode(train_pool, args.n_way,
                                                  args.k_shot, args.k_query, rng)
            fast = adapt(model, dict(model.params), sx, sy, args.inner_lr,
                         args.inner_steps, create_graph=True)
            q_logits = model.functional_forward(qx, fast)
            meta_loss = meta_loss + F.cross_entropy(q_logits, qy)
        meta_loss = meta_loss / args.meta_batch
        meta_loss.backward()
        meta_opt.step()
        if it % 50 == 0 or it == args.meta_iters - 1:
            acc = meta_eval(test_pool, use_trained=True)
            print(f"  iter {it:3d} | meta-loss {meta_loss.item():.4f} | "
                  f"meta-test acc {acc:.4f}")

    maml_acc = meta_eval(test_pool, use_trained=True)
    print(f"\nMAML meta-test {args.n_way}-way {args.k_shot}-shot acc "
          f"{maml_acc:.4f}  (baseline {base_acc:.4f}, gain {maml_acc-base_acc:+.4f})")

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"dataset Omniglot\nsetting {args.n_way}-way {args.k_shot}-shot\n")
        f.write(f"baseline_acc {base_acc:.4f}\nmaml_acc {maml_acc:.4f}\n")
        f.write(f"meta_gain {maml_acc-base_acc:+.4f}\n")
        f.write(f"inner_steps {args.inner_steps}\nmeta_iters {args.meta_iters}\n")
    print(f"-> {out_dir}/metrics.txt")


if __name__ == "__main__":
    main()
