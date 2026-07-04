# HW15 — Meta Learning (MAML)

Few-shot image classification with **Model-Agnostic Meta-Learning** (MAML, Finn
et al. 2017) on the real **Omniglot** dataset (the official HW15 benchmark).

MAML is implemented from scratch: a functional-forward 4-conv-block CNN, an
inner-loop that adapts a copy of the weights with a few SGD steps on the support
set (with `create_graph=True` at meta-train time for the second-order update),
and an outer loop that meta-updates the initialization on the query loss. We
compare the meta-learned init against a **random-init baseline** that does the
same inner-loop adaptation — isolating the value of the learned starting point.

## Run
```bash
python hw15_maml.py --n-way 5 --k-shot 1 --meta-iters 300 --eval-episodes 100
```
Omniglot auto-downloads via torchvision; meta-train / meta-test use disjoint
character classes.

## Measured result (CPU, 3 threads, Omniglot, 5-way 1-shot)
| method | meta-test accuracy |
|---|---|
| random-init + same adaptation (baseline) | 0.3972 |
| **MAML** | **0.6512**  (meta-gain **+0.2540**) |

The meta-learned initialization gives a **+25 pp** improvement over adapting from
a random init — MAML has learned a starting point from which a handful of
gradient steps generalizes to unseen character classes.
