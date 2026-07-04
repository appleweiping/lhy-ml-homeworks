# HW14 — Life-long Learning (Elastic Weight Consolidation)

Regularisation-based continual learning: train sequentially on a stream of tasks
without catastrophically forgetting earlier ones. We implement **EWC**
(Kirkpatrick et al. 2017) — a diagonal-Fisher-weighted L2 penalty anchoring each
parameter to its value at the end of previous tasks — and evaluate on the
standard **Permuted-MNIST** benchmark built from the real MNIST dataset.

## Run
```bash
python hw14_ewc.py --tasks 4 --epochs 3 --lam 2000
```

## Measured result (CPU, 3 threads, 4 permuted-MNIST tasks)
| method | final avg acc (all tasks) | forgetting ↓ |
|---|---|---|
| naive SGD | 0.7477 | 0.2235 |
| **EWC**   | **0.8914** | **0.0270** |

EWC cuts catastrophic forgetting ~8× (0.224 → 0.027) and lifts final average
accuracy across all four tasks from 0.75 to 0.89. `forgetting` = mean over old
tasks of (best-past-acc − final-acc); lower is better.
