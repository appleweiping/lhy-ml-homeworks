# HW13 — Network Compression (Knowledge Distillation + Pruning)

Compress a large teacher CNN into a tiny student under a parameter budget. The
official task compresses a food-11 classifier; food-11 is gated, so we run the
**identical pipeline** on real **FashionMNIST**:

1. Train a large **teacher** CNN.
2. Distill it into a small depthwise-separable **student** via **Hinton KD**
   (`α·T²·KL(soft_teacher‖soft_student) + (1−α)·CE`), vs a student trained from
   scratch on hard labels only.
3. Apply **global L1 unstructured pruning** to the KD student and re-measure
   accuracy vs the real (nonzero) parameter density.

## Run
```bash
python hw13_compression.py --teacher-epochs 12 --epochs 4 --subset 15000
```

## Measured result (CPU, 3 threads, FashionMNIST)
| model | params | test acc |
|---|---|---|
| teacher | 390,858 | 0.8966 |
| student (from scratch) | 7,172 | 0.8262 |
| **student (KD)** | 7,172 | **0.8439**  (KD gain **+0.0177**) |

Compression ratio **54.5×**. Distillation from the stronger teacher lifts the
7k-param student above training it from scratch. Global L1 pruning of the KD
student degrades gracefully: ~0.83 acc at 30 % pruned (density 0.71), ~0.78 at
50 %, ~0.56 at 70 %, chance at 90 % — see `results/metrics.txt`.
