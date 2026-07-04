# HW9 — Explainable AI

The official explainability methods applied to an image classifier, all
implemented from scratch:

1. **Saliency map** — |∂loss/∂input|
2. **Smooth-grad** — saliency averaged over noisy inputs
3. **Integrated gradients** — path integral of gradients from a zero baseline
4. **Occlusion sensitivity** — confidence drop when patches are masked

The official task explains a food-11 CNN; we train a small CNN on real
**CIFAR-10** and produce all four attribution maps.

## Run
```bash
python hw9_explainable.py --epochs 8
```

## Measured result (CPU, 3 threads, CIFAR-10)
Quantitative faithfulness check — the **deletion metric**: masking the top-k most
salient pixels should drop the predicted-class probability more than masking
random pixels.

| masked pixels | predicted-prob drop |
|---|---|
| top-k salient (smooth-grad) | **0.3065** |
| random | 0.2486 |

Top-k > random → `attribution_valid = True`: the attributions are meaningful.
Figure: `results/attributions.png` (6 images × 4 methods).
