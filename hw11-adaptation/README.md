# HW11 — Domain Adaptation (DaNN)

**Domain-Adversarial Neural Network** (Ganin et al.) — a shared feature extractor
feeding a label predictor and a domain classifier, connected through a **Gradient
Reversal Layer** so the features become domain-invariant. This is exactly the
official HW11 method.

The official task adapts real photos → hand-drawn sketches. We reproduce the setup
on the real **CIFAR-10** dataset: the *source* domain is the normal RGB images
(labelled); the *target* domain is a shifted version (grayscale + edge-emphasis,
labels hidden during training). We report target-domain accuracy with vs without
adaptation.

## Run
```bash
python hw11_dann.py --epochs 10
```

## Measured result (CPU, 3 threads, CIFAR-10 RGB → edge domain)
| model | target-domain accuracy |
|---|---|
| source-only (no adaptation) | 0.5100 |
| **DaNN** | **0.5445**  (adaptation gain **+0.0345**) |

The gradient-reversal adversarial term (λ ramped 0 → 1 over training) makes the
features domain-invariant, improving accuracy on the unlabelled target domain.
