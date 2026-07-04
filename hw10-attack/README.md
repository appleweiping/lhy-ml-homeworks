# HW10 — Adversarial Attack

White-box L-∞ adversarial attacks against an image classifier — the official
HW10 task, done on the real **CIFAR-10** dataset (torchvision auto-download):

- **FGSM** — single-step sign-of-gradient
- **I-FGSM / PGD** — iterative, projected onto the ε-ball
- **MI-FGSM** — momentum iterative

Attacks operate in pixel space with proper de-normalise / re-normalise so the
ε budget (8/255) is enforced on real pixels.

## Run
```bash
python hw10_attack.py --epochs 6 --eps 0.03137 --n-test 1000
```

## Measured result (CPU, 3 threads, CIFAR-10, ε = 8/255)
| setting | accuracy |
|---|---|
| clean | 0.5390 |
| FGSM | 0.0200 |
| **PGD-10** | **0.0000** |
| MI-FGSM-10 | 0.0000 |

Max L-∞ perturbation = 0.03137 = exactly the 8/255 budget. PGD completely breaks
the classifier. Figure: `results/adversarial_examples.png` (clean / adversarial /
perturbation rows).
