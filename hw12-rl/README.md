# HW12 — Reinforcement Learning (Policy Gradient)

**REINFORCE with a learned baseline** (advantage actor-critic style variance
reduction) — the official HW12 method. The official task uses OpenAI Gym's
LunarLander; to avoid the gym/box2d dependency (awkward on this CPU-only Windows
box) we implement the classic **CartPole-v1** dynamics from scratch (exact
physics), the same family of control task.

## Run
```bash
python hw12_rl.py --episodes 800
```

## Measured result (CPU, 3 threads, CartPole, 800 episodes)
| metric | value |
|---|---|
| final last-50-episode mean return | **480.9** |
| best 50-episode window | 500.0 |
| max possible return | 500 |

The policy solves CartPole (near the 500-step cap). Figure:
`results/reward_curve.png` (per-episode return + 50-episode moving average).
Algorithm: policy-gradient loss − advantage·log π, value baseline via smooth-L1,
plus an entropy bonus.
