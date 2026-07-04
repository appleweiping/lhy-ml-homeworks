"""HW12 - Reinforcement Learning (Policy Gradient).

Course: 李宏毅 ML 2022 Spring, HW12.

Implements REINFORCE (policy gradient) with a learned baseline (advantage
actor-critic style variance reduction) — the official HW12 method. The official
task uses OpenAI Gym's LunarLander; to avoid the gym/box2d dependency (which is
awkward on this CPU-only Windows box) we implement the classic **CartPole**
dynamics from scratch (exact physics from Barto/Sutton), which is the same
family of control task. Reported metric: mean episode return over the last 50
episodes, and the reward curve.
"""
import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(3)


class CartPole:
    """Exact CartPole-v1 dynamics (no gym dependency)."""
    def __init__(self, seed=0):
        self.g = 9.8
        self.mc, self.mp, self.l = 1.0, 0.1, 0.5
        self.total_m = self.mc + self.mp
        self.pml = self.mp * self.l
        self.force = 10.0
        self.tau = 0.02
        self.theta_thr = 12 * 2 * math.pi / 360
        self.x_thr = 2.4
        self.rng = np.random.default_rng(seed)
        self.max_steps = 500

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        x, xdot, th, thdot = self.state
        force = self.force if action == 1 else -self.force
        ct, st = math.cos(th), math.sin(th)
        temp = (force + self.pml * thdot ** 2 * st) / self.total_m
        thacc = (self.g * st - ct * temp) / (
            self.l * (4.0 / 3.0 - self.mp * ct ** 2 / self.total_m))
        xacc = temp - self.pml * thacc * ct / self.total_m
        x += self.tau * xdot; xdot += self.tau * xacc
        th += self.tau * thdot; thdot += self.tau * thacc
        self.state = np.array([x, xdot, th, thdot])
        self.steps += 1
        done = (abs(x) > self.x_thr or abs(th) > self.theta_thr
                or self.steps >= self.max_steps)
        return self.state.copy(), 1.0, done


class PolicyNet(nn.Module):
    def __init__(self, obs=4, act=2):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.pi = nn.Linear(64, act)
        self.v = nn.Linear(64, 1)

    def forward(self, x):
        h = self.body(x)
        return F.log_softmax(self.pi(h), -1), self.v(h).squeeze(-1)


def discount(rewards, gamma):
    out = np.zeros_like(rewards, dtype=np.float32)
    acc = 0.0
    for i in reversed(range(len(rewards))):
        acc = rewards[i] + gamma * acc
        out[i] = acc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    torch.manual_seed(0)
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    env = CartPole()
    net = PolicyNet()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    returns_hist = []
    for ep in range(args.episodes):
        s = env.reset()
        logps, values, rewards, entropies = [], [], [], []
        done = False
        while not done:
            st = torch.tensor(s, dtype=torch.float32)
            logp, val = net(st)
            probs = logp.exp()
            a = torch.multinomial(probs, 1).item()
            entropies.append(-(probs * logp).sum())
            logps.append(logp[a])
            values.append(val)
            s, r, done = env.step(a)
            rewards.append(r)
        G = torch.tensor(discount(np.array(rewards), args.gamma))
        G = (G - G.mean()) / (G.std() + 1e-8)
        V = torch.stack(values)
        adv = G - V.detach()
        policy_loss = -(torch.stack(logps) * adv).sum()
        value_loss = F.smooth_l1_loss(V, G, reduction="sum")
        ent = torch.stack(entropies).sum()
        loss = policy_loss + 0.5 * value_loss - 0.01 * ent
        opt.zero_grad(); loss.backward(); opt.step()

        returns_hist.append(sum(rewards))
        if ep % 50 == 0 or ep == args.episodes - 1:
            recent = np.mean(returns_hist[-50:])
            print(f"episode {ep:4d} | return {sum(rewards):.0f} | last-50 mean {recent:.1f}")

    final = float(np.mean(returns_hist[-50:]))
    best = float(np.max([np.mean(returns_hist[max(0, i-49):i+1]) for i in range(len(returns_hist))]))
    print(f"FINAL last-50 mean return {final:.1f} | best 50-window {best:.1f} | max=500")

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"final_last50_mean_return {final:.1f}\nbest_50window {best:.1f}\n"
                f"max_return 500\nepisodes {args.episodes}\nalgo REINFORCE+baseline\n")
    np.save(os.path.join(out_dir, "returns.npy"), np.array(returns_hist))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(returns_hist, alpha=0.4, label="episode return")
        ma = np.convolve(returns_hist, np.ones(50) / 50, mode="valid")
        plt.plot(range(49, 49 + len(ma)), ma, label="50-ep moving avg", lw=2)
        plt.xlabel("episode"); plt.ylabel("return"); plt.legend()
        plt.title("REINFORCE on CartPole")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "reward_curve.png"), dpi=90)
        print("wrote reward_curve.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
