"""Approximate best-response learning and empirical exploitability for the traffic game.

For a trained joint policy pi and an agent k, a deviation policy pi_k' (same
architecture and information as pi_k, initialised from pi_k) is trained with PPO
on agent k's individual reward while every other agent keeps pi_-k frozen. Then

    delta_k = J_k(pi_k', pi_-k) - J_k(pi)

is estimated from independent evaluation episodes. Because pi_k' is only an
approximate best response, delta_k estimates a LOWER bound on agent k's true
exploitability within the policy class; sum_k delta_k over the sampled agents is
reported as "approximate NashConv (sampled agents)".
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical

from .mappo import MAPPOTrainer


class _ValueNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _discounted(r: np.ndarray, gamma: float) -> float:
    return float((gamma ** np.arange(len(r)) * r).sum())


def estimate_J(tr: MAPPOTrainer, agent: int, n_episodes: int, dev_actor=None) -> list[float]:
    out = []
    for _ in range(n_episodes):
        b = tr.rollout(dev_actor=dev_actor, dev_agent=agent if dev_actor is not None else None)
        out.append(_discounted(b["r"][:, agent], tr.cfg.gamma))
    return out


def approximate_best_response(tr: MAPPOTrainer, agent: int, episodes: int = 20, eval_episodes: int = 5,
                              lr: float = 5e-4, seed: int = 0) -> dict:
    cfg, dev = tr.cfg, tr.device
    torch.manual_seed(seed)
    base_J = estimate_J(tr, agent, eval_episodes)
    dev_actor = copy.deepcopy(tr.actor)
    vnet = _ValueNet(tr.N * tr.D + 1, cfg.hidden).to(dev)
    opt_a = torch.optim.Adam(dev_actor.parameters(), lr=lr)
    opt_v = torch.optim.Adam(vnet.parameters(), lr=1e-3)
    curve = []
    for ep in range(episodes):
        b = tr.rollout(dev_actor=dev_actor, dev_agent=agent)
        T = b["r"].shape[0]
        obs_all = torch.as_tensor(np.concatenate([b["obs"], b["last_obs"][None]]), device=dev)
        tf_all = torch.as_tensor(np.append(b["tfeat"], b["last_tfeat"]), dtype=torch.float32, device=dev)
        g_in = torch.cat([obs_all.reshape(T + 1, -1), tf_all.unsqueeze(-1)], -1)
        with torch.no_grad():
            v = vnet(g_in).cpu().numpy()
        r = b["r"][:, agent]
        adv = np.zeros(T)
        last = 0.0
        for t in reversed(range(T)):
            delta = r[t] + cfg.gamma * v[t + 1] - v[t]
            last = delta + cfg.gamma * cfg.gae_lambda * last
            adv[t] = last
        ret = torch.as_tensor(adv + v[:-1], dtype=torch.float32, device=dev)
        adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), dtype=torch.float32, device=dev)
        obs = obs_all[:-1]
        masks = torch.as_tensor(b["mask"], device=dev)
        ea = torch.as_tensor(b["eattr"], device=dev)
        act = torch.as_tensor(b["act"][:, agent], device=dev)
        gate = torch.as_tensor(b["gate"][:, agent], dtype=torch.float32, device=dev)
        old_logp = torch.as_tensor(b["logp"][:, agent], device=dev)
        for _ in range(cfg.epochs):
            for idx in torch.randperm(T).chunk(cfg.minibatches):
                logits, gl = tr._actor_forward(dev_actor, obs[idx], masks[idx], ea[idx])
                dist = Categorical(logits=logits[:, agent])
                logp = dist.log_prob(act[idx])
                ent = dist.entropy()
                if gl is not None:
                    gd = Bernoulli(logits=gl[:, agent])
                    logp = logp + gd.log_prob(gate[idx])
                ratio = torch.exp(logp - old_logp[idx])
                pg = -torch.min(ratio * adv_t[idx], torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_t[idx]).mean()
                loss = pg - cfg.ent_coef * ent.mean()
                opt_a.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(dev_actor.parameters(), cfg.max_grad_norm)
                opt_a.step()
                lv = ((vnet(g_in[:-1][idx]) - ret[idx]) ** 2).mean()
                opt_v.zero_grad()
                lv.backward()
                opt_v.step()
        curve.append(_discounted(r, cfg.gamma))
    br_J = estimate_J(tr, agent, eval_episodes, dev_actor=dev_actor)
    gain = np.array(br_J).mean() - np.array(base_J).mean()
    se = np.sqrt(np.var(br_J, ddof=1) / len(br_J) + np.var(base_J, ddof=1) / len(base_J)) if eval_episodes > 1 else np.nan
    return {"agent": agent, "J_base": float(np.mean(base_J)), "J_br": float(np.mean(br_J)), "gain": float(gain),
            "gain_se": float(se), "train_curve": curve}


def sampled_nash_conv(tr: MAPPOTrainer, agents: list[int], **kw) -> dict:
    res = [approximate_best_response(tr, k, seed=tr.seed * 100 + k, **kw) for k in agents]
    gains = np.array([r["gain"] for r in res])
    return {"agents": agents, "per_agent": res, "nash_conv_sampled": float(np.maximum(gains, 0).sum()),
            "nash_conv_sampled_raw": float(gains.sum())}
