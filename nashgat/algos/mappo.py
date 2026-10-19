"""Unified PPO / MAPPO trainer covering all compared methods.

A method is a point in the design space
    actor encoder  x  critic  x  reward mode  x  graph dynamics  x  (lambda, mu):

  IPPO          mlp actor, local critic,   individual rewards
  MAPPO         mlp actor, central critic, welfare (mean) reward
  GNN-MAPPO     GAT actor (static graph, no edge features), graph critic, welfare
  Nash-MAPPO    MAPPO + exploitability-gap regulariser (lambda > 0)
  NashGAT-MARL  GAT actor with traffic-coupling edge features and learned
                communication gates (dynamic E_t), graph critic, welfare reward,
                gap regulariser (lambda > 0) and communication penalty (mu > 0)

The gap regulariser for agent i at state s is
    gap_i(s) = max_a Q_i(s, a) - sum_a pi_i(a | h_i) Q_i(s, a)  >= 0,
where Q_i is a centralised critic of agent i's *individual* reward under the current
joint policy. Its expectation under d^pi upper-bounds agent i's exploitability up to a
distribution-mismatch factor (MATHEMATICAL_MODEL.md, Prop. 1).
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical

from ..envs.traffic_env import TrafficMAEnv
from ..graph import DynamicGraphBuilder, batch_graphs
from ..models import CRITICS, Actor


@dataclass
class AlgoConfig:
    name: str = "nashgat"
    actor_encoder: str = "gat"  # mlp | gat | mean
    edge_features: bool = True  # traffic-coupling edge features in actor attention
    comm_gate: bool = True  # learned dynamic communication graph
    critic: str = "graph"  # local | central | graph
    reward_mode: str = "welfare"  # individual | welfare
    nash_lambda: float = 0.5
    comm_mu: float = 0.05
    hidden: int = 64
    heads: int = 4
    lr_actor: float = 5e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    episodes: int = 150
    comm_fail_prob: float = 0.0


PRESETS: dict[str, dict] = {
    "ippo": dict(actor_encoder="mlp", edge_features=False, comm_gate=False, critic="local",
                 reward_mode="individual", nash_lambda=0.0, comm_mu=0.0),
    "mappo": dict(actor_encoder="mlp", edge_features=False, comm_gate=False, critic="central",
                  reward_mode="welfare", nash_lambda=0.0, comm_mu=0.0),
    "gnn_mappo": dict(actor_encoder="gat", edge_features=False, comm_gate=False, critic="graph",
                      reward_mode="welfare", nash_lambda=0.0, comm_mu=0.0),
    "nash_mappo": dict(actor_encoder="mlp", edge_features=False, comm_gate=False, critic="central",
                       reward_mode="welfare", nash_lambda=0.5, comm_mu=0.0),
    "nashgat": dict(actor_encoder="gat", edge_features=True, comm_gate=True, critic="graph",
                    reward_mode="welfare", nash_lambda=0.5, comm_mu=0.05),
    # ablations of NashGAT-MARL
    "nashgat_no_gnn": dict(actor_encoder="mlp", edge_features=False, comm_gate=False, critic="graph",
                           reward_mode="welfare", nash_lambda=0.5, comm_mu=0.0),
    "nashgat_no_attention": dict(actor_encoder="mean", edge_features=False, comm_gate=True, critic="graph",
                                 reward_mode="welfare", nash_lambda=0.5, comm_mu=0.05),
    "nashgat_no_nash": dict(actor_encoder="gat", edge_features=True, comm_gate=True, critic="graph",
                            reward_mode="welfare", nash_lambda=0.0, comm_mu=0.05),
    "nashgat_no_comm_penalty": dict(actor_encoder="gat", edge_features=True, comm_gate=True, critic="graph",
                                    reward_mode="welfare", nash_lambda=0.5, comm_mu=0.0),
    "nashgat_static_graph": dict(actor_encoder="gat", edge_features=True, comm_gate=False, critic="graph",
                                 reward_mode="welfare", nash_lambda=0.5, comm_mu=0.0),
}


def make_config(preset: str, **overrides) -> AlgoConfig:
    base = AlgoConfig(name=preset)
    return replace(base, **{**PRESETS[preset], **overrides})


class MAPPOTrainer:
    def __init__(self, env: TrafficMAEnv, cfg: AlgoConfig, seed: int = 0, device: str | torch.device = "cpu"):
        self.env, self.cfg, self.seed = env, cfg, seed
        self.device = torch.device(device)
        self.N, self.A, self.D = env.n_agents, env.n_actions, env.obs_dim
        self.edge_index_np = env.edge_index
        self.edge_index = torch.as_tensor(env.edge_index, device=self.device)
        self.rng = np.random.default_rng(seed)
        self.uses_messages = cfg.actor_encoder != "mlp"
        self.graph = DynamicGraphBuilder(env.edge_index, self.N, gated=cfg.comm_gate,
                                         fail_prob=cfg.comm_fail_prob, rng=self.rng)
        edim = env.edge_dim if cfg.edge_features else None
        self.actor = Actor(self.D, self.A, cfg.hidden, cfg.actor_encoder, edim, cfg.heads, cfg.comm_gate).to(self.device)
        self.critic = CRITICS[cfg.critic](obs_dim=self.D, n_agents=self.N, n_actions=self.A, hidden=cfg.hidden,
                                          edge_dim=env.edge_dim, heads=cfg.heads).to(self.device)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.max_out = max(1, int(self.graph.out_degree.max()))
        self.history: list[dict] = []
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    # ------------------------------------------------------------ helpers
    def _actor_forward(self, actor: Actor, obs: torch.Tensor, masks: torch.Tensor, eattr: torch.Tensor,
                       return_attention: bool = False):
        """obs (B,N,D), masks (B,E), eattr (B,E,F) -> logits (B,N,A), gate logits (B,N) | None."""
        B = obs.shape[0]
        ei, ea = batch_graphs(self.edge_index, self.N, masks, eattr if self.cfg.edge_features else None)
        out = actor(obs.reshape(B * self.N, self.D), ei, ea, return_attention=return_attention)
        logits = out[0].view(B, self.N, self.A)
        gl = out[1].view(B, self.N) if out[1] is not None else None
        if return_attention:
            return logits, gl, out[2]
        return logits, gl

    def _critic_forward(self, obs: torch.Tensor, tfeat: torch.Tensor, eattr: torch.Tensor):
        B = obs.shape[0]
        if self.cfg.critic == "graph":
            full = torch.ones(B, self.edge_index.shape[1], dtype=torch.bool, device=self.device)
            ei, ea = batch_graphs(self.edge_index, self.N, full, eattr)
            return self.critic(obs, tfeat, ei, ea)
        return self.critic(obs, tfeat)

    # ------------------------------------------------------------ rollout
    @torch.no_grad()
    def rollout(self, greedy: bool = False, dev_actor: Actor | None = None, dev_agent: int | None = None,
                record_attention_steps: tuple[int, ...] = (), env: TrafficMAEnv | None = None) -> dict:
        env = env or self.env
        obs, _ = env.reset()
        self.graph.reset()
        T = int(np.ceil(env.cfg.horizon / env.cfg.decision_interval))
        E = self.edge_index.shape[1]
        buf = {k: [] for k in ("obs", "eattr", "mask", "act", "gate", "logp", "logits", "r", "comm", "tfeat")}
        attention = {}
        t = 0
        while True:
            mask = self.graph.active_mask() if self.uses_messages else np.zeros(E, bool)
            eattr = env.edge_features()
            o = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            m = torch.as_tensor(mask, device=self.device).unsqueeze(0)
            ea = torch.as_tensor(eattr, device=self.device).unsqueeze(0)
            if t in record_attention_steps and self.cfg.actor_encoder == "gat":
                logits, gl, att = self._actor_forward(self.actor, o, m, ea, return_attention=True)
                attention[t] = (att[0].cpu().numpy(), att[1].mean(-1).cpu().numpy())
            else:
                logits, gl = self._actor_forward(self.actor, o, m, ea)
            if dev_actor is not None:
                dl, dg = self._actor_forward(dev_actor, o, m, ea)
                logits = logits.clone()
                logits[0, dev_agent] = dl[0, dev_agent]
                if gl is not None:
                    gl = gl.clone()
                    gl[0, dev_agent] = dg[0, dev_agent]
            dist = Categorical(logits=logits[0])
            a = logits[0].argmax(-1) if greedy else dist.sample()
            logp = dist.log_prob(a)
            if gl is not None:
                gd = Bernoulli(logits=gl[0])
                g = (gl[0] > 0).float() if greedy else gd.sample()
                logp = logp + gd.log_prob(g)
            else:
                g = torch.ones(self.N, device=self.device)
            comm = self.graph.comm_cost(mask) if self.uses_messages else np.zeros(self.N)
            buf["obs"].append(obs)
            buf["eattr"].append(eattr)
            buf["mask"].append(mask)
            buf["act"].append(a.cpu().numpy())
            buf["gate"].append(g.cpu().numpy())
            buf["logp"].append(logp.cpu().numpy())
            buf["logits"].append(logits[0].cpu().numpy())
            buf["comm"].append(comm)
            buf["tfeat"].append(t / T)
            self.graph.update(g.cpu().numpy().astype(np.int64))
            obs, r, _, trunc, info = env.step(a.cpu().numpy())
            buf["r"].append(r)
            t += 1
            if trunc:
                break
        out = {k: np.asarray(v) for k, v in buf.items()}
        out["last_obs"] = obs
        out["last_eattr"] = env.edge_features()
        out["last_mask"] = self.graph.active_mask() if self.uses_messages else np.zeros(E, bool)
        out["last_tfeat"] = t / T
        out["metrics"] = info["metrics"]
        out["attention"] = attention
        return out

    # --------------------------------------------------------------- update
    def _returns(self, rewards: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """GAE(lambda) with time-limit bootstrapping. rewards (T,N), values (T+1,N)."""
        cfg = self.cfg
        T = rewards.shape[0]
        adv = np.zeros_like(rewards)
        last = np.zeros(rewards.shape[1])
        for t in reversed(range(T)):
            delta = rewards[t] + cfg.gamma * values[t + 1] - values[t]
            last = delta + cfg.gamma * cfg.gae_lambda * last
            adv[t] = last
        return adv, adv + values[:-1]

    def shaped_rewards(self, buf: dict) -> np.ndarray:
        cfg = self.cfg
        r = buf["r"]
        c = buf["gate"] * self.graph.out_degree[None, :] / self.max_out if cfg.comm_gate else np.zeros_like(r)
        if cfg.reward_mode == "individual":
            return r - cfg.comm_mu * c
        team = r.mean(1, keepdims=True) - cfg.comm_mu * c.mean(1, keepdims=True)
        return np.repeat(team, self.N, axis=1)

    def update(self, buf: dict) -> dict:
        cfg, dev = self.cfg, self.device
        T = buf["r"].shape[0]
        obs_all = torch.as_tensor(np.concatenate([buf["obs"], buf["last_obs"][None]]), device=dev)
        ea_all = torch.as_tensor(np.concatenate([buf["eattr"], buf["last_eattr"][None]]), device=dev)
        tf_all = torch.as_tensor(np.append(buf["tfeat"], buf["last_tfeat"]), dtype=torch.float32, device=dev)
        mask_all = torch.as_tensor(np.concatenate([buf["mask"], buf["last_mask"][None]]), device=dev)
        with torch.no_grad():
            v_all, q_all = self._critic_forward(obs_all, tf_all, ea_all)
            last_logits, _ = self._actor_forward(self.actor, obs_all[-1:], mask_all[-1:], ea_all[-1:])
            pi_all = torch.softmax(torch.cat([torch.as_tensor(buf["logits"], device=dev), last_logits]), -1)
            v_ind = (pi_all * q_all).sum(-1)
        R = self.shaped_rewards(buf)
        adv, ret = self._returns(R, v_all.cpu().numpy())
        _, q_ret = self._returns(buf["r"], v_ind.cpu().numpy())
        adv_std = float(adv.std()) + 1e-8
        adv_n = (adv - adv.mean()) / adv_std

        obs = obs_all[:-1]
        ea = ea_all[:-1]
        tf = tf_all[:-1]
        masks = mask_all[:-1]
        act = torch.as_tensor(buf["act"], device=dev)
        gate = torch.as_tensor(buf["gate"], dtype=torch.float32, device=dev)
        old_logp = torch.as_tensor(buf["logp"], device=dev)
        adv_t = torch.as_tensor(adv_n, dtype=torch.float32, device=dev)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=dev)
        qret_t = torch.as_tensor(q_ret, dtype=torch.float32, device=dev)
        stats = {"pg": 0.0, "v": 0.0, "q": 0.0, "gap": 0.0, "ent": 0.0, "kl": 0.0}
        n_mb = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(T, generator=self.gen)
            for idx in perm.chunk(cfg.minibatches):
                idx = idx.to(dev)
                logits, gl = self._actor_forward(self.actor, obs[idx], masks[idx], ea[idx])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act[idx])
                ent = dist.entropy()
                if gl is not None:
                    gd = Bernoulli(logits=gl)
                    logp = logp + gd.log_prob(gate[idx])
                    ent = ent + gd.entropy()
                ratio = torch.exp(logp - old_logp[idx])
                a_mb = adv_t[idx]
                pg = -torch.min(ratio * a_mb, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a_mb).mean()
                v, q = self._critic_forward(obs[idx], tf[idx], ea[idx])
                qd = q.detach()
                pi = torch.softmax(logits, -1)
                gap = qd.max(-1).values - (pi * qd).sum(-1)  # (B,N), >= 0
                loss_a = pg - cfg.ent_coef * ent.mean()
                if cfg.nash_lambda > 0:
                    loss_a = loss_a + cfg.nash_lambda * gap.mean() / adv_std
                self.opt_a.zero_grad()
                loss_a.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.opt_a.step()
                q_taken = q.gather(-1, act[idx].unsqueeze(-1)).squeeze(-1)
                loss_v = ((v - ret_t[idx]) ** 2).mean()
                loss_q = ((q_taken - qret_t[idx]) ** 2).mean()
                self.opt_c.zero_grad()
                (loss_v + loss_q).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.opt_c.step()
                with torch.no_grad():
                    stats["kl"] += float((old_logp[idx] - logp).mean())
                stats["pg"] += float(pg.detach())
                stats["v"] += float(loss_v.detach())
                stats["q"] += float(loss_q.detach())
                stats["gap"] += float(gap.mean().detach())
                stats["ent"] += float(ent.mean().detach())
                n_mb += 1
        return {k: v / n_mb for k, v in stats.items()}

    # ---------------------------------------------------------------- loops
    def train(self, episodes: int | None = None, log_every: int = 10, callback=None) -> list[dict]:
        episodes = episodes or self.cfg.episodes
        for ep in range(episodes):
            t0 = time.time()
            buf = self.rollout()
            st = self.update(buf)
            rec = {"episode": ep, "seed": self.seed, "method": self.cfg.name,
                   "ep_return_ind_mean": float(buf["r"].sum(0).mean()),
                   "ep_return_team": float(buf["r"].mean(1).sum()),
                   "comm_per_agent_step": float(buf["comm"].mean()),
                   "gate_rate": float(buf["gate"].mean()),
                   **{f"loss_{k}": v for k, v in st.items()},
                   **{k: v for k, v in buf["metrics"].items() if k != "time"},
                   "wall_s": time.time() - t0}
            self.history.append(rec)
            if callback:
                callback(rec)
        return self.history

    def evaluate(self, n_episodes: int = 3, greedy: bool = False, env: TrafficMAEnv | None = None) -> dict:
        """Average traffic metrics, discounted individual returns J_i and comm cost."""
        mets, J, comm, queue = [], [], [], []
        env = env or self.env
        for _ in range(n_episodes):
            b = self.rollout(greedy=greedy, env=env)
            disc = self.cfg.gamma ** np.arange(b["r"].shape[0])
            J.append((disc[:, None] * b["r"]).sum(0))
            mets.append(b["metrics"])
            comm.append(b["comm"].mean())
            queue.append(-b["r"].mean() * env.cfg.reward_scale)
        keys = [k for k in mets[0] if k != "time"]
        out = {k: float(np.mean([m[k] for m in mets])) for k in keys}
        out["J_per_agent"] = np.mean(J, 0).tolist()
        out["J_sum"] = float(np.mean(J, 0).sum())
        out["comm_per_agent_step"] = float(np.mean(comm))
        out["avg_queue_per_intersection"] = float(np.mean(queue))
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": asdict(self.cfg), "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "seed": self.seed}, path)

    def load(self, path: str | Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
