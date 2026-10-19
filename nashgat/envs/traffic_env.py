"""Multi-agent traffic-signal environment (one agent per signalised intersection).

The environment is a partially observable general-sum Markov game:
  * agent i observes the queues/approaching counts of its own 12 movements and its
    current phase,
  * action a_i selects one of the controllable light phases for the next
    `decision_interval` seconds (a yellow interval is inserted on change),
  * reward r_i = -(sum of queued vehicles on i's movements) / reward_scale,
    measured at the end of the decision interval.

Rewards are individual: an intersection that flushes its own queues into a
congested neighbour is rewarded even though the neighbour pays the cost, which
is what makes the game general-sum rather than fully cooperative.

The same API is served by two backends: QueueSim (always available) and CityFlow
(when the `cityflow` package can be imported). Results from the two must never be
pooled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .queue_sim import QueueSim, SimConfig
from .roadnet import RoadNet, expand_departures, load_flows, load_roadnet


@dataclass
class EnvConfig:
    roadnet: str
    flow: str
    backend: str = "queuesim"  # 'queuesim' | 'cityflow'
    horizon: int = 3600
    decision_interval: int = 10
    yellow_time: int = 3
    reward_scale: float = 20.0
    obs_scale: float = 20.0
    obs_noise_std: float = 0.0
    demand_scale: float = 1.0  # >1 duplicates, <1 thins vehicles (transformed demand)
    departure_jitter: int = 0  # +- seconds of uniform jitter applied to departures
    failed_agents: list[int] = field(default_factory=list)
    seed: int = 0


def transform_departures(deps: list[tuple[int, list[str]]], scale: float, jitter: int,
                         rng: np.random.Generator, horizon: int) -> list[tuple[int, list[str]]]:
    """Seeded demand perturbation used only for robustness experiments."""
    out = []
    for t, rt in deps:
        k = int(scale) + (1 if rng.random() < scale - int(scale) else 0)
        for _ in range(k):
            tt = t + (int(rng.integers(-jitter, jitter + 1)) if jitter > 0 else 0)
            out.append((int(np.clip(tt, 0, horizon - 1)), rt))
    out.sort(key=lambda z: z[0])
    return out


class TrafficMAEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.net: RoadNet = load_roadnet(cfg.roadnet)
        self.raw_flows = load_flows(cfg.flow)
        self.agents = self.net.agents
        self.n_agents = len(self.agents)
        self.rng = np.random.default_rng(cfg.seed)
        self._build_backend()
        nm = {len(self.net.intersections[a].movements) for a in self.agents}
        npz = {len(self.net.intersections[a].phases) - 1 for a in self.agents}
        assert len(nm) == 1 and len(npz) == 1, "heterogeneous intersections are not supported"
        self.n_movements = nm.pop()
        self.n_actions = npz.pop()
        self.obs_dim = 2 * self.n_movements + self.n_actions
        # directed agent graph (j -> i) from connecting roads
        e = self.net.neighbour_edges()
        self.edge_index = np.array([[s for s, _, _ in e], [d for _, d, _ in e]], dtype=np.int64).reshape(2, -1)
        self.edge_roads = [r for _, _, r in e]
        self.edge_dim = 2
        self._edge_road_idx = [self.sim.ridx[r] for r in self.edge_roads]
        self._edge_movs = [[m for m, (a, _) in enumerate(self.sim.mov_key) if a == ri] for ri in self._edge_road_idx]

    def _departures(self) -> list[tuple[int, list[str]]]:
        deps = expand_departures(self.raw_flows)
        if self.cfg.demand_scale != 1.0 or self.cfg.departure_jitter > 0:
            deps = transform_departures(deps, self.cfg.demand_scale, self.cfg.departure_jitter, self.rng,
                                        self.cfg.horizon)
        return [d for d in deps if d[0] < self.cfg.horizon]

    def _build_backend(self) -> None:
        if self.cfg.backend == "queuesim":
            self.sim = QueueSim(self.net, self._departures(),
                                SimConfig(yellow_time=self.cfg.yellow_time, horizon=self.cfg.horizon))
        elif self.cfg.backend == "cityflow":
            from .cityflow_backend import CityFlowBackend

            self.sim = CityFlowBackend(self.net, self.cfg)
        else:
            raise ValueError(self.cfg.backend)
        self.sim.failed_agents = set(self.cfg.failed_agents)

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            if self.cfg.demand_scale != 1.0 or self.cfg.departure_jitter > 0:
                self._build_backend()
        self.sim.reset()
        self.steps = 0
        return self._obs(), {}

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (self.n_agents,)
        assert actions.min() >= 0 and actions.max() < self.n_actions
        self.sim.set_phases(actions)
        for _ in range(self.cfg.decision_interval):
            self.sim.step_second()
        self.steps += 1
        q = self.sim.queue_lengths()
        rewards = -np.array([q[m].sum() for m in self.sim.agent_movs], dtype=np.float64) / self.cfg.reward_scale
        truncated = self.sim.t >= self.cfg.horizon
        info = {"queue_per_agent": -rewards * self.cfg.reward_scale}
        if truncated:
            info["metrics"] = self.sim.metrics()
        return self._obs(), rewards, False, truncated, info

    def _obs(self) -> np.ndarray:
        q = self.sim.queue_lengths()
        app = self.sim.approach
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        for a, movs in enumerate(self.sim.agent_movs):
            obs[a, : self.n_movements] = q[movs] / self.cfg.obs_scale
            obs[a, self.n_movements: 2 * self.n_movements] = app[movs] / self.cfg.obs_scale
            obs[a, 2 * self.n_movements + int(self.sim.phase[a]) - 1] = 1.0
        if self.cfg.obs_noise_std > 0:
            obs[:, : 2 * self.n_movements] += self.rng.normal(0, self.cfg.obs_noise_std,
                                                              (self.n_agents, 2 * self.n_movements)).astype(np.float32)
        return obs

    def edge_features(self) -> np.ndarray:
        """Traffic coupling on each directed agent edge j->i: occupancy of the
        connecting road and the share of it already queued at i."""
        feats = np.zeros((len(self.edge_roads), self.edge_dim), dtype=np.float32)
        q = self.sim.queue_lengths()
        for k, ri in enumerate(self._edge_road_idx):
            feats[k, 0] = self.sim.count[ri] / self.sim.cap[ri]
            feats[k, 1] = q[self._edge_movs[k]].sum() / self.cfg.obs_scale
        return feats

    def metrics(self) -> dict:
        return self.sim.metrics()


def default_env_config(root: str | Path, city: str = "jinan", flow_variant: str = "real", **kw) -> EnvConfig:
    root = Path(root)
    if city == "jinan":
        d = root / "data/raw/colight/data/Jinan/3_4"
        rn = d / "roadnet_3_4.json"
        fl = d / {"real": "anon_3_4_jinan_real.json", "2000": "anon_3_4_jinan_real_2000.json",
                  "2500": "anon_3_4_jinan_real_2500.json"}[flow_variant]
    elif city == "hangzhou":
        d = root / "data/raw/colight/data/Hangzhou/4_4"
        rn = d / "roadnet_4_4.json"
        fl = d / {"real": "anon_4_4_hangzhou_real.json", "5734": "anon_4_4_hangzhou_real_5734.json",
                  "5816": "anon_4_4_hangzhou_real_5816.json"}[flow_variant]
    elif city == "newyork":
        d = root / "data/raw/colight/data/NewYork/28_7"
        rn = d / "roadnet_28_7.json"
        fl = d / {"double": "anon_28_7_newyork_real_double.json", "triple": "anon_28_7_newyork_real_triple.json",
                  "real": "anon_28_7_newyork_real_double.json"}[flow_variant]
    else:
        raise ValueError(city)
    return EnvConfig(roadnet=str(rn), flow=str(fl), **kw)
