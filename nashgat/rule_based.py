"""Non-learning reference controllers (FixedTime, MaxPressure) on the same simulator."""
from __future__ import annotations

import numpy as np

from .envs.traffic_env import TrafficMAEnv


def _phase_pressures(env: TrafficMAEnv, a: int) -> np.ndarray:
    sim = env.sim
    q = sim.queue_lengths()
    out_load = np.zeros(sim.n_mov)
    for m, (_, r_out) in enumerate(sim.mov_key):
        out_load[m] = sim.count[r_out] / sim.cap[r_out] * 10.0  # downstream occupancy proxy
    pressures = []
    for p in range(1, env.n_actions + 1):
        movs = sim.agent_phases[a][p]
        pressures.append(sum(q[m] - out_load[m] for m in movs))
    return np.array(pressures)


def run_rule(env: TrafficMAEnv, policy: str, cycle_steps: int = 3, gamma: float = 0.99) -> dict:
    """policy: 'fixed' (each phase for cycle_steps decisions) or 'maxpressure'."""
    env.reset()
    k, rs = 0, []
    while True:
        if policy == "fixed":
            act = np.full(env.n_agents, (k // cycle_steps) % env.n_actions)
        elif policy == "maxpressure":
            act = np.array([int(np.argmax(_phase_pressures(env, a))) for a in range(env.n_agents)])
        else:
            raise ValueError(policy)
        _, r, _, trunc, info = env.step(act)
        rs.append(r)
        k += 1
        if trunc:
            break
    R = np.array(rs)
    out = {kk: v for kk, v in info["metrics"].items() if kk != "time"}
    out["J_sum"] = float((gamma ** np.arange(len(R))[:, None] * R).sum())
    out["avg_queue_per_intersection"] = float(-R.mean() * env.cfg.reward_scale)
    # MaxPressure needs downstream state from every neighbour at every decision
    out["comm_per_agent_step"] = env.edge_index.shape[1] / env.n_agents if policy == "maxpressure" else 0.0
    return out
