"""Single-run experiment driver shared by scripts and the Colab notebook."""
from __future__ import annotations

import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .algos.best_response import sampled_nash_conv
from .algos.mappo import MAPPOTrainer, make_config
from .envs.traffic_env import TrafficMAEnv, default_env_config
from .utils import save_json, set_seed


def evaluate_suite(tr: MAPPOTrainer, data_root: str | Path, eval_episodes: int = 3, robustness: bool = True) -> dict:
    res = {}
    base = tr.env.cfg
    res["nominal_stochastic"] = tr.evaluate(eval_episodes)
    res["nominal_greedy"] = tr.evaluate(1, greedy=True)
    if not robustness:
        return res
    res["obs_noise_0.1"] = tr.evaluate(eval_episodes, env=TrafficMAEnv(replace(base, obs_noise_std=0.1, seed=7)))
    old = tr.graph.fail_prob
    tr.graph.fail_prob = 0.3
    res["comm_fail_0.3"] = tr.evaluate(eval_episodes)
    tr.graph.fail_prob = old
    for fv in ("2000", "2500"):
        cfg = replace(default_env_config(data_root, "jinan", fv, horizon=base.horizon), backend=base.backend)
        res[f"flow_{fv}"] = tr.evaluate(eval_episodes, env=TrafficMAEnv(cfg))
    res["demand_x1.3"] = tr.evaluate(eval_episodes,
                                     env=TrafficMAEnv(replace(base, demand_scale=1.3, departure_jitter=30, seed=11)))
    res["agent_fail_2"] = tr.evaluate(eval_episodes, env=TrafficMAEnv(replace(base, failed_agents=[4, 7])))
    hz = TrafficMAEnv(replace(default_env_config(data_root, "hangzhou", "real", horizon=base.horizon), backend=base.backend))
    tr_hz = MAPPOTrainer(hz, replace(tr.cfg, critic="local"), seed=tr.seed)
    tr_hz.actor.load_state_dict(tr.actor.state_dict())
    res["transfer_hangzhou"] = tr_hz.evaluate(eval_episodes)
    return res


def run_single(method: str, seed: int, data_root: str | Path, out_root: str | Path, episodes: int,
               horizon: int = 3600, backend: str = "queuesim", eval_episodes: int = 3, robustness: bool = True,
               br_agents: tuple[int, ...] = (0, 1, 5), br_episodes: int = 20, br_eval_episodes: int = 5,
               device: str = "cpu", tag: str = "", log_every: int = 0, **overrides) -> dict:
    set_seed(seed)
    cfg = make_config(method, episodes=episodes, **overrides)
    label = method + (f"_{tag}" if tag else "")
    cfg = replace(cfg, name=label)
    out = Path(out_root) / label / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    env = TrafficMAEnv(replace(default_env_config(data_root, "jinan", "real", horizon=horizon, seed=seed), backend=backend))
    tr = MAPPOTrainer(env, cfg, seed=seed, device=device)
    save_json({"algo": asdict(cfg), "env": asdict(env.cfg)}, out / "config.json")
    t0 = time.time()

    def cb(rec):
        if log_every and rec["episode"] % log_every == 0:
            print(f"  [{label} s{seed}] ep {rec['episode']:4d} team_return {rec['ep_return_team']:8.1f} "
                  f"ATT {rec['avg_travel_time']:6.1f} gap {rec['loss_gap']:.3f} msgs {rec['comm_per_agent_step']:.2f}")

    tr.train(episodes, callback=cb)
    train_time = time.time() - t0
    pd.DataFrame(tr.history).to_csv(out / "history.csv", index=False)
    tr.save(out / "model.pt")
    ev = evaluate_suite(tr, data_root, eval_episodes, robustness)
    ev["train_time_s"] = train_time
    ev["n_actor_params"] = sum(p.numel() for p in tr.actor.parameters())
    save_json(ev, out / "eval.json")
    if cfg.actor_encoder == "gat":
        T = int(np.ceil(horizon / env.cfg.decision_interval))
        steps = tuple(sorted({T // 6, T // 2, (5 * T) // 6}))
        b = tr.rollout(record_attention_steps=steps)
        np.savez(out / "attention.npz", **{f"t{t}_edges": v[0] for t, v in b["attention"].items()},
                 **{f"t{t}_alpha": v[1] for t, v in b["attention"].items()},
                 **{f"t{t}_eattr": b["eattr"][t] for t in b["attention"]})
    if br_agents:
        nc = sampled_nash_conv(tr, list(br_agents), episodes=br_episodes, eval_episodes=br_eval_episodes)
        save_json(nc, out / "nashconv.json")
    return {"label": label, "seed": seed, "trainer": tr, "eval": ev}
