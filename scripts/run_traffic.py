"""Train one (method, seed) on the traffic game and run all evaluations.

Outputs (under --out/<method>/seed<k>/):
  history.csv         per-episode training log
  model.pt            actor/critic checkpoint
  eval.json           nominal + robustness + transfer evaluations
  nashconv.json       approximate best-response exploitability on sampled agents
  attention.npz       GAT attention weights at a few time steps (GAT actors only)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nashgat.algos.best_response import sampled_nash_conv  # noqa: E402
from nashgat.algos.mappo import MAPPOTrainer, make_config  # noqa: E402
from nashgat.envs.traffic_env import TrafficMAEnv, default_env_config  # noqa: E402
from nashgat.utils import save_json, set_seed  # noqa: E402


def evaluate_suite(tr: MAPPOTrainer, args) -> dict:
    res = {}
    base = tr.env.cfg
    res["nominal_stochastic"] = tr.evaluate(args.eval_episodes)
    res["nominal_greedy"] = tr.evaluate(1, greedy=True)
    if args.skip_robustness:
        return res
    # observation noise
    res["obs_noise_0.1"] = tr.evaluate(args.eval_episodes, env=TrafficMAEnv(replace(base, obs_noise_std=0.1, seed=7)))
    # communication failures (each active message lost w.p. 0.3)
    old = tr.graph.fail_prob
    tr.graph.fail_prob = 0.3
    res["comm_fail_0.3"] = tr.evaluate(args.eval_episodes)
    tr.graph.fail_prob = old
    # demand changes: other published Jinan flow files and a seeded +30% perturbation
    for fv in ("2000", "2500"):
        cfg = default_env_config(ROOT, "jinan", fv, horizon=base.horizon)
        res[f"flow_{fv}"] = tr.evaluate(args.eval_episodes, env=TrafficMAEnv(cfg))
    res["demand_x1.3"] = tr.evaluate(args.eval_episodes,
                                     env=TrafficMAEnv(replace(base, demand_scale=1.3, departure_jitter=30, seed=11)))
    # agent failures: two controllers fall back to a fixed-time plan
    res["agent_fail_2"] = tr.evaluate(args.eval_episodes, env=TrafficMAEnv(replace(base, failed_agents=[4, 7])))
    # zero-shot transfer to the larger Hangzhou 4x4 network (actor weights only)
    hz = TrafficMAEnv(default_env_config(ROOT, "hangzhou", "real", horizon=base.horizon))
    tr_hz = MAPPOTrainer(hz, replace(tr.cfg, critic="local"), seed=tr.seed)
    tr_hz.actor.load_state_dict(tr.actor.state_dict())
    res["transfer_hangzhou"] = tr_hz.evaluate(args.eval_episodes)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--horizon", type=int, default=3600)
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--br-episodes", type=int, default=20)
    ap.add_argument("--br-eval-episodes", type=int, default=5)
    ap.add_argument("--br-agents", default="0,1,5")
    ap.add_argument("--skip-br", action="store_true")
    ap.add_argument("--skip-robustness", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--out", default=str(ROOT / "results/traffic"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(1)
    set_seed(args.seed)
    overrides = {}
    for kv in args.set:
        k, v = kv.split("=")
        overrides[k] = type(getattr(make_config(args.method), k))(v) if not isinstance(getattr(make_config(args.method), k), bool) else v.lower() == "true"
    cfg = make_config(args.method, episodes=args.episodes, **overrides)
    label = args.method + (f"_{args.tag}" if args.tag else "")
    cfg = replace(cfg, name=label)
    out = Path(args.out) / label / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    env = TrafficMAEnv(default_env_config(ROOT, "jinan", "real", horizon=args.horizon, seed=args.seed))
    tr = MAPPOTrainer(env, cfg, seed=args.seed, device=args.device)
    save_json({"algo": asdict(cfg), "env": asdict(env.cfg), "args": vars(args)}, out / "config.json")
    t0 = time.time()
    tr.train(args.episodes)
    train_time = time.time() - t0
    pd.DataFrame(tr.history).to_csv(out / "history.csv", index=False)
    tr.save(out / "model.pt")
    t1 = time.time()
    ev = evaluate_suite(tr, args)
    ev["train_time_s"] = train_time
    ev["eval_time_s"] = time.time() - t1
    ev["n_actor_params"] = sum(p.numel() for p in tr.actor.parameters())
    save_json(ev, out / "eval.json")
    if cfg.actor_encoder == "gat":
        b = tr.rollout(record_attention_steps=(60, 180, 300))
        np.savez(out / "attention.npz", **{f"t{t}_edges": v[0] for t, v in b["attention"].items()},
                 **{f"t{t}_alpha": v[1] for t, v in b["attention"].items()},
                 **{f"t{t}_eattr": b["eattr"][t] for t in b["attention"]})
    if not args.skip_br:
        t2 = time.time()
        nc = sampled_nash_conv(tr, [int(a) for a in args.br_agents.split(",")], episodes=args.br_episodes,
                               eval_episodes=args.br_eval_episodes)
        nc["time_s"] = time.time() - t2
        save_json(nc, out / "nashconv.json")
    print(json.dumps({"label": label, "seed": args.seed, "train_time_s": round(train_time),
                      "att": ev["nominal_stochastic"]["avg_travel_time"]}), flush=True)


if __name__ == "__main__":
    main()
