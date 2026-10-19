"""Zero-shot evaluation of Jinan-trained actors on New York 28x7 (196 agents), QueueSim.

Only actor weights are loaded (parameter sharing makes them size-independent).
Writes results/transfer_newyork.csv (one row per method x seed) plus rule-based rows.
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields, replace
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nashgat.algos.mappo import AlgoConfig, MAPPOTrainer  # noqa: E402
from nashgat.envs.traffic_env import TrafficMAEnv, default_env_config  # noqa: E402
from nashgat.rule_based import run_rule  # noqa: E402

METHODS = ["ippo", "mappo", "gnn_mappo", "nash_mappo", "nashgat"]


def main() -> None:
    torch.set_num_threads(2)
    flow = sys.argv[1] if len(sys.argv) > 1 else "double"
    cfg = default_env_config(ROOT, "newyork", flow, horizon=3600)
    rows = []
    for pol in ("fixed", "maxpressure"):
        rows.append({"method": pol, "seed": -1, **run_rule(TrafficMAEnv(cfg), pol)})
        print(pol, rows[-1]["avg_travel_time"], flush=True)
    for m in METHODS:
        for run in sorted((ROOT / "results/traffic" / m).glob("seed*")):
            if not (run / "model.pt").exists():
                continue
            conf = json.loads((run / "config.json").read_text())
            algo = AlgoConfig(**{f.name: conf["algo"][f.name] for f in fields(AlgoConfig)})
            env = TrafficMAEnv(cfg)
            tr = MAPPOTrainer(env, replace(algo, critic="local"), seed=int(run.name[4:]))
            ck = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
            tr.actor.load_state_dict(ck["actor"])
            ev = tr.evaluate(1)
            rows.append({"method": m, "seed": int(run.name[4:]), **{k: v for k, v in ev.items() if not isinstance(v, list)}})
            print(m, run.name, round(ev["avg_travel_time"], 1), flush=True)
    out = ROOT / f"results/transfer_newyork_{flow}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
