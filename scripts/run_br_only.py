"""Run the best-response exploitability step for an already trained checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nashgat.algos.best_response import sampled_nash_conv  # noqa: E402
from nashgat.algos.mappo import AlgoConfig, MAPPOTrainer  # noqa: E402
from nashgat.envs.traffic_env import EnvConfig, TrafficMAEnv  # noqa: E402
from nashgat.utils import save_json, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--agents", default="0,1,5")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--eval-episodes", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(1)
    run = Path(args.run_dir)
    conf = json.loads((run / "config.json").read_text())
    algo = AlgoConfig(**{f.name: conf["algo"][f.name] for f in fields(AlgoConfig)})
    envc = EnvConfig(**{f.name: conf["env"][f.name] for f in fields(EnvConfig)})
    seed = int(run.name[4:])
    set_seed(seed)
    tr = MAPPOTrainer(TrafficMAEnv(envc), algo, seed=seed)
    tr.load(run / "model.pt")
    t0 = time.time()
    nc = sampled_nash_conv(tr, [int(a) for a in args.agents.split(",")], episodes=args.episodes,
                           eval_episodes=args.eval_episodes)
    nc["time_s"] = time.time() - t0
    nc["note"] = "run by run_br_only.py from saved checkpoint"
    save_json(nc, run / "nashconv.json")
    print("done", run, nc["nash_conv_sampled"])


if __name__ == "__main__":
    main()
