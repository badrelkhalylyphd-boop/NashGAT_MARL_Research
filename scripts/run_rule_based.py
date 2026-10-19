"""Evaluate FixedTime and MaxPressure reference controllers on QueueSim."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nashgat.envs.traffic_env import TrafficMAEnv, default_env_config  # noqa: E402
from nashgat.rule_based import run_rule  # noqa: E402


def main() -> None:
    rows = []
    scen = {"nominal": default_env_config(ROOT, "jinan", "real"),
            "flow_2000": default_env_config(ROOT, "jinan", "2000"),
            "flow_2500": default_env_config(ROOT, "jinan", "2500"),
            "demand_x1.3": replace(default_env_config(ROOT, "jinan", "real"), demand_scale=1.3, departure_jitter=30, seed=11),
            "agent_fail_2": replace(default_env_config(ROOT, "jinan", "real"), failed_agents=[4, 7]),
            "transfer_hangzhou": default_env_config(ROOT, "hangzhou", "real")}
    for sname, cfg in scen.items():
        for pol in ("fixed", "maxpressure"):
            r = run_rule(TrafficMAEnv(cfg), pol)
            rows.append({"scenario": sname, "controller": pol, **r})
    df = pd.DataFrame(rows)
    out = ROOT / "results/rule_based.csv"
    df.to_csv(out, index=False)
    print(df[["scenario", "controller", "avg_travel_time", "avg_waiting_time", "throughput", "avg_queue_per_intersection"]].to_string(index=False))


if __name__ == "__main__":
    main()
