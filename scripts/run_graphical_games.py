"""Exact-exploitability experiments on graphical games with known equilibria.

Produces results/graphical_games/*.csv and figures. Every number is computed in
closed form from the learned mixed profile (no estimation noise).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nashgat.games.polymatrix import (grid_anti_coordination, networked_prisoners_dilemma,  # noqa: E402
                                      train_softmax_profile)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/graphical_games")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    games = {"npd_3x4": networked_prisoners_dilemma(3, 4), "anticoord_3x4": grid_anti_coordination(3, 4)}
    rows, curves = [], []
    for gname, g in games.items():
        settings = ([("independent", 0.0), ("welfare", 0.0)]
                    + [(o, l) for o in ("welfare_gap", "welfare_mix", "welfare_fullgap") for l in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)])
        for obj, lam in settings:
            for s in range(args.seeds):
                res = train_softmax_profile(g, obj, lam=lam, steps=args.steps, seed=s, init_scale=1.0)
                rows.append({"game": gname, "objective": obj, "lambda": lam, "seed": s, "welfare": res["welfare"],
                             "nash_conv": res["nash_conv"],
                             "p_action0_mean": float(res["x"][:, 0].mean())})
                for t, w, nc in res["history"]:
                    curves.append({"game": gname, "objective": obj, "lambda": lam, "seed": s, "step": t,
                                   "welfare": w, "nash_conv": nc})
        print(gname, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "final_profiles.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "curves.csv", index=False)
    summ = df.groupby(["game", "objective", "lambda"]).agg(welfare_mean=("welfare", "mean"), welfare_std=("welfare", "std"),
                                                         nash_conv_mean=("nash_conv", "mean"),
                                                         nash_conv_std=("nash_conv", "std"),
                                                         p0_mean=("p_action0_mean", "mean")).reset_index()
    summ.to_csv(out / "summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
