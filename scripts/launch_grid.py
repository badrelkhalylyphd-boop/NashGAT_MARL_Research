"""Run the full experimental grid with a process pool (one CPU thread per job).

Jobs whose eval.json already exists are skipped, so the grid is resumable.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAIN = ["ippo", "mappo", "gnn_mappo", "nash_mappo", "nashgat"]
ABLATIONS = ["nashgat_no_gnn", "nashgat_no_attention", "nashgat_no_nash", "nashgat_no_comm_penalty",
             "nashgat_static_graph"]
SWEEPS = [("nashgat", "lam0.1", ["nash_lambda=0.1"]), ("nashgat", "lam2.0", ["nash_lambda=2.0"]),
          ("nashgat", "mu0.2", ["comm_mu=0.2"]), ("nash_mappo", "lam2.0", ["nash_lambda=2.0"])]


def jobs(args):
    for s in range(args.seeds):
        for m in MAIN:
            yield m, s, "", []
    for s in range(args.ablation_seeds):
        for m in ABLATIONS:
            yield m, s, "", []
        for m, tag, sets in SWEEPS:
            yield m, s, tag, sets


def run(job, args):
    m, s, tag, sets = job
    label = m + (f"_{tag}" if tag else "")
    out = Path(args.out) / label / f"seed{s}"
    if (out / "eval.json").exists() and (args.skip_br or (out / "nashconv.json").exists()):
        return f"skip {label} {s}"
    if (out / "eval.json").exists() and (out / "model.pt").exists():
        # training finished earlier; only the best-response step is missing
        with open(out / "log_br.txt", "w") as fh:
            rc = subprocess.call([sys.executable, str(ROOT / "scripts/run_br_only.py"), str(out)],
                                 stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
        msg = f"{'ok' if rc == 0 else 'FAIL'} BR-only {label} seed{s}"
        print(msg, flush=True)
        return msg
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts/run_traffic.py"), "--method", m, "--seed", str(s),
           "--episodes", str(args.episodes), "--horizon", str(args.horizon), "--out", args.out]
    if tag:
        cmd += ["--tag", tag, "--set", *sets]
    if args.skip_br:
        cmd.append("--skip-br")
    with open(out / "log.txt", "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    msg = f"{'ok' if rc == 0 else 'FAIL'} {label} seed{s}"
    print(msg, flush=True)
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ablation-seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--horizon", type=int, default=3600)
    ap.add_argument("--skip-br", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results/traffic"))
    args = ap.parse_args()
    js = list(jobs(args))
    print(f"{len(js)} jobs", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(lambda j: run(j, args), js))


if __name__ == "__main__":
    main()
