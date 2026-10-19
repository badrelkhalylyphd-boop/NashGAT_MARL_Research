"""Turn results/tables/*.csv into LaTeX tables in results/latex_tables/ (numbers only, no prose)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nashgat.report import MAIN, NAMES  # noqa: E402

T = ROOT / "results/tables"
O = ROOT / "results/latex_tables"
O.mkdir(parents=True, exist_ok=True)


def ci(r, c, d=1):
    m, lo, hi = r.get(f"{c}_mean"), r.get(f"{c}_ci_lo"), r.get(f"{c}_ci_hi")
    if m is None or pd.isna(m):
        return "--"
    if pd.isna(lo):
        return f"{m:.{d}f}"
    return f"{m:.{d}f} {{\\scriptsize[{lo:.{d}f}, {hi:.{d}f}]}}"


def esc(s: str) -> str:
    return s.replace("_", r"\_").replace("λ", r"$\lambda$").replace("μ", r"$\mu$")


def main_table(S: pd.DataFrame, rule: pd.DataFrame) -> None:
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Method & ATT (s) $\downarrow$ & Wait (s) $\downarrow$ & Queue/int. $\downarrow$ & Throughput $\uparrow$ & $\widehat{\mathcal E}_{\mathrm{BR}}$ $\downarrow$ & Msgs/agent \\",
             r"\midrule"]
    for m in MAIN:
        r = S[S.method == m]
        if r.empty:
            continue
        r = r.iloc[0]
        lines.append(f"{esc(NAMES[m])} ($n={int(r.n)}$) & {ci(r, 'avg_travel_time')} & {ci(r, 'avg_waiting_time')} & "
                     f"{ci(r, 'avg_queue_per_intersection', 2)} & {ci(r, 'throughput', 0)} & {ci(r, 'nash_conv_sampled', 2)} & "
                     f"{ci(r, 'comm_per_agent_step', 2)} \\\\")
    if rule is not None and not rule.empty:
        lines.append(r"\midrule")
        for _, r in rule[rule.scenario == "nominal"].iterrows():
            lines.append(f"{'FixedTime' if r.controller == 'fixed' else 'MaxPressure'} (rule) & {r.avg_travel_time:.1f} & "
                         f"{r.avg_waiting_time:.1f} & {r.avg_queue_per_intersection:.2f} & {r.throughput:.0f} & -- & "
                         f"{r.comm_per_agent_step:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (O / "main_results.tex").write_text("\n".join(lines), encoding="utf-8")


def ablation_table(S: pd.DataFrame) -> None:
    order = ["nashgat", "nashgat_no_gnn", "nashgat_no_attention", "nashgat_no_nash", "nashgat_no_comm_penalty",
             "nashgat_static_graph", "nashgat_lam0.1", "nashgat_lam2.0", "nashgat_mu0.2", "gnn_mappo", "nash_mappo", "nash_mappo_lam2.0",
             "mappo"]
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Variant & ATT (s) & $\sum_iJ_i$ & $\widehat{\mathcal E}_{\mathrm{BR}}$ & Msgs/agent \\", r"\midrule"]
    for m in order:
        r = S[S.method == m]
        if r.empty:
            continue
        r = r.iloc[0]
        lines.append(f"{esc(NAMES.get(m, m))} ($n={int(r.n)}$) & {ci(r, 'avg_travel_time')} & {ci(r, 'J_sum')} & "
                     f"{ci(r, 'nash_conv_sampled', 2)} & {ci(r, 'comm_per_agent_step', 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (O / "ablations.tex").write_text("\n".join(lines), encoding="utf-8")


def robustness_table(R: pd.DataFrame) -> None:
    scen = [("nominal_stochastic", "Nominal"), ("obs_noise_0.1", "Obs. noise"), ("comm_fail_0.3", "Link fail."),
            ("flow_2000", "Flow 2000"), ("flow_2500", "Flow 2500"), ("demand_x1.3", "Demand$\\times$1.3"),
            ("agent_fail_2", "2 failed"), ("transfer_hangzhou", "Hangzhou")]
    lines = [r"\begin{tabular}{l" + "c" * len(scen) + "}", r"\toprule",
             "Method & " + " & ".join(s for _, s in scen) + r" \\", r"\midrule"]
    for m in MAIN:
        d = R[R.method == m].set_index("scenario")
        if d.empty:
            continue
        cells = [f"{d.loc[k, 'att_mean']:.1f}" if k in d.index else "--" for k, _ in scen]
        lines.append(esc(NAMES[m]) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (O / "robustness.tex").write_text("\n".join(lines), encoding="utf-8")


def tests_table(ST: pd.DataFrame) -> None:
    lab = {"avg_travel_time": "ATT", "avg_waiting_time": "Wait", "J_sum": r"$\sum_iJ_i$",
           "nash_conv_sampled": r"$\widehat{\mathcal E}_{\mathrm{BR}}$", "comm_per_agent_step": "Msgs"}
    lines = [r"\begin{tabular}{llrrr}", r"\toprule", r"Metric & vs. & $\Delta$ (\%) & $p_{\mathrm{Welch}}$ & $p_{\mathrm{Holm}}$ \\",
             r"\midrule"]
    for _, r in ST.iterrows():
        rel = "--" if pd.isna(r["rel_change_%"]) or not np.isfinite(r["rel_change_%"]) else f"{r['rel_change_%']:+.1f}"
        lines.append(f"{lab.get(r.metric, r.metric)} & {esc(NAMES.get(r.vs, r.vs))} & {rel} & {r.p_welch:.3g} & {r.p_welch_holm:.3g} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (O / "stat_tests.tex").write_text("\n".join(lines), encoding="utf-8")


def graphical_table(G: pd.DataFrame) -> None:
    lines = [r"\begin{tabular}{llccc}", r"\toprule", r"Game & Objective & $\lambda$ & Welfare & NashConv \\", r"\midrule"]
    for _, r in G.iterrows():
        if r["lambda"] not in (0.0, 0.25, 1.0, 4.0):
            continue
        lines.append(f"{esc(r.game)} & {esc(r.objective)} & {r['lambda']:g} & {r.welfare:.2f} $\\pm$ {r.welfare_sd:.2f} & "
                     f"{r.nash_conv:.2f} $\\pm$ {r.nash_conv_sd:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (O / "graphical_games.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    S = pd.read_csv(T / "summary_all_methods.csv")
    from nashgat.stats import summarise
    W = pd.read_csv(T / "per_seed_nominal.csv")
    S3 = summarise(W[W.seed < 3], "method", ["avg_travel_time", "J_sum", "nash_conv_sampled", "comm_per_agent_step"])
    S3.to_csv(T / "summary_seeds0-2.csv", index=False)
    rule = pd.read_csv(ROOT / "results/rule_based.csv") if (ROOT / "results/rule_based.csv").exists() else None
    main_table(S, rule)
    ablation_table(S3)
    robustness_table(pd.read_csv(T / "robustness.csv"))
    if (T / "stat_tests_nashgat_vs_baselines.csv").exists():
        tests_table(pd.read_csv(T / "stat_tests_nashgat_vs_baselines.csv"))
    if (T / "graphical_games_summary.csv").exists():
        graphical_table(pd.read_csv(T / "graphical_games_summary.csv"))
    print("tables written to", O)
