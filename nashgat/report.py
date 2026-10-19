"""Aggregate experiment outputs into CSV tables, statistics and figures.

Reads <results>/traffic/**, <results>/rule_based.csv, <results>/graphical_games/*.csv
and writes <results>/tables/*.csv and <figures>/*.png|pdf. Only numbers present in
those files are used. Call `configure()` to change the directories.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .stats import compare_to, mean_ci, summarise  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAB = RES / "tables"
FIG = ROOT / "results/figures"
DATA_ROOT = ROOT


def configure(results: str | Path, figures: str | Path, data_root: str | Path | None = None) -> None:
    global RES, TAB, FIG, DATA_ROOT
    RES, FIG = Path(results), Path(figures)
    TAB = RES / "tables"
    if data_root is not None:
        DATA_ROOT = Path(data_root)
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

# fixed categorical order (validated reference palette, light mode)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MAIN = ["ippo", "mappo", "gnn_mappo", "nash_mappo", "nashgat"]
NAMES = {"ippo": "IPPO", "mappo": "MAPPO", "gnn_mappo": "GNN-MAPPO", "nash_mappo": "Nash-MAPPO",
         "nashgat": "NashGAT-MARL", "nashgat_no_gnn": "w/o GNN", "nashgat_no_attention": "w/o attention (mean agg.)",
         "nashgat_no_nash": "w/o Nash reg. (λ=0)", "nashgat_no_comm_penalty": "w/o comm. penalty (μ=0)",
         "nashgat_static_graph": "static graph", "nashgat_lam0.1": "λ=0.1", "nashgat_lam2.0": "λ=2.0",
         "nashgat_mu0.2": "μ=0.2", "nash_mappo_lam2.0": "Nash-MAPPO λ=2.0"}
COLOR = {m: PALETTE[k] for k, m in enumerate(MAIN)}
METRICS = ["avg_travel_time", "avg_waiting_time", "avg_queue_per_intersection", "throughput", "J_sum",
           "comm_per_agent_step"]
LOWER = {"avg_travel_time": True, "avg_waiting_time": True, "avg_queue_per_intersection": True,
         "nash_conv_sampled": True, "nash_conv_sampled_raw": True, "comm_per_agent_step": True}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
                     "axes.labelcolor": "#2b2b29", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 110})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_runs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, hist, br = [], [], []
    for ev in sorted((RES / "traffic").glob("*/seed*/eval.json")):
        label, seed = ev.parent.parent.name, int(ev.parent.name[4:])
        d = json.loads(ev.read_text())
        for scen, m in d.items():
            if isinstance(m, dict):
                rows.append({"method": label, "seed": seed, "scenario": scen,
                             **{k: v for k, v in m.items() if not isinstance(v, list)}})
        nc = ev.parent / "nashconv.json"
        if nc.exists():
            n = json.loads(nc.read_text())
            for p in n["per_agent"]:
                br.append({"method": label, "seed": seed, "agent": p["agent"], "gain": p["gain"], "gain_se": p["gain_se"],
                           "J_base": p["J_base"], "J_br": p["J_br"]})
            rows.append({"method": label, "seed": seed, "scenario": "nashconv",
                         "nash_conv_sampled": n["nash_conv_sampled"], "nash_conv_sampled_raw": n["nash_conv_sampled_raw"]})
        rows.append({"method": label, "seed": seed, "scenario": "cost", "train_time_s": d.get("train_time_s"),
                     "n_actor_params": d.get("n_actor_params")})
        h = pd.read_csv(ev.parent / "history.csv")
        h["method"], h["seed"] = label, seed
        hist.append(h)
    return pd.DataFrame(rows), (pd.concat(hist) if hist else pd.DataFrame()), pd.DataFrame(br)


def wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, seed) with nominal metrics, NashConv, cost and final-gap columns."""
    nom = df[df.scenario == "nominal_stochastic"].set_index(["method", "seed"])[METRICS]
    gr = df[df.scenario == "nominal_greedy"].set_index(["method", "seed"])[["avg_travel_time", "throughput"]]
    gr.columns = ["greedy_" + c for c in gr.columns]
    nc = df[df.scenario == "nashconv"].set_index(["method", "seed"])[["nash_conv_sampled", "nash_conv_sampled_raw"]]
    cost = df[df.scenario == "cost"].set_index(["method", "seed"])[["train_time_s", "n_actor_params"]]
    return nom.join(gr, how="left").join(nc, how="left").join(cost, how="left").reset_index()


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df, hist, br = load_runs()
    if df.empty:
        print("no traffic results yet")
    else:
        df.to_csv(TAB / "all_runs_long.csv", index=False)
        hist.to_csv(TAB / "training_history.csv", index=False)
        br.to_csv(TAB / "best_response_per_agent.csv", index=False)
        W = wide(df)
        last = hist[hist.episode >= hist.episode.max() - 19].groupby(["method", "seed"]).agg(
            final_train_gap=("loss_gap", "mean"), final_gate_rate=("gate_rate", "mean")).reset_index()
        W = W.merge(last, on=["method", "seed"], how="left")
        W.to_csv(TAB / "per_seed_nominal.csv", index=False)
        cols = METRICS + ["greedy_avg_travel_time", "nash_conv_sampled", "nash_conv_sampled_raw", "train_time_s",
                          "final_train_gap"]
        S = summarise(W, "method", [c for c in cols if c in W])
        S.insert(1, "name", S.method.map(NAMES))
        S.to_csv(TAB / "summary_all_methods.csv", index=False)
        main_df = W[W.method.isin(MAIN)]
        if main_df.method.nunique() > 1 and "nashgat" in set(main_df.method):
            cmp = compare_to(main_df, "method", "nashgat",
                             ["avg_travel_time", "avg_waiting_time", "J_sum", "nash_conv_sampled", "comm_per_agent_step"],
                             LOWER)
            cmp.to_csv(TAB / "stat_tests_nashgat_vs_baselines.csv", index=False)
        # robustness
        rob = df[~df.scenario.isin(["nashconv", "cost"])]
        R = rob.groupby(["method", "scenario"]).agg(att_mean=("avg_travel_time", "mean"), att_std=("avg_travel_time", "std"),
                                                     n=("seed", "nunique"), thr_mean=("throughput", "mean"),
                                                     queue_mean=("avg_queue_per_intersection", "mean")).reset_index()
        R.to_csv(TAB / "robustness.csv", index=False)
        figures(W, hist, R, br)
    rb = RES / "rule_based.csv"
    if rb.exists():
        pd.read_csv(rb).to_csv(TAB / "rule_based.csv", index=False)
    graphical_figures()
    print("aggregation complete")


def _band(ax, h, metric, m, smooth=10):
    piv = h[h.method == m].pivot_table(index="episode", columns="seed", values=metric)
    if piv.empty:
        return
    piv = piv.rolling(smooth, min_periods=1).mean()
    mu = piv.mean(1)
    n = piv.notna().sum(1)
    se = piv.std(1) / np.sqrt(n)
    ax.plot(mu.index, mu, color=COLOR[m], lw=2, label=f"{NAMES[m]} (n={int(n.max())})")
    ax.fill_between(mu.index, mu - 1.96 * se, mu + 1.96 * se, color=COLOR[m], alpha=0.15, lw=0)


def figures(W, hist, R, br):
    ms = [m for m in MAIN if m in set(W.method)]
    # training curves
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    for m in ms:
        _band(axs[0], hist, "ep_return_team", m)
        _band(axs[1], hist, "avg_travel_time", m)
        _band(axs[2], hist, "loss_gap", m)
    axs[0].set(title="Team return (training, 10-ep. mean)", xlabel="episode", ylabel="Σ_t mean_i r_i")
    axs[1].set(title="Average travel time (training)", xlabel="episode", ylabel="s")
    axs[2].set(title="Critic exploitability gap Ĝ", xlabel="episode", ylabel="mean gap (return units)")
    axs[0].legend(fontsize=7, loc="lower right")
    save(fig, "training_curves")
    # congestion bars with per-seed dots
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.8))
    for ax, (c, t) in zip(axs, [("avg_travel_time", "Avg. travel time (s)"), ("avg_waiting_time", "Avg. waiting time (s)"),
                                ("avg_queue_per_intersection", "Queue / intersection (veh)"), ("throughput", "Throughput (veh)")]):
        for k, m in enumerate(ms):
            v = W[W.method == m][c].dropna().to_numpy()
            if len(v) == 0:
                continue
            mu, sd, lo, hi = mean_ci(v)
            ax.bar(k, mu, color=COLOR[m], width=0.7, yerr=None if np.isnan(lo) else [[mu - lo], [hi - mu]],
                   capsize=3, error_kw={"lw": 1, "ecolor": "#2b2b29"})
            ax.scatter(np.full(len(v), k) + np.linspace(-0.15, 0.15, len(v)), v, s=9, color="#2b2b29", zorder=3)
        ax.set_xticks(range(len(ms)), [NAMES[m] for m in ms], rotation=35, ha="right", fontsize=7)
        ax.set_title(t, fontsize=9)
    save(fig, "congestion_comparison")
    # exploitability vs welfare and comm vs welfare
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    for m in ms:  # ablations are shown in their own figure
        d = W[W.method == m]
        col = COLOR.get(m, "#9a9994")
        mk = "o" if m in MAIN else "s"
        if d.nash_conv_sampled.notna().any():
            axs[0].errorbar(d.J_sum.mean(), d.nash_conv_sampled.mean(), xerr=d.J_sum.std(), yerr=d.nash_conv_sampled.std(),
                            fmt=mk, color=col, ms=6, capsize=2, label=NAMES.get(m, m))
        axs[1].errorbar(d.comm_per_agent_step.mean(), d.avg_travel_time.mean(), yerr=d.avg_travel_time.std(), fmt=mk,
                        color=col, ms=6, capsize=2, label=NAMES.get(m, m))
    axs[0].set(xlabel="Σ_i J_i (discounted, higher = better)", ylabel="Sampled BR NashConv (3 agents)",
               title="Welfare vs. measured exploitability")
    axs[1].set(xlabel="messages per agent per decision", ylabel="avg. travel time (s)", title="Communication vs. congestion")
    axs[1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "tradeoffs")
    # robustness heatmap-like table figure
    scen = ["nominal_stochastic", "nominal_greedy", "obs_noise_0.1", "comm_fail_0.3", "flow_2000", "flow_2500",
            "demand_x1.3", "agent_fail_2", "transfer_hangzhou"]
    P = R[R.method.isin(ms)].pivot(index="method", columns="scenario", values="att_mean").reindex(index=ms)
    P = P[[s for s in scen if s in P.columns]]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    rel = P / P.min(0)
    im = ax.imshow(rel.to_numpy(), cmap="Blues", aspect="auto", vmin=1.0, vmax=max(1.3, float(np.nanmax(rel.to_numpy()))))
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            v = P.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if rel.iloc[i, j] > 1.2 else "#0b0b0b")
    ax.set_xticks(range(P.shape[1]), P.columns, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(P.shape[0]), [NAMES[m] for m in P.index], fontsize=7)
    ax.grid(False)
    ax.set_title("Average travel time (s) per scenario; shade = ratio to best method in column", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.02)
    save(fig, "robustness")
    # ablations and sweeps
    abl = [m for m in ["nashgat", "nashgat_no_gnn", "nashgat_no_attention", "nashgat_no_nash", "nashgat_no_comm_penalty",
                       "nashgat_static_graph", "nashgat_lam0.1", "nashgat_lam2.0", "nashgat_mu0.2", "nash_mappo",
                       "nash_mappo_lam2.0"] if m in set(W.method)]
    if len(abl) > 1:
        # ablations ran on seeds 0-2: compare every variant on those seeds only
        Wa = W[W.seed < 3]
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
        for ax, (c, t, cap) in zip(axs, [("avg_travel_time", "Avg. travel time (s), seeds 0-2", 340),
                                         ("nash_conv_sampled", "Sampled BR NashConv", 40),
                                         ("comm_per_agent_step", "Messages / agent / step", None)]):
            for k, m in enumerate(abl):
                v = Wa[(Wa.method == m)][c].dropna().to_numpy()
                if len(v) == 0:
                    continue
                col = PALETTE[4] if m == "nashgat" else "#6f6e69"
                shown = np.minimum(v, cap) if cap else v
                ax.scatter(shown, np.full(len(v), k), s=16, color=col, zorder=3)
                ax.plot([v.mean()] * 2 if not cap or v.mean() <= cap else [cap] * 2, [k - 0.3, k + 0.3], color=col, lw=2)
                if cap is not None and (v > cap).any():
                    ax.annotate(", ".join(f"{x:.0f}" for x in v[v > cap]) + " →", (cap, k), xytext=(-4, 5),
                                textcoords="offset points", ha="right", fontsize=6, color="#52514e")
            if cap is not None:
                lo = np.nanmin(Wa[Wa.method.isin(abl)][c]) if c == "avg_travel_time" else 0
                ax.set_xlim(lo - 5 if c == "avg_travel_time" else -1, cap)
            ax.set_yticks(range(len(abl)), [NAMES.get(m, m) for m in abl], fontsize=7)
            ax.invert_yaxis()
            ax.set_title(t, fontsize=9)
        save(fig, "ablations")
    # BR per-agent gains
    if not br.empty:
        fig, ax = plt.subplots(figsize=(6, 2.8))
        for k, m in enumerate(ms):
            d = br[br.method == m]
            ax.scatter(np.full(len(d), k) + (d.agent.to_numpy() - 3) * 0.08, d.gain, color=COLOR[m], s=14)
            ax.hlines(d.gain.mean(), k - 0.3, k + 0.3, color="#2b2b29", lw=1.5)
        ax.axhline(0, color="#8a8984", lw=0.8)
        ax.set_xticks(range(len(ms)), [NAMES[m] for m in ms], rotation=20, fontsize=7)
        ax.set_ylabel("J_k(BR) − J_k(π)")
        ax.set_title("Per-agent best-response gain (agents 0, 1, 5; all seeds)", fontsize=9)
        save(fig, "best_response_gains")
    attention_figure()


def attention_figure():
    import networkx as nx

    from .envs.roadnet import load_roadnet
    cands = sorted((RES / "traffic").glob("nashgat/seed*/attention.npz"))
    if not cands:
        return
    path = cands[0]
    z = np.load(path)
    net = load_roadnet(DATA_ROOT / "data/raw/colight/data/Jinan/3_4/roadnet_3_4.json")
    raw = json.loads((DATA_ROOT / "data/raw/colight/data/Jinan/3_4/roadnet_3_4.json").read_text())
    pos_all = {it["id"]: (it["point"]["x"], it["point"]["y"]) for it in raw["intersections"]}
    agents = net.agents
    steps = sorted({int(k.split("_")[0][1:]) for k in z.files})
    fig, axs = plt.subplots(1, len(steps), figsize=(3.6 * len(steps), 3.2))
    axs = np.atleast_1d(axs)
    for ax, t in zip(axs, steps):
        ei, al = z[f"t{t}_edges"], z[f"t{t}_alpha"]
        G = nx.DiGraph()
        pos = {k: pos_all[a] for k, a in enumerate(agents)}
        G.add_nodes_from(pos)
        self_w = np.zeros(len(agents))
        for (s, d), a in zip(ei.T, al):
            if s == d:
                self_w[d] = a
            else:
                G.add_edge(int(s), int(d), w=float(a))
        widths = [0.5 + 5 * G[u][v]["w"] for u, v in G.edges]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=120 + 400 * self_w, node_color=PALETTE[4], edgecolors="#2b2b29",
                               linewidths=0.6)
        nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color=PALETTE[0], arrows=True, arrowsize=8,
                               connectionstyle="arc3,rad=0.15", node_size=200)
        ax.set_title(f"t = {t * 10} s: {G.number_of_edges()} active msgs", fontsize=8)
        ax.set_axis_off()
    fig.suptitle(f"NashGAT-MARL learned attention ({path.parent.name}, head mean). Edge width ∝ α_ij; node size ∝ self-attention",
                 fontsize=8)
    save(fig, "attention_graph")


def graphical_figures():
    s = RES / "graphical_games/final_profiles.csv"
    if not s.exists():
        return
    df = pd.read_csv(s)
    summ = df.groupby(["game", "objective", "lambda"]).agg(welfare=("welfare", "mean"), welfare_sd=("welfare", "std"),
                                                          nash_conv=("nash_conv", "mean"),
                                                          nash_conv_sd=("nash_conv", "std")).reset_index()
    summ.to_csv(TAB / "graphical_games_summary.csv", index=False)
    games = list(summ.game.unique())
    fig, axs = plt.subplots(1, len(games), figsize=(4.5 * len(games), 3.2))
    axs = np.atleast_1d(axs)
    sty = {"welfare_gap": (PALETTE[0], "o", "welfare + λ·gap (semi-grad.)"),
           "welfare_mix": (PALETTE[1], "x", "welfare + λ·own payoff (mixing)"),
           "welfare_fullgap": (PALETTE[2], "s", "welfare + λ·NashConv (full grad.)")}
    for ax, g in zip(axs, games):
        d = summ[summ.game == g]
        for obj, (c, mk, lab) in sty.items():
            e = d[d.objective == obj].sort_values("lambda")
            ax.plot(e.welfare, e.nash_conv, color=c, marker=mk, lw=1.5, ms=5, label=lab)
            for _, r in e.iterrows():
                if obj == "welfare_fullgap" and r["lambda"] in (0.5, 1.0, 2.0, 4.0):
                    ax.annotate(f"λ={r['lambda']:g}", (r.welfare, r.nash_conv), fontsize=6, xytext=(3, 3),
                                textcoords="offset points", color="#52514e")
        for obj, mk, lab in [("independent", "^", "independent grad."), ("welfare", "v", "welfare only")]:
            e = d[d.objective == obj]
            ax.scatter(e.welfare, e.nash_conv, marker=mk, color="#2b2b29", s=30, label=lab, zorder=4)
        ax.set(xlabel="social welfare Σ_i u_i", ylabel="NashConv (exact)", title=g)
    axs[0].legend(fontsize=6)
    save(fig, "graphical_games_frontier")



