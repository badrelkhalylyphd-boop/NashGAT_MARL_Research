"""Statistical summaries for multi-seed comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(x: np.ndarray, conf: float = 0.95) -> tuple[float, float, float, float]:
    """Mean, sample std, and Student-t confidence interval half-width bounds."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean())
    s = float(x.std(ddof=1)) if n > 1 else float("nan")
    if n < 2:
        return m, s, float("nan"), float("nan")
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * s / np.sqrt(n)
    return m, s, m - h, m + h


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    r = stats.ttest_ind(a, b, equal_var=False)
    return float(r.statistic), float(r.pvalue)


def mann_whitney(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        adj[i] = running
    return adj.tolist()


def summarise(df: pd.DataFrame, group: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for g, d in df.groupby(group, sort=False):
        row = {group: g, "n": len(d)}
        for m in metrics:
            v = d[m].dropna().to_numpy()
            if len(v) == 0:
                row.update({f"{m}_mean": float("nan"), f"{m}_std": float("nan"), f"{m}_ci_lo": float("nan"),
                            f"{m}_ci_hi": float("nan"), f"{m}_n": 0})
                continue
            mu, sd, lo, hi = mean_ci(v)
            row[f"{m}_n"] = len(v)
            row.update({f"{m}_mean": mu, f"{m}_std": sd, f"{m}_ci_lo": lo, f"{m}_ci_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def compare_to(df: pd.DataFrame, group: str, ref: str, metrics: list[str], lower_is_better: dict[str, bool]) -> pd.DataFrame:
    """Relative change of `ref` against every other group with Welch and Mann-Whitney
    tests; Holm correction is applied per metric across the comparisons."""
    rows = []
    a_all = df[df[group] == ref]
    for m in metrics:
        tmp = []
        for g, d in df.groupby(group, sort=False):
            if g == ref:
                continue
            a, b = a_all[m].dropna().to_numpy(), d[m].dropna().to_numpy()
            if len(a) < 2 or len(b) < 2:
                continue
            t, p = welch(a, b)
            rel = (a.mean() - b.mean()) / abs(b.mean()) * 100 if b.mean() != 0 else float("nan")
            better = (a.mean() < b.mean()) if lower_is_better.get(m, False) else (a.mean() > b.mean())
            tmp.append({"metric": m, "vs": g, "ref_mean": a.mean(), "other_mean": b.mean(), "rel_change_%": rel,
                        "ref_better": bool(better), "welch_t": t, "p_welch": p, "p_mwu": mann_whitney(a, b)})
        adj = holm([r["p_welch"] for r in tmp])
        for r, q in zip(tmp, adj):
            r["p_welch_holm"] = q
        rows += tmp
    return pd.DataFrame(rows)
