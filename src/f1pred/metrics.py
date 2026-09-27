"""Scoring a forecast against what happened.

Ranking and probability are graded separately, because a model can order the
field well and still be over-confident, or the reverse.

    ranking      NDCG@3, NDCG@5, winner hit, podium and top-5 overlap,
                 Spearman and Kendall rank correlation
    probability  win log loss, win Brier (over the field), podium and top-10
                 Brier (per driver), and reliability: when it says 30%, does
                 it happen 30% of the time

Comparisons between two approaches are paired by race - both are graded on the
same races - and the uncertainty in the difference comes from resampling
races (bootstrap), since races, not drivers, are the independent unit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-12

# Lower is better for these; higher for everything else.
LOWER_IS_BETTER = {
    "win_logloss",
    "win_brier",
    "podium_brier",
    "top10_brier",
    "pole_logloss",
    "driver_points_mae",
    "team_points_mae",
    "teammate_gap_mae",
    "champion_logloss",
}


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def ndcg_at_k(pred_order: list[str], true_rank: dict[str, int], k: int = 5) -> float:
    """Order-sensitive top-k quality. Relevance = 1/log2(true_position+1), so
    putting the actual winner first is worth more than the actual fifth."""

    def rel(driver: str) -> float:
        pos = true_rank.get(driver)
        return 1.0 / np.log2(pos + 1) if pos else 0.0

    dcg = sum(rel(d) / np.log2(i + 2) for i, d in enumerate(pred_order[:k]))
    ideal = sorted((rel(d) for d in true_rank), reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def ranking(pred_rank: pd.Series, actual: pd.Series) -> dict:
    """pred_rank and actual: finishing positions indexed by driver_id (1 = first)."""
    d = pd.DataFrame({"pred": pred_rank, "actual": actual}).dropna()
    if len(d) < 3:
        return {}
    pred_order = d.sort_values("pred").index.tolist()
    true_order = d.sort_values("actual").index.tolist()
    true_rank = {k: int(v) for k, v in d["actual"].rank(method="first").items()}
    return {
        "ndcg3": ndcg_at_k(pred_order, true_rank, 3),
        "ndcg5": ndcg_at_k(pred_order, true_rank, 5),
        "winner_hit": int(pred_order[0] == true_order[0]),
        "podium_overlap": len(set(pred_order[:3]) & set(true_order[:3])),
        "top5_overlap": len(set(pred_order[:5]) & set(true_order[:5])),
        "spearman": float(stats.spearmanr(d["pred"], d["actual"]).statistic),
        "kendall": float(stats.kendalltau(d["pred"], d["actual"]).statistic),
    }


# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------
def probability(table: pd.DataFrame, actual: pd.Series) -> dict:
    """table: p_win, p_podium, p_top10 indexed by driver_id; actual: positions.

    Win log loss is -log P(actual winner). Win Brier is summed over the field
    (the multi-class Brier score); podium and top-10 Brier are per driver, as
    those are yes/no questions asked of each driver.
    """
    d = table.join(actual.rename("actual"), how="inner").dropna(subset=["actual"])
    if d.empty:
        return {}
    won = (d["actual"] == d["actual"].min()).astype(float)
    podium = (d["actual"] <= 3).astype(float)
    top10 = (d["actual"] <= 10).astype(float)
    p_win = d["p_win"] / d["p_win"].sum()
    return {
        "win_logloss": float(-np.log(max(float(p_win[won == 1].iloc[0]), EPS))),
        "win_brier": float(((p_win - won) ** 2).sum()),
        "podium_brier": float(((d["p_podium"] - podium) ** 2).mean()) if "p_podium" in d else np.nan,
        "top10_brier": float(((d["p_top10"] - top10) ** 2).mean()) if "p_top10" in d else np.nan,
        "p_winner": float(p_win[won == 1].iloc[0]),
    }


def outcome_rows(table: pd.DataFrame, actual: pd.Series, **keys) -> pd.DataFrame:
    """One row per driver: stated probabilities beside what happened, for reliability."""
    d = table.join(actual.rename("actual"), how="inner").dropna(subset=["actual"])
    out = pd.DataFrame(
        {
            "driver_id": d.index,
            "p_win": d["p_win"].to_numpy(),
            "won": (d["actual"] == d["actual"].min()).astype(int).to_numpy(),
            "p_podium": d["p_podium"].to_numpy() if "p_podium" in d else np.nan,
            "podium": (d["actual"] <= 3).astype(int).to_numpy(),
            "p_top10": d["p_top10"].to_numpy() if "p_top10" in d else np.nan,
            "top10": (d["actual"] <= 10).astype(int).to_numpy(),
        }
    )
    for k, v in keys.items():
        out[k] = v
    return out


def reliability(p: pd.Series, y: pd.Series, bins: tuple[float, ...] | None = None) -> pd.DataFrame:
    """Stated probability against observed frequency, per bucket, with a Wilson
    interval so a bucket of eight races isn't read as a verdict."""
    edges = bins or (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)
    d = pd.DataFrame({"p": p.to_numpy(), "y": y.to_numpy()})
    d["bucket"] = pd.cut(d["p"], list(edges), include_lowest=True)
    g = d.groupby("bucket", observed=True).agg(stated=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    g = g[g["n"] > 0].reset_index()
    intervals = [wilson(round(r.observed * r.n), int(r.n)) for r in g.itertuples()]
    g["ci_low"] = [lo for lo, _ in intervals]
    g["ci_high"] = [hi for _, hi in intervals]
    g["bucket"] = g["bucket"].astype(str)
    return g


def expected_calibration_error(p: pd.Series, y: pd.Series, n_bins: int = 10) -> float:
    """Mean |stated - observed| over equal-count buckets, weighted by size."""
    d = pd.DataFrame({"p": p.to_numpy(), "y": y.to_numpy()}).sort_values("p")
    if d.empty:
        return float("nan")
    groups = np.array_split(np.arange(len(d)), min(n_bins, len(d)))
    total = 0.0
    for idx in groups:
        part = d.iloc[idx]
        total += len(part) * abs(part["p"].mean() - part["y"].mean())
    return total / len(d)


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = hits / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def bootstrap_mean(values: np.ndarray, n_boot: int = 5000, seed: int = 0, level: float = 0.95) -> tuple:
    """Mean and percentile interval from resampling the races."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(v.mean()), float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def paired_comparison(
    races: pd.DataFrame, a: str, b: str, metrics: list[str], n_boot: int = 5000, seed: int = 0
) -> pd.DataFrame:
    """Race-by-race difference a - b on each metric, with a bootstrap interval.

    races: one row per (method, season, round). Only races both methods were
    graded on count. `better` says which side the interval supports, or
    'unclear' when it straddles zero.
    """
    keys = ["season", "round"]
    left = races[races["method"] == a].set_index(keys)
    right = races[races["method"] == b].set_index(keys)
    common = left.index.intersection(right.index)
    rows = []
    for m in metrics:
        if m not in left or m not in right:
            continue
        diff = (left.loc[common, m] - right.loc[common, m]).to_numpy(dtype=float)
        mean, lo, hi = bootstrap_mean(diff, n_boot, seed)
        lower_better = m in LOWER_IS_BETTER
        if lo > 0:
            better = b if lower_better else a
        elif hi < 0:
            better = a if lower_better else b
        else:
            better = "unclear"
        rows.append(
            {
                "metric": m,
                a: float(left.loc[common, m].mean()),
                b: float(right.loc[common, m].mean()),
                "difference": mean,
                "ci_low": lo,
                "ci_high": hi,
                "better": better,
                "n_races": len(common),
            }
        )
    return pd.DataFrame(rows)
