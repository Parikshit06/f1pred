"""Walk-forward evaluation.

For every race in the test window: train on races strictly before it, predict
it, score the prediction. No shuffled splits - those let the model see the
future, which is the most common way a sports model flatters its author.

Everything is scored against baselines. A metric without a baseline is
decoration: "we got 38% of winners right" only means something next to "the
pole sitter won 41% of the time".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, model, simulate

log = logging.getLogger(__name__)

MIN_TRAIN_RACES = 30


# ---------------------------------------------------------------------------
# Metrics
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


def score_race(pred: pd.DataFrame, actual_col: str, k: int = 5) -> dict:
    """pred needs: driver_id, p_win, pred_rank, and the actual position column."""
    d = pred.dropna(subset=[actual_col]).copy()
    if d.empty:
        return {}

    d = d.sort_values("pred_rank")
    pred_top = d["driver_id"].head(k).tolist()
    actual = d.sort_values(actual_col)
    actual_top = actual["driver_id"].head(k).tolist()
    winner = actual_top[0] if actual_top else None

    true_rank = dict(zip(d["driver_id"], d[actual_col].astype(int)))
    p = d.set_index("driver_id")["p_win"]
    p = p / p.sum() if p.sum() > 0 else p

    p_winner = float(p.get(winner, 1e-12))
    brier = float(((p - (d.set_index("driver_id")[actual_col] == 1).astype(float)) ** 2).sum())

    return {
        "top5_overlap": len(set(pred_top) & set(actual_top)),
        "top1_hit": int(pred_top[0] == winner) if pred_top else 0,
        "podium_overlap": len(set(pred_top[:3]) & set(actual_top[:3])),
        "ndcg5": ndcg_at_k(d["driver_id"].tolist(), true_rank, k),
        "spearman": float(d[actual_col].corr(d["pred_rank"], method="spearman")),
        "logloss": -np.log(max(p_winner, 1e-12)),
        "brier": brier,
        "p_winner": p_winner,
        "n_drivers": len(d),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def baseline_predictions(race: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Each baseline produces the same shape as the model so scoring is shared."""
    d = race.copy()
    n = len(d)

    if kind == "grid":
        d["pred_rank"] = d["grid"].fillna(n).rank(method="first")
    elif kind == "championship":
        d["pred_rank"] = d["champ_position_before"].fillna(n).rank(method="first")
    elif kind == "recent_form":
        d["pred_rank"] = d["drv_avg_finish_3"].fillna(n).rank(method="first")
    elif kind == "team_form":
        d["pred_rank"] = d["team_avg_finish_3"].fillna(n).rank(method="first")
    else:
        raise ValueError(kind)

    # Crude but honest probabilities: geometric decay down the predicted order.
    w = 0.55 ** (d["pred_rank"].to_numpy() - 1)
    d["p_win"] = w / w.sum()
    return d


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    races: pd.DataFrame = field(default_factory=pd.DataFrame)
    temperature: float = 1.0
    blend_weight: float = 0.5

    def summary(self) -> pd.DataFrame:
        if self.races.empty:
            return pd.DataFrame()
        metrics = ["top5_overlap", "podium_overlap", "top1_hit", "ndcg5", "spearman", "logloss", "brier"]
        out = self.races.groupby("method")[metrics].mean().round(3)
        out["n_races"] = self.races.groupby("method").size()
        # Model first, then baselines by quality.
        return out.sort_values("ndcg5", ascending=False)

    def calibration(self, bins: int = 5) -> pd.DataFrame:
        # Guard the frame before the column, the way summary() does. A run that
        # graded nothing returns a DataFrame with no columns at all, and
        # `self.races.method` on one of those raises AttributeError - which is
        # how `verify` turned "the backtest produced no races" into a traceback
        # instead of the FAIL it had already written.
        if self.races.empty or "method" not in self.races.columns:
            return pd.DataFrame()
        d = self.races[self.races.method == "model"]
        if d.empty:
            return pd.DataFrame()
        # Edges have to span the column being binned. They were built from
        # p_winner - the probability given to whoever actually won - while the
        # cut is on p_top_pick, the probability given to the model's own pick.
        # p_top_pick is the larger of the two whenever the favourite lost, so
        # the top edge sat below the data and pd.cut returned NaN for the most
        # confident races, which then vanished from the table: 61 of 62 races
        # reported, with the missing one exactly the kind you most want graded.
        edges = np.linspace(0, max(0.6, float(d["p_top_pick"].max())), bins + 1)
        d = d.assign(bucket=pd.cut(d["p_top_pick"], edges, include_lowest=True))
        g = d.groupby("bucket", observed=True).agg(
            predicted=("p_top_pick", "mean"), actual=("top1_hit", "mean"), n=("top1_hit", "size")
        )
        return g.round(3).reset_index()


def walk_forward(
    df: pd.DataFrame,
    start_season: int,
    end_season: int | None = None,
    retrain_every: int = 1,
    current_season_weight: float = config.DEFAULT_CURRENT_SEASON_WEIGHT,
    n_sims: int = 2000,
    blend_weight: float = config.DEFAULT_BLEND_WEIGHT,
    temperature: float = config.DEFAULT_TEMPERATURE,
    adaptive_temperature: bool = True,
) -> BacktestResult:
    df = df.sort_values(["race_seq", "driver_id"]).reset_index(drop=True)
    race_keys = df[["race_seq", "season", "round"]].drop_duplicates().sort_values("race_seq")
    targets = race_keys[race_keys["season"] >= start_season]
    if end_season is not None:
        targets = targets[targets["season"] <= end_season]

    rows: list[dict] = []
    ranker: model.Ranker | None = None
    since_fit = 10**9
    # Trailing record of (scores, who actually won), used to re-fit confidence
    # from recent races only. Finished races only, so no future leaks in.
    seen_scores: list[np.ndarray] = []
    seen_winners: list[int] = []
    live_temperature = temperature

    for _, key in targets.iterrows():
        seq = int(key["race_seq"])
        train = df[df["race_seq"] < seq]
        if train["race_seq"].nunique() < MIN_TRAIN_RACES:
            continue

        if since_fit >= retrain_every or ranker is None:
            ranker = model.train_race(train, current_season_weight=current_season_weight)
            since_fit = 0
        since_fit += 1

        race = df[df["race_seq"] == seq].copy()
        if race["position"].notna().sum() < 5:
            continue

        race["score"] = ranker.score(race)
        race["pred_rank"] = (-race["score"]).rank(method="first")

        if adaptive_temperature:
            live_temperature = simulate.rolling_temperature(seen_scores, seen_winners, temperature)

        p_model = simulate.plackett_luce(race["score"].to_numpy(), live_temperature)
        sim = simulate.simulate(
            simulate.SimInputs(
                driver_ids=race["driver_id"].tolist(),
                scores=race["score"].to_numpy(),
                dnf_prob=simulate.dnf_probability(race),
                grid=race["grid"].fillna(len(race)).to_numpy(),
                overtaking_score=float(race["circuit_overtaking_score"].iloc[0])
                if pd.notna(race["circuit_overtaking_score"].iloc[0])
                else 3.0,
            ),
            n_sims=n_sims,
            temperature=live_temperature,
        )
        race["p_win"] = simulate.blend(p_model, sim["p_win"].to_numpy(), blend_weight)

        # Record the outcome AFTER predicting it, so it can only inform later races.
        positions = race["position"].to_numpy()
        if np.isfinite(np.where(np.isnan(positions), np.inf, positions)).any():
            seen_scores.append(race["score"].to_numpy())
            seen_winners.append(int(np.nanargmin(np.where(np.isnan(positions), np.inf, positions))))

        scored = score_race(race, "position")
        if scored:
            rows.append(
                {
                    "method": "model",
                    "season": int(key["season"]),
                    "round": int(key["round"]),
                    "p_top_pick": float(race.sort_values("pred_rank")["p_win"].iloc[0]),
                    "temperature": live_temperature,
                    **scored,
                }
            )

        for kind in ("grid", "championship", "recent_form", "team_form"):
            b = baseline_predictions(race, kind)
            s = score_race(b, "position")
            if s:
                rows.append(
                    {
                        "method": kind,
                        "season": int(key["season"]),
                        "round": int(key["round"]),
                        "p_top_pick": float(b.sort_values("pred_rank")["p_win"].iloc[0]),
                        **s,
                    }
                )

    return BacktestResult(pd.DataFrame(rows), temperature, blend_weight)


# ---------------------------------------------------------------------------
# Hyperparameter choices that would otherwise be guesses
# ---------------------------------------------------------------------------
def tune_temperature(
    df: pd.DataFrame, start_season: int, end_season: int | None = None, retrain_every: int = 4
) -> float:
    """Fit the Plackett-Luce temperature on out-of-sample races."""
    df = df.sort_values(["race_seq", "driver_id"]).reset_index(drop=True)
    keys = df[["race_seq", "season"]].drop_duplicates().sort_values("race_seq")
    keys = keys[keys["season"] >= start_season]
    if end_season is not None:
        keys = keys[keys["season"] <= end_season]

    groups, winners = [], []
    ranker, since = None, 10**9
    for _, k in keys.iterrows():
        seq = int(k["race_seq"])
        train = df[df["race_seq"] < seq]
        if train["race_seq"].nunique() < MIN_TRAIN_RACES:
            continue
        if since >= retrain_every or ranker is None:
            ranker = model.train_race(train)
            since = 0
        since += 1

        race = df[df["race_seq"] == seq]
        if race["position"].notna().sum() < 5:
            continue
        scores = ranker.score(race)
        winner_pos = race["position"].to_numpy()
        idx = int(np.nanargmin(np.where(np.isnan(winner_pos), np.inf, winner_pos)))
        groups.append(scores)
        winners.append(idx)

    t = simulate.fit_temperature(groups, winners)
    log.info("Fitted temperature %.3f on %d races", t, len(groups))
    return t


def tune_blend(
    df: pd.DataFrame,
    start_season: int,
    temperature: float,
    end_season: int | None = None,
    retrain_every: int = 4,
) -> float:
    """Pick the model/simulation blend by log loss, not by preference."""
    best_w, best_loss = 0.5, np.inf
    for w in np.linspace(0.0, 1.0, 11):
        res = walk_forward(
            df,
            start_season,
            end_season=end_season,
            retrain_every=retrain_every,
            n_sims=800,
            blend_weight=float(w),
            temperature=temperature,
        )
        m = res.races[res.races.method == "model"]
        if m.empty:
            continue
        loss = float(m["logloss"].mean())
        if loss < best_loss:
            best_w, best_loss = float(w), loss
    log.info("Best blend weight %.2f (log loss %.3f)", best_w, best_loss)
    return best_w


def tune_recency(
    df: pd.DataFrame, start_season: int, end_season: int | None = None, retrain_every: int = 4
) -> float:
    """The reel weighted the current season 3x because it felt right. Measure it."""
    best_w, best_loss = config.DEFAULT_CURRENT_SEASON_WEIGHT, np.inf
    for w in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        res = walk_forward(
            df,
            start_season,
            end_season=end_season,
            retrain_every=retrain_every,
            n_sims=400,
            current_season_weight=w,
        )
        m = res.races[res.races.method == "model"]
        if m.empty:
            continue
        loss = float(m["logloss"].mean())
        log.info("  recency weight %.1f -> log loss %.3f", w, loss)
        if loss < best_loss:
            best_w, best_loss = w, loss
    return best_w
