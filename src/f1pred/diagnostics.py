"""Is the model biased toward particular drivers?

The obvious way to ask is to average, per driver, how far the predicted rank
sits from the actual one. That answer is wrong, and wrong in a way that looks
convincing: it says the model badly over-rates every front-runner.

It does not. Finishing position is noisy in one direction at each end of the
grid. A fast driver who retires is classified near last, so their *mean* result
sits well below their true pace; a slow driver inherits places when quicker
cars break, so theirs sits above. Comparing a deterministic ranking against a
mean that has been pulled toward the middle manufactures a bias gradient
running the length of the grid.

On 2024-26 that gradient explained 66% of the variance in per-driver bias, and
retirement rate correlated 0.51 with it. So this module fits the trend and
reports the residual - what is left once position on the grid is accounted for.
That residual is the part that is actually about the driver.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import backtest, features, model

log = logging.getLogger(__name__)

MIN_RACES = 15


def collect(df: pd.DataFrame, start_season: int, retrain_every: int = 3) -> pd.DataFrame:
    """Walk-forward predictions with the error attached, one row per entry."""
    df = df.sort_values(["race_seq", "driver_id"]).reset_index(drop=True)
    keys = df[["race_seq", "season"]].drop_duplicates().sort_values("race_seq")

    frames: list[pd.DataFrame] = []
    ranker: model.Ranker | None = None
    since = 10**9

    for _, key in keys[keys["season"] >= start_season].iterrows():
        seq = int(key["race_seq"])
        train = df[df["race_seq"] < seq]
        if train["race_seq"].nunique() < backtest.MIN_TRAIN_RACES:
            continue
        if since >= retrain_every or ranker is None:
            ranker = model.train_race(train)
            since = 0
        since += 1

        race = df[df["race_seq"] == seq].copy()
        if race["position"].notna().sum() < 5:
            continue

        race["predicted_rank"] = pd.Series(-ranker.score(race), index=race.index).rank(method="first")
        race["actual_rank"] = race["position"].rank(method="first")
        # Negative means the model placed them better than they finished.
        race["error"] = race["predicted_rank"] - race["actual_rank"]
        frames.append(
            race[["season", "round", "driver_id", "constructor_id", "predicted_rank", "actual_rank", "error"]]
        )

    if not frames:
        raise RuntimeError("no races scored - is start_season inside the data?")
    return pd.concat(frames, ignore_index=True)


def driver_bias(entries: pd.DataFrame, min_races: int = MIN_RACES) -> pd.DataFrame:
    """Per-driver bias, before and after removing the grid-position trend."""
    g = (
        entries.groupby("driver_id")
        .agg(
            races=("error", "size"),
            raw_bias=("error", "mean"),
            mean_abs_error=("error", lambda s: s.abs().mean()),
            avg_predicted=("predicted_rank", "mean"),
            avg_actual=("actual_rank", "mean"),
        )
        .query("races >= @min_races")
    )
    if len(g) < 3:
        g["driver_specific_bias"] = np.nan
        return g.sort_values("raw_bias")

    slope, intercept = np.polyfit(g["avg_predicted"], g["raw_bias"], 1)
    g["expected_from_rank"] = slope * g["avg_predicted"] + intercept
    g["driver_specific_bias"] = g["raw_bias"] - g["expected_from_rank"]
    return g.sort_values("driver_specific_bias")


def trend_strength(entries: pd.DataFrame, min_races: int = MIN_RACES) -> dict:
    """How much of the apparent bias is just a function of where they rank."""
    g = (
        entries.groupby("driver_id")
        .agg(races=("error", "size"), raw_bias=("error", "mean"), avg_predicted=("predicted_rank", "mean"))
        .query("races >= @min_races")
    )
    if len(g) < 3:
        return {"r": float("nan"), "variance_explained": float("nan")}
    r = float(np.corrcoef(g["avg_predicted"], g["raw_bias"])[0, 1])
    return {"r": r, "variance_explained": r**2, "n_drivers": len(g)}


def report(start_season: int = 2024, retrain_every: int = 3) -> str:
    entries = collect(features.load(), start_season, retrain_every)
    bias = driver_bias(entries)
    trend = trend_strength(entries)

    lines = [
        f"Driver bias, {start_season}+ walk-forward ({entries['round'].size} entries)",
        "",
        (
            f"Apparent bias correlates {trend['r']:.3f} with predicted rank, so "
            f"{trend['variance_explained'] * 100:.0f}% of it is an artifact of"
        ),
        "comparing a fixed ranking against a mean pulled toward the middle by retirements.",
        "The column that matters is driver_specific_bias - what survives that correction.",
        "",
        "negative = model rates them better than they finish",
        "",
        bias[["races", "avg_predicted", "avg_actual", "raw_bias", "driver_specific_bias"]]
        .round(2)
        .to_string(),
        "",
        f"field mean absolute error: {entries['error'].abs().mean():.2f} positions",
    ]
    return "\n".join(lines)
