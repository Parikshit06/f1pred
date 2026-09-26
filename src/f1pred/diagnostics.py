"""Per-driver bias, corrected for where each driver sits on the grid.

Raw bias (mean predicted rank minus mean actual) says every front-runner is
over-rated. That's an artefact: a fast driver who retires is classified near
last, dragging their mean result down, while slow drivers inherit places.
The gradient that creates runs the length of the grid.

So the trend against predicted rank is fitted and removed; the residual is
the part that's actually about the driver. `f1pred.cli bias` prints it.
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

    if g["avg_predicted"].nunique() > 1:
        slope, intercept = np.polyfit(g["avg_predicted"], g["raw_bias"], 1)
    else:  # everyone at the same rank: no trend to remove
        slope, intercept = 0.0, float(g["raw_bias"].mean())
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


def measure(start_season: int = 2024, retrain_every: int = 3) -> dict:
    """Run the walk-forward and return the bias figures, ready to save."""
    entries = collect(features.load(), start_season, retrain_every)
    bias = driver_bias(entries)
    trend = trend_strength(entries)
    cols = ["races", "avg_predicted", "avg_actual", "raw_bias", "driver_specific_bias"]
    return {
        "start_season": start_season,
        "entries": int(entries["round"].size),
        "r": round(float(trend["r"]), 3),
        "variance_explained": round(float(trend["variance_explained"]), 3),
        "mae": round(float(entries["error"].abs().mean()), 2),
        "drivers": bias[cols].round(2).reset_index().to_dict("records"),
    }


def report(result: dict) -> str:
    table = pd.DataFrame(result["drivers"]).set_index("driver_id")
    return "\n".join(
        [
            f"Driver bias, {result['start_season']}+ walk-forward ({result['entries']} entries)",
            "",
            (
                f"Apparent bias correlates {result['r']:.3f} with predicted rank, so "
                f"{result['variance_explained']:.0%} of it is an artefact of comparing a fixed"
            ),
            "ranking against a mean pulled toward the middle by retirements.",
            "driver_specific_bias is what survives that correction.",
            "",
            "negative = rated better than they finish",
            "",
            table.to_string(),
            "",
            f"field mean absolute error: {result['mae']:.2f} positions",
        ]
    )
