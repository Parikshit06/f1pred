"""Calibrate the width of the season projection's 10th-90th band.

The band should contain the real final total eight times in ten. Race luck
alone averages out over a season, so without an extra term it was far too
narrow. Most of the gap is the model being wrong about a car today, which
carries into every remaining race.

Two terms, fitted in order:

    SEASON_PACE_UNCERTAINTY    one offset per team. Graded on constructors'
                               final points.
    SEASON_DRIVER_UNCERTAINTY  one offset per driver on top. Graded on the
                               final points gap between teammates, which the
                               team term can't move.

Each is swept on one set of seasons and graded on another.

    f1pred.cli calibrate-spread
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import championship, config, model, simulate
from .store import connect

log = logging.getLogger(__name__)

# Same checkpoints the title backtest uses, so the two grade the same moments.
CHECKPOINTS = (0.35, 0.55, 0.75, 0.90)
FIT_SEASONS = (2019, 2020, 2021, 2022)
GRADE_SEASONS = (2023, 2024, 2025)
CANDIDATES = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
TARGET_COVERAGE = 0.80
MIN_TRAIN_RACES = 30


def final_points(season: int) -> tuple[dict[str, float], dict[str, float]]:
    """Final points by driver and by constructor.

    Constructors' points are summed from the results as awarded - a driver who
    changed team mid-season scored for both - not from the drivers' totals.
    """
    with connect(read_only=True) as con:
        rows = con.execute(
            """
            SELECT driver_id, points FROM raw_standings
            WHERE season = ? AND round = (SELECT max(round) FROM raw_standings WHERE season = ?)
              AND driver_id IS NOT NULL
            """,
            [season, season],
        ).fetchall()
    drivers = {d: float(p) for d, p in rows}
    return drivers, championship.constructor_points(season, 10**6)


def checkpoints(df: pd.DataFrame, seasons: tuple[int, ...]) -> list[dict]:
    """Fit the ranker once per checkpoint, so every candidate reuses the same
    scores and the comparison between them is paired."""
    out: list[dict] = []
    for season in seasons:
        drivers, teams = final_points(season)
        rounds = sorted(df[df.season == season]["round"].unique())
        if not teams or len(rounds) < 6:
            continue
        for frac in CHECKPOINTS:
            after = round(len(rounds) * frac)
            nxt = df[(df.season == season) & (df["round"] == after + 1)]
            if after < 3 or after >= len(rounds) or nxt.empty:
                continue
            train = df[df["race_seq"] < int(nxt["race_seq"].iloc[0])]
            if train["race_seq"].nunique() < MIN_TRAIN_RACES:
                continue
            race = nxt.copy()
            race["score"] = championship.season_strength(model.train_race(train), race, train)
            out.append(
                {
                    "season": season,
                    "after": after,
                    "elapsed": after / len(rounds),
                    "race": race,
                    "actual": teams,
                    "drivers": drivers,
                }
            )
            log.info("fitted %d after r%d", season, after)
    return out


def coverage(points: list[dict], team: float, driver: float = 0.0, n_sims: int = 6000) -> pd.DataFrame:
    """One row per team and per teammate pair at each checkpoint: did the band
    contain what really happened?"""
    saved = config.SEASON_PACE_UNCERTAINTY, config.SEASON_DRIVER_UNCERTAINTY
    config.SEASON_PACE_UNCERTAINTY, config.SEASON_DRIVER_UNCERTAINTY = team, driver
    rows: list[dict] = []
    try:
        for cp in points:
            race = cp["race"]
            out = championship.project(
                race["driver_id"].tolist(),
                race["constructor_id"].tolist(),
                race["score"].to_numpy(),
                simulate.dnf_probability(race),
                cp["season"],
                cp["after"],
                n_sims=n_sims,
            )
            if not out:
                continue
            base = {"season": cp["season"], "elapsed": cp["elapsed"]}
            for t in out["constructors"].itertuples():
                if t.team in cp["actual"]:
                    truth = cp["actual"][t.team]
                    rows.append(
                        {
                            **base,
                            "kind": "team",
                            "inside": int(t.low <= truth <= t.high),
                            "width": t.high - t.low,
                        }
                    )
            drivers = cp.get("drivers", {})
            for p in out.get("teammates", pd.DataFrame()).itertuples():
                if p.first in drivers and p.second in drivers:
                    truth = drivers[p.first] - drivers[p.second]
                    rows.append(
                        {
                            **base,
                            "kind": "gap",
                            "inside": int(p.low <= truth <= p.high),
                            "width": p.high - p.low,
                        }
                    )
    finally:
        config.SEASON_PACE_UNCERTAINTY, config.SEASON_DRIVER_UNCERTAINTY = saved
    return pd.DataFrame(rows, columns=["season", "elapsed", "kind", "inside", "width"])


def _sweep(frames: dict[float, pd.DataFrame], kind: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "spread": c,
                "coverage": float(f.loc[f.kind == kind, "inside"].mean()),
                "width": float(f.loc[f.kind == kind, "width"].mean()),
            }
            for c, f in frames.items()
        ]
    )


def _closest(sweep: pd.DataFrame) -> float:
    return float(sweep.loc[(sweep["coverage"] - TARGET_COVERAGE).abs().idxmin(), "spread"])


def run(df: pd.DataFrame, n_sims: int = 6000) -> dict:
    """Sweep the team term, then the driver term with the team term fixed, on
    the fit window; grade both on the held-out one."""
    fit_points = checkpoints(df, FIT_SEASONS)
    grade_points = checkpoints(df, GRADE_SEASONS)
    if not fit_points or not grade_points:
        return {}

    sweep = _sweep({c: coverage(fit_points, c, 0.0, n_sims) for c in CANDIDATES}, "team")
    best = _closest(sweep)
    driver_sweep = _sweep({c: coverage(fit_points, best, c, n_sims) for c in CANDIDATES}, "gap")
    best_driver = _closest(driver_sweep)

    graded = {
        "before": coverage(grade_points, 0.0, 0.0, n_sims),
        "after": coverage(grade_points, best, best_driver, n_sims),
    }
    return {
        "sweep": sweep,
        "best": best,
        "driver_sweep": driver_sweep,
        "best_driver": best_driver,
        "graded": graded,
        "n_checkpoints": len(grade_points),
    }


def held_out(result: dict) -> dict:
    """The graded figures, as saved to reports/spread_calibration.json."""
    out = {}
    for label, frame in result["graded"].items():
        team, gap = frame[frame.kind == "team"], frame[frame.kind == "gap"]
        out[label] = {
            "coverage": float(team["inside"].mean()),
            "width": float(team["width"].mean()),
            "n": len(team),
            "teammate_coverage": float(gap["inside"].mean()),
            "teammate_n": len(gap),
        }
    return out


def report(result: dict) -> str:
    if not result:
        return "not enough completed seasons to calibrate."

    out = [f"Fit on {FIT_SEASONS[0]}-{FIT_SEASONS[-1]}"]
    for title, sweep, best in (
        ("team term, graded on constructors", result["sweep"], result["best"]),
        ("driver term, graded on teammate gaps", result.get("driver_sweep"), result.get("best_driver")),
    ):
        if sweep is None:
            continue
        out += ["", title, f"{'spread (positions)':<20}{'coverage':>10}{'band width':>12}"]
        for _, r in sweep.iterrows():
            out.append(f"{r['spread']:<20.1f}{r['coverage']:>9.1%}{r['width']:>12.1f}")
        out.append(f"closest to {TARGET_COVERAGE:.0%}: {best:.1f}")

    out += [
        "",
        (
            f"Held out: {GRADE_SEASONS[0]}-{GRADE_SEASONS[-1]}, "
            f"{result['n_checkpoints']} checkpoints, never seen by the sweep"
        ),
        f"{'':<20}{'teams':>10}{'width':>10}{'teammates':>12}",
    ]
    for label, frame in (
        ("race luck only", result["graded"]["before"]),
        ("calibrated", result["graded"]["after"]),
    ):
        team = frame[frame.kind == "team"] if "kind" in frame else frame
        gap = frame[frame.kind == "gap"] if "kind" in frame else frame.iloc[0:0]
        out.append(
            f"{label:<20}{team['inside'].mean():>9.1%}{team['width'].mean():>10.1f}"
            f"{gap['inside'].mean() if len(gap) else np.nan:>11.1%}"
        )
    out.append(f"{'target':<20}{TARGET_COVERAGE:>9.1%}")
    return "\n".join(out)
