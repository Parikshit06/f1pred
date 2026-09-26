"""Calibrate the width of the season projection's 10th-90th band.

The band should contain the real final total eight times in ten. Race luck
alone averages out over a season, so without an extra term the band was far
too narrow. Development is real but small (about 1.4 finishing positions a
season); most of the gap is the model being wrong about a car today, which
carries into every remaining race.

config.SEASON_PACE_UNCERTAINTY covers all of it and is chosen by coverage:
swept on one set of seasons, graded on another.

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


def final_constructor_points(season: int) -> dict[str, float]:
    """Where each team actually finished the season, in points.

    Constructors' points are the sum of their drivers', taken at the final
    round - so a mid-season driver switch counts toward the team they ended
    with, which is how the real table works.
    """
    with connect(read_only=True) as con:
        rows = con.execute(
            """
            SELECT constructor_id, sum(points) FROM raw_standings
            WHERE season = ? AND round = (SELECT max(round) FROM raw_standings WHERE season = ?)
              AND constructor_id IS NOT NULL
            GROUP BY constructor_id
            """,
            [season, season],
        ).fetchall()
    return {r[0]: float(r[1]) for r in rows}


def checkpoints(df: pd.DataFrame, seasons: tuple[int, ...]) -> list[dict]:
    """Fit the ranker once per checkpoint.

    Every candidate spread then reuses the same scores, so the only thing that
    differs between arms is the simulator - and the comparison is paired rather
    than two independent runs that happen to disagree.
    """
    out: list[dict] = []
    for season in seasons:
        actual = final_constructor_points(season)
        rounds = sorted(df[df.season == season]["round"].unique())
        if not actual or len(rounds) < 6:
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
            race["score"] = model.train_race(train).score(championship.typical_weekend(race, train))
            out.append(
                {
                    "season": season,
                    "after": after,
                    "elapsed": after / len(rounds),
                    "race": race,
                    "actual": actual,
                }
            )
            log.info("fitted %d after r%d", season, after)
    return out


def coverage(points: list[dict], spread: float, n_sims: int = 6000) -> pd.DataFrame:
    """One row per team per checkpoint: did the band contain the real total?"""
    original = config.SEASON_PACE_UNCERTAINTY
    config.SEASON_PACE_UNCERTAINTY = spread
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
            for _, team in out["constructors"].iterrows():
                if team["team"] not in cp["actual"]:
                    continue
                truth = cp["actual"][team["team"]]
                rows.append(
                    {
                        "season": cp["season"],
                        "elapsed": cp["elapsed"],
                        "team": team["team"],
                        "truth": truth,
                        "inside": int(team["low"] <= truth <= team["high"]),
                        "width": team["high"] - team["low"],
                    }
                )
    finally:
        config.SEASON_PACE_UNCERTAINTY = original
    return pd.DataFrame(rows)


def run(df: pd.DataFrame, n_sims: int = 6000) -> dict:
    """Sweep on the fit window, grade the winner on the held-out one."""
    fit_points = checkpoints(df, FIT_SEASONS)
    grade_points = checkpoints(df, GRADE_SEASONS)
    if not fit_points or not grade_points:
        return {}

    sweep = []
    for candidate in CANDIDATES:
        frame = coverage(fit_points, candidate, n_sims)
        sweep.append(
            {
                "spread": candidate,
                "coverage": float(frame["inside"].mean()),
                "width": float(frame["width"].mean()),
            }
        )
    sweep = pd.DataFrame(sweep)
    best = float(sweep.loc[(sweep["coverage"] - TARGET_COVERAGE).abs().idxmin(), "spread"])

    graded = {
        "before": coverage(grade_points, 0.0, n_sims),
        "after": coverage(grade_points, best, n_sims),
    }
    return {"sweep": sweep, "best": best, "graded": graded, "n_checkpoints": len(grade_points)}


def report(result: dict) -> str:
    """The table this prints is the one the method page quotes."""
    if not result:
        return "not enough completed seasons to calibrate."

    out = [
        f"Calibration sweep on {FIT_SEASONS[0]}-{FIT_SEASONS[-1]}",
        f"{'spread (positions)':<20}{'coverage':>10}{'band width':>12}",
    ]
    for _, r in result["sweep"].iterrows():
        out.append(f"{r['spread']:<20.1f}{r['coverage']:>9.1%}{r['width']:>12.1f}")
    out.append(f"closest to {TARGET_COVERAGE:.0%}: {result['best']:.1f}")

    out += [
        "",
        (
            f"Held out: {GRADE_SEASONS[0]}-{GRADE_SEASONS[-1]}, "
            f"{result['n_checkpoints']} checkpoints, never seen by the sweep"
        ),
        f"{'':<20}{'coverage':>10}{'band width':>12}",
    ]
    for label, frame in (
        ("fixed pace", result["graded"]["before"]),
        (f"calibrated ({result['best']:.1f})", result["graded"]["after"]),
    ):
        out.append(f"{label:<20}{frame['inside'].mean():>9.1%}{frame['width'].mean():>12.1f}")
    out.append(f"{'target':<20}{TARGET_COVERAGE:>9.1%}")

    # Coverage by how far into the season the call was made. The old band was
    # worst early, which is the whole reason this exists, so the breakdown is
    # part of the result rather than a footnote.
    out += ["", "Coverage by how much of the season had been run:"]
    cols = sorted(result["graded"]["after"]["elapsed"].round(2).unique())
    out.append(f"{'':<20}" + "".join(f"{c:>8.0%}" for c in cols))
    for label, frame in (
        ("fixed pace", result["graded"]["before"]),
        ("calibrated", result["graded"]["after"]),
    ):
        by = frame.groupby(frame["elapsed"].round(2))["inside"].mean()
        out.append(f"{label:<20}" + "".join(f"{by.get(c, np.nan):>8.0%}" for c in cols))
    return "\n".join(out)
