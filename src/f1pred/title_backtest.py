"""Grade the title projection against completed seasons.

For each season from 2019, stop at a few checkpoints, project the title from
a model trained only on earlier races, and record what it said against who
won. A small sample by nature - a handful of checkpoints a season - and 2018
can't be graded because its checkpoints have too little history behind them.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import championship, model, simulate
from .store import connect

log = logging.getLogger(__name__)

# Quarter, half, three-quarter and late season. Enough to cover the range from
# "wide open" to "all but settled" without refitting the model twenty times a
# season.
CHECKPOINT_FRACTIONS = (0.35, 0.55, 0.75, 0.90)


def actual_champion(season: int) -> str | None:
    with connect(read_only=True) as con:
        row = con.execute(
            """
            SELECT driver_id FROM raw_standings
            WHERE season = ? AND round = (SELECT max(round) FROM raw_standings WHERE season = ?)
              AND driver_id IS NOT NULL
            ORDER BY points DESC LIMIT 1
            """,
            [season, season],
        ).fetchone()
    return row[0] if row else None


def run(
    df: pd.DataFrame,
    seasons: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024, 2025),
    n_sims: int = 4000,
) -> pd.DataFrame:
    """One row per checkpoint: what the projection said, and what happened."""
    rows: list[dict] = []
    df = df.sort_values(["race_seq", "driver_id"]).reset_index(drop=True)

    for season in seasons:
        champion = actual_champion(season)
        if not champion:
            continue
        rounds = sorted(df[df.season == season]["round"].unique())
        if len(rounds) < 6:
            continue

        for frac in CHECKPOINT_FRACTIONS:
            after = round(len(rounds) * frac)
            if after < 3 or after >= len(rounds):
                continue
            nxt = df[(df.season == season) & (df["round"] == after + 1)]
            if nxt.empty:
                continue

            seq = int(nxt["race_seq"].iloc[0])
            train = df[df["race_seq"] < seq]
            if train["race_seq"].nunique() < 30:
                continue

            ranker = model.train_race(train)
            race = nxt.copy()
            # A typical weekend, not the next race's: that race's qualifying had
            # not happened at the checkpoint, and its track is one of many left.
            race["score"] = ranker.score(championship.typical_weekend(race, train))

            out = championship.project(
                race["driver_id"].tolist(),
                race["constructor_id"].tolist(),
                race["score"].to_numpy(),
                simulate.dnf_probability(race),
                season,
                after,
                n_sims=n_sims,
            )
            if not out:
                continue

            drivers = out["drivers"].set_index("driver_id")
            top = out["drivers"].iloc[0]
            said = float(drivers.loc[champion, "p_title"]) if champion in drivers.index else 0.0

            rows.append(
                {
                    "season": season,
                    "after_round": after,
                    "races_left": out["n_races"],
                    "favourite": top["driver_id"],
                    "p_favourite": float(top["p_title"]),
                    "champion": champion,
                    "favourite_was_right": int(top["driver_id"] == champion),
                    "p_on_actual_champion": said,
                }
            )
            log.info(
                "%d after r%d: favourite %s at %.1f%% -> champion %s",
                season,
                after,
                top["driver_id"],
                top["p_title"] * 100,
                champion,
            )

    return pd.DataFrame(rows)


def calibration(frame: pd.DataFrame) -> pd.DataFrame:
    """How often the favourite actually won, bucketed by what was claimed."""
    if frame.empty:
        return pd.DataFrame()
    edges = [0.0, 0.5, 0.8, 0.95, 1.0001]
    labels = ["under 50%", "50-80%", "80-95%", "over 95%"]
    f = frame.assign(bucket=pd.cut(frame["p_favourite"], edges, labels=labels, include_lowest=True))
    g = (
        f.groupby("bucket", observed=True)
        .agg(
            claimed=("p_favourite", "mean"),
            happened=("favourite_was_right", "mean"),
            n=("favourite_was_right", "size"),
        )
        .reset_index()
    )
    lo, hi = zip(*[_wilson(int(r.happened * r.n), int(r.n)) for r in g.itertuples()])
    g["ci_low"], g["ci_high"] = lo, hi
    return g.round(3)


def _wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = hits / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)


def report(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No completed seasons to grade the title projection against."
    brier = float(((frame["p_favourite"] - frame["favourite_was_right"]) ** 2).mean())
    lines = [
        f"{len(frame)} checkpoints across {frame['season'].nunique()} completed seasons.",
        f"The favourite went on to win the title in {frame['favourite_was_right'].mean() * 100:.0f}% of them.",
        f"Brier score on the title call: {brier:.3f}.",
        "",
        calibration(frame).to_string(index=False),
        "",
        frame[
            [
                "season",
                "after_round",
                "races_left",
                "favourite",
                "p_favourite",
                "champion",
                "favourite_was_right",
            ]
        ].to_string(index=False),
    ]
    return "\n".join(lines)
