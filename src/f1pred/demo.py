"""A small deterministic championship, for running the whole pipeline offline.

    make demo

Builds data/demo/demo.duckdb from a seeded generator, then runs what the live
system runs - validate, features, the walk-forward against the grid baseline,
a forecast of the next race and the forecast page - without touching an API.

The drivers and teams are invented. The numbers show that the machinery
works end to end, not how good the model is; reports/ holds the real ones.

The generator includes the awkward cases the real pipeline has to handle: a
mid-season driver replacement, a pit-lane start, a grid penalty, a driver
with no qualifying time, retirements, and sprint weekends.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

SEED = 7
SEASONS = (2022, 2023, 2024, 2025)
ROUNDS = 10
UNRUN = 2  # rounds at the end of the last season still to come
TEAMS = ("aurora", "boreal", "cobalt", "dune", "ember", "fjord", "garnet", "harbor", "iris", "juniper")
CIRCUITS = ("north", "south", "harbour", "desert", "forest", "street")
BASE_LAP_MS = 90_000
POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
SPRINT_POINTS = [8, 7, 6, 5, 4, 3, 2, 1]


def generate(seed: int = SEED, now: datetime | None = None) -> dict[str, pd.DataFrame]:
    """The raw tables of a made-up championship, shaped like the ingest writes them."""
    rng = np.random.default_rng(seed)
    now = (now or datetime.now(UTC)).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    n_races = len(SEASONS) * ROUNDS
    # Weekly races, the last run one four days ago, so the next is in the future.
    first = now - timedelta(days=4 + 7 * (n_races - UNRUN - 1))

    drivers = [f"{t}_{c}" for t in TEAMS for c in ("one", "two")]
    skill = dict(zip(drivers, rng.normal(0.0, 0.25, len(drivers))))
    skill["reserve_x"] = -0.1
    car = dict(zip(TEAMS, np.linspace(0.8, -0.8, len(TEAMS))))

    races, results, quali, sprint, standings, people = [], [], [], [], [], {}
    k = 0
    for season in SEASONS:
        totals: dict[str, float] = {}
        car = {t: v + rng.normal(0.0, 0.25) for t, v in car.items()}  # the order shuffles each winter
        for rnd in range(1, ROUNDS + 1):
            start = first + timedelta(days=7 * k)
            k += 1
            circuit = CIRCUITS[(rnd - 1) % len(CIRCUITS)]
            has_sprint = rnd in (3, 7)
            races.append(
                {
                    "season": season,
                    "round": rnd,
                    "race_name": f"{circuit.title()} Grand Prix",
                    "circuit_id": circuit,
                    "circuit_name": f"{circuit.title()} Circuit",
                    "locality": circuit.title(),
                    "country": "Nowhere",
                    "lat": 0.0,
                    "lon": 0.0,
                    "race_date": start.date(),
                    "race_time": start.strftime("%H:%M:%SZ"),
                    "race_start_utc": start,
                    "quali_start_utc": start - timedelta(days=1),
                    "sprint_start_utc": start - timedelta(hours=26) if has_sprint else None,
                }
            )
            if season == SEASONS[-1] and rnd > ROUNDS - UNRUN:
                continue  # scheduled, not run

            # The last season's second half has a stand-in for dune_two.
            field = [
                ("reserve_x" if (d == "dune_two" and season == SEASONS[-1] and 4 <= rnd <= 6) else d)
                for d in drivers
            ]
            team_of = {d: (d.rsplit("_", 1)[0] if d != "reserve_x" else "dune") for d in field}
            for d in field:
                people[d] = team_of[d]

            pace = np.array([car[team_of[d]] + skill[d] for d in field])
            q_noise = pace + rng.normal(0.0, 0.15, len(field))
            q_order = np.argsort(-q_noise)
            q_pos = np.empty(len(field), dtype=int)
            q_pos[q_order] = np.arange(1, len(field) + 1)
            lap = BASE_LAP_MS * (1 + 0.01 * (q_noise.max() - q_noise))
            no_time = (season, rnd) == (SEASONS[1], 5)
            for i, d in enumerate(field):
                # Top ten reach Q3, the next five Q2; each later session a little faster.
                t = int(lap[i])
                if q_pos[i] <= 10:
                    q1, q2, q3 = t + 300, t + 150, t
                elif q_pos[i] <= 15:
                    q1, q2, q3 = t + 300, t, None
                else:
                    q1, q2, q3 = t, None, None
                if no_time and i == 0:
                    q1 = q2 = q3 = None
                best = min([x for x in (q1, q2, q3) if x is not None], default=None)
                quali.append(
                    {
                        "season": season,
                        "round": rnd,
                        "driver_id": d,
                        "constructor_id": team_of[d],
                        "position": int(q_pos[i]),
                        "q1_ms": q1,
                        "q2_ms": q2,
                        "q3_ms": q3,
                        "best_ms": best,
                    }
                )

            grid = q_pos.astype(int).copy()
            if (season, rnd) == (SEASONS[2], 4):  # a five-place penalty for the pole-sitter
                pole = int(np.argmin(grid))
                grid[(grid > 1) & (grid <= 6)] -= 1
                grid[pole] = 6
            if (season, rnd) == (SEASONS[2], 8):  # a pit-lane start
                grid[int(np.argmin(grid))] = 0

            race_pace = (
                pace + rng.normal(0.0, 0.35, len(field)) - 0.02 * np.where(grid == 0, len(field), grid)
            )
            dnf = rng.random(len(field)) < 0.06
            race_pace = np.where(dnf, -100 + rng.random(len(field)), race_pace)
            order = np.argsort(-race_pace)
            laps_total = 50
            for pos, i in enumerate(order, start=1):
                d = field[i]
                pts = float(POINTS[pos - 1]) if pos <= 10 and not dnf[i] else 0.0
                totals[d] = totals.get(d, 0.0) + pts
                results.append(
                    {
                        "season": season,
                        "round": rnd,
                        "driver_id": d,
                        "constructor_id": team_of[d],
                        "grid": int(grid[i]),
                        "position": pos,
                        "classified": not dnf[i],
                        "position_text": str(pos) if not dnf[i] else "R",
                        "points": pts,
                        "laps": laps_total if not dnf[i] else int(rng.integers(1, laps_total - 5)),
                        "status": "Finished" if not dnf[i] else "Engine",
                        "finished": not dnf[i],
                        "dnf": bool(dnf[i]),
                        "millis": None,
                        "fastest_lap_rank": None,
                    }
                )
            if has_sprint:
                s_order = np.argsort(-(pace + rng.normal(0.0, 0.3, len(field))))
                for pos, i in enumerate(s_order, start=1):
                    d = field[i]
                    pts = float(SPRINT_POINTS[pos - 1]) if pos <= 8 else 0.0
                    totals[d] = totals.get(d, 0.0) + pts
                    sprint.append(
                        {
                            "season": season,
                            "round": rnd,
                            "driver_id": d,
                            "constructor_id": team_of[d],
                            "grid": int(q_pos[i]),
                            "position": pos,
                            "points": pts,
                            "status": "Finished",
                        }
                    )
            ranked = sorted(totals.items(), key=lambda kv: -kv[1])
            for place, (d, pts) in enumerate(ranked, start=1):
                standings.append(
                    {
                        "season": season,
                        "round": rnd,
                        "driver_id": d,
                        "constructor_id": people[d],
                        "position": place,
                        "points": pts,
                        "wins": 0,
                    }
                )

    driver_rows = [
        {
            "season": season,
            "driver_id": d,
            "code": d[:1].upper() + d.split("_")[0][1:3].upper() if d != "reserve_x" else "RSX",
            "permanent_number": i + 2,
            "given_name": d.split("_")[0].title(),
            "family_name": d.split("_")[1].title() if "_" in d else d.title(),
            "dob": None,
            "nationality": "Nowhere",
        }
        for season in SEASONS
        for i, d in enumerate([*drivers, "reserve_x"])
    ]
    return {
        "raw_races": pd.DataFrame(races),
        "raw_drivers": pd.DataFrame(driver_rows),
        "raw_results": pd.DataFrame(results),
        "raw_qualifying": pd.DataFrame(quali),
        "raw_sprint": pd.DataFrame(sprint),
        "raw_standings": pd.DataFrame(standings),
    }


KEYS = {
    "raw_races": ["season", "round"],
    "raw_drivers": ["season", "driver_id"],
    "raw_results": ["season", "round", "driver_id"],
    "raw_qualifying": ["season", "round", "driver_id"],
    "raw_sprint": ["season", "round", "driver_id"],
    "raw_standings": ["season", "round", "driver_id"],
}


def use(directory: Path) -> None:
    """Point every path the pipeline writes to at `directory`, away from the real data."""
    directory.mkdir(parents=True, exist_ok=True)
    config.DB_PATH = directory / "demo.duckdb"
    config.PREDICTIONS = directory / "predictions"
    config.REPORTS = directory / "reports"
    config.SEASON_NOW = directory / "season.json"
    for p in (config.PREDICTIONS, config.REPORTS):
        p.mkdir(parents=True, exist_ok=True)


def build_database(tables: dict[str, pd.DataFrame] | None = None) -> None:
    from . import store

    config.DB_PATH.unlink(missing_ok=True)
    store.init_db()
    tables = tables or generate()
    with store.connect() as con:
        for name, frame in tables.items():
            store.upsert(con, name, frame, KEYS[name])


def run(directory: Path | None = None, n_sims: int = 2000) -> dict:
    """The whole pipeline on the synthetic championship. Returns what it printed."""
    from . import backtest, features, predict, probability, report, validate

    directory = directory or (config.DATA / "demo")
    use(directory)
    build_database()

    checks = validate.run()
    print(f"validate: {len(checks.errors)} error(s), {len(checks.warnings)} warning(s)")
    if not checks.ok:
        print(checks.render(show_samples=False))
        raise SystemExit(1)

    df = features.build(include_upcoming=True)
    features.save(df)
    done = df[df["position"].notna()]
    print(
        f"features: {len(df)} rows, {done['race_seq'].nunique()} completed races, {len(features.RACE_FEATURES)} race features"
    )

    res = backtest.walk_forward(
        df, SEASONS[-1], retrain_every=2, n_seeds=2, settings=backtest.Settings(n_sims=n_sims), warmup=8
    )
    summary = res.summary()[["ndcg5", "winner_hit", "win_logloss", "podium_brier"]]
    print(
        f"\nwalk-forward over {int(res.summary().loc['model', 'n_races'])} races (trained only on earlier races):"
    )
    print(summary.round(3).to_string())

    pred = predict.run(n_sims=n_sims)
    matrix = np.array(pred.position_matrix)
    problems = probability.check_distribution(matrix, atol=1e-3)
    print(f"\nforecast: {pred.race_name} ({pred.meta['stage']}), field from {pred.meta['entry_source']}")
    for i, d in enumerate(pred.race_board[:5], 1):
        print(
            f"  {i}. {d['name']:<14} win {d['p_win']:6.1%}  podium {d['p_podium']:6.1%}  "
            f"top 10 {d['p_top10']:6.1%}  expected P{d['exp_position']:.1f}"
        )
    print(f"distribution check: {'coherent' if not problems else problems}")
    page = report.write(json.loads(json.dumps(asdict(pred), default=str)))
    print(f"page: {page}")
    return {"summary": summary, "prediction": pred, "problems": problems}
