"""Season projection: run the rest of the calendar many times from current form.

Each simulated race uses the same noise model as simulate.simulate - pace
variance, safety cars, per-driver retirement hazard - so title odds stay
consistent with the published race probabilities.

Assumes current form holds; it can't see an upgrade coming. Sprint points
are left out because future sprint rounds aren't in the data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .store import connect

log = logging.getLogger(__name__)

# 2010-present scoring, top ten.
POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)


def standings_after(season: int, rnd: int) -> pd.DataFrame:
    """Driver standings as they stand, including any sprint points scored."""
    with connect(read_only=True) as con:
        return con.execute(
            """
            SELECT driver_id, constructor_id, points, position
            FROM raw_standings
            WHERE season = ? AND round = ? AND driver_id IS NOT NULL
            ORDER BY position
            """,
            [season, rnd],
        ).fetchdf()


def points_history(season: int) -> pd.DataFrame:
    """Cumulative points per driver after every round run so far."""
    with connect(read_only=True) as con:
        return con.execute(
            """
            SELECT round, driver_id, points
            FROM raw_standings
            WHERE season = ? AND driver_id IS NOT NULL
            ORDER BY round
            """,
            [season],
        ).fetchdf()


def remaining_rounds(season: int, after_round: int) -> int:
    with connect(read_only=True) as con:
        row = con.execute(
            "SELECT count(*) FROM raw_races WHERE season = ? AND round > ?",
            [season, after_round],
        ).fetchone()
    return int(row[0]) if row else 0


def simulate_seasons(
    scores: np.ndarray,
    dnf_prob: np.ndarray,
    base_points: np.ndarray,
    n_races: int,
    team_index: np.ndarray | None = None,
    n_sims: int = 10_000,
    safety_car_prob: float = 0.35,
    season_fraction_left: float = 0.0,
    seed: int = config.RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Final totals per simulation, plus per-round mean, 10th and 90th percentile
    points and expected wins (what the progression chart draws).

    Same race noise as simulate.simulate, minus the grid term. On top, each
    simulated season draws one pace offset per team and holds it all year:
    race luck averages out over a dozen races, but error in today's read of a
    car doesn't. Sized by config.SEASON_PACE_UNCERTAINTY. It models how far a
    team could drift, not which team will.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    score_sd = float(np.std(scores)) or 1.0

    # One offset per team, shared by teammates, plus a smaller one per driver so
    # teammates can still drift apart. Both scale with the share of the season
    # still to run.
    left = float(np.clip(season_fraction_left, 0.0, 1.0))
    to_score = left * score_sd / config.POSITIONS_PER_SCORE_SD
    team_sd = config.SEASON_PACE_UNCERTAINTY * to_score
    driver_sd = config.SEASON_DRIVER_UNCERTAINTY * to_score
    drift = np.zeros((1, n))
    if team_sd > 0:
        if team_index is None:
            drift = drift + rng.normal(0.0, team_sd, size=(n_sims, n))
        else:
            drift = drift + rng.normal(0.0, team_sd, size=(n_sims, int(team_index.max()) + 1))[:, team_index]
    if driver_sd > 0:
        drift = drift + rng.normal(0.0, driver_sd, size=(n_sims, n))

    totals = np.tile(np.asarray(base_points, dtype=float), (n_sims, 1))
    track = np.zeros((n_races, n))
    lo_track = np.zeros((n_races, n))
    hi_track = np.zeros((n_races, n))
    wins = np.zeros(n)
    rows = np.arange(n_sims)[:, None]
    take = min(len(POINTS), n)

    for r in range(n_races):
        chaos = np.where(rng.random((n_sims, 1)) < safety_car_prob, 1.9, 1.0)
        pace = scores + drift + rng.normal(0.0, 1.0, size=(n_sims, n)) * (0.55 * score_sd * chaos)

        retired = rng.random((n_sims, n)) < dnf_prob
        pace = np.where(retired, -np.inf, pace)

        order = np.argsort(-pace, axis=1)
        awarded = np.zeros((n_sims, n))
        awarded[rows, order[:, :take]] = POINTS[:take]
        # A car that stopped scores nothing, even in a thin field.
        awarded = np.where(retired, 0.0, awarded)

        np.add.at(wins, order[:, 0], 1)

        totals += awarded
        track[r] = totals.mean(axis=0)
        lo_track[r] = np.percentile(totals, 10, axis=0)
        hi_track[r] = np.percentile(totals, 90, axis=0)

    # Wins are counted per simulated race, so this is expected wins over the
    # remaining calendar.
    return totals, track, lo_track, hi_track, wins / n_sims


def allocate_odds(
    probs: np.ndarray,
    clinched: np.ndarray | None = None,
    rank_by: np.ndarray | None = None,
) -> np.ndarray:
    """Round title odds to 0.1% so the column still sums to 100.

    Largest-remainder rounding, and a leader who hasn't clinched is never shown
    at 100% - the simulation can't resolve past 1 in n_sims, and the real
    uncertainty is whether form holds.
    """
    probs = np.asarray(probs, dtype=float)
    total = probs.sum()
    if total <= 0:
        return np.zeros_like(probs)

    # Largest remainder, in tenths of a percent.
    exact = probs / total * 1000.0
    tenths = np.floor(exact).astype(int)
    short = 1000 - tenths.sum()
    if short > 0:
        for i in np.argsort(-(exact - np.floor(exact)))[:short]:
            tenths[i] += 1

    leader = int(np.argmax(tenths))
    is_clinched = bool(clinched[leader]) if clinched is not None else False
    if tenths[leader] >= 1000 and not is_clinched:
        # Everyone else is on a simulated zero, so ordering by probability would
        # pick an arbitrary rival. Order by points when given them.
        order = np.argsort(-(rank_by if rank_by is not None else probs))
        rival = int(order[1]) if len(probs) > 1 else leader
        if rival != leader:
            tenths[leader] -= 1
            tenths[rival] += 1

    return tenths / 1000.0


def clinch_round(leader_points: float, rival_points: float, races_left: int, win: float = 25.0) -> int | None:
    """Races from now until the leader could mathematically seal the title.

    Leader wins every race, rival scores nothing: settled once
    L + 25k > R + 25(races_left - k). None if not before the finale. Ignores
    sprints, so it's a lower bound.
    """
    if races_left <= 0:
        return None
    gap = leader_points - rival_points
    need = (win * races_left - gap) / (2 * win)
    k = int(np.ceil(need + 1e-9))
    k = max(k, 1)
    return k if k <= races_left else None


def project(
    driver_ids: list[str],
    teams: list[str],
    scores: np.ndarray,
    dnf_prob: np.ndarray,
    season: int,
    after_round: int,
    n_sims: int = 10_000,
    safety_car_prob: float = 0.35,
) -> dict:
    """Title projection for drivers and constructors."""
    standings = standings_after(season, after_round)
    if standings.empty:
        return {}

    have = dict(zip(standings["driver_id"], standings["points"]))
    base = np.array([have.get(d, 0.0) for d in driver_ids], dtype=float)
    n_races = remaining_rounds(season, after_round)
    if n_races == 0:
        return {}

    names = {t: i for i, t in enumerate(dict.fromkeys(teams))}
    team_index = np.array([names[t] for t in teams])
    totals, track, lo_track, hi_track, wins = simulate_seasons(
        scores,
        dnf_prob,
        base,
        n_races,
        n_sims=n_sims,
        safety_car_prob=safety_car_prob,
        team_index=team_index,
        season_fraction_left=n_races / max(after_round + n_races, 1),
    )

    # Ties for the title are vanishingly rare and are broken on countback in
    # reality; argmax picks one, which is close enough at this precision.
    champion = np.argmax(totals, axis=1)
    raw_title = np.bincount(champion, minlength=len(driver_ids)) / n_sims
    rank = (-totals).argsort(axis=1).argsort(axis=1) + 1

    # Mathematically alive: could still reach the leader by winning every
    # remaining race while the leader scores nothing.
    max_reachable = base + POINTS[0] * n_races
    alive = max_reachable >= base.max()
    clinched = alive.sum() == 1
    title = allocate_odds(raw_title, alive & clinched, totals.mean(axis=0))

    drivers = pd.DataFrame(
        {
            "driver_id": driver_ids,
            "team": teams,
            "now": base,
            "projected": totals.mean(axis=0),
            "low": np.percentile(totals, 10, axis=0),
            "high": np.percentile(totals, 90, axis=0),
            "exp_wins": wins,
            "p_title": title,
            "alive": alive,
            "p_top3": (rank <= 3).mean(axis=0),
        }
    ).sort_values("projected", ascending=False)

    # Constructors: both cars, summed inside each simulated season, so a team's
    # title odds account for the correlation between its two drivers.
    team_names = sorted(set(teams))
    idx = {t: [i for i, x in enumerate(teams) if x == t] for t in team_names}
    team_totals = np.stack([totals[:, idx[t]].sum(axis=1) for t in team_names], axis=1)
    team_base = np.array([base[idx[t]].sum() for t in team_names])
    t_champ = np.argmax(team_totals, axis=1)
    t_raw = np.bincount(t_champ, minlength=len(team_names)) / n_sims
    t_max = team_base + 2 * POINTS[0] * n_races  # both cars, every race
    t_alive = t_max >= team_base.max()
    constructors = pd.DataFrame(
        {
            "team": team_names,
            "now": team_base,
            "projected": team_totals.mean(axis=0),
            "low": np.percentile(team_totals, 10, axis=0),
            "high": np.percentile(team_totals, 90, axis=0),
            "p_title": allocate_odds(t_raw, t_alive & (t_alive.sum() == 1), team_totals.mean(axis=0)),
            "alive": t_alive,
        }
    ).sort_values("projected", ascending=False)

    # Points gap between teammates, first-listed minus second. Graded by
    # calibrate-spread; the team offset cancels here, so only the per-driver
    # term can widen it.
    pairs = []
    for t in team_names:
        if len(idx[t]) == 2:
            a, b = idx[t]
            gap = totals[:, a] - totals[:, b]
            pairs.append(
                {
                    "team": t,
                    "first": driver_ids[a],
                    "second": driver_ids[b],
                    "low": float(np.percentile(gap, 10)),
                    "high": float(np.percentile(gap, 90)),
                }
            )

    return {
        "drivers": drivers,
        "constructors": constructors,
        "teammates": pd.DataFrame(pairs, columns=["team", "first", "second", "low", "high"]),
        "projection": pd.DataFrame(
            track, columns=driver_ids, index=range(after_round + 1, after_round + 1 + n_races)
        ),
        "projection_low": pd.DataFrame(
            lo_track, columns=driver_ids, index=range(after_round + 1, after_round + 1 + n_races)
        ),
        "projection_high": pd.DataFrame(
            hi_track, columns=driver_ids, index=range(after_round + 1, after_round + 1 + n_races)
        ),
        "n_races": n_races,
        "n_sims": n_sims,
        "after_round": after_round,
        "points_available": int(POINTS[0] * n_races),
        "lead": float(np.sort(base)[-1] - np.sort(base)[-2]) if len(base) > 1 else 0.0,
        "clinch_in": clinch_round(float(np.sort(base)[-1]), float(np.sort(base)[-2]), n_races)
        if len(base) > 1
        else None,
        "clinched": bool(clinched),
    }


# ---------------------------------------------------------------------------
# What a driver is worth on a weekend that is not this one
# ---------------------------------------------------------------------------
# Ten of the race model's features describe the weekend rather than the driver:
# where they start, and the track they start at. Together they carry about half
# the model - qualifying position alone is a third of it.
WEEKEND_GRID = ("grid", "quali_position", "quali_gap_to_pole_pct", "quali_gap_to_teammate_pct")
WEEKEND_CIRCUIT = (
    "drv_circuit_avg_finish",
    "team_circuit_avg_finish",
    "circuit_overtaking_score",
    "circuit_dnf_rate",
    "drv_circuit_starts",
    "circuit_pole_win_rate",
)


def typical_weekend(race: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """The same field on an ordinary weekend at an ordinary track.

    The rest of the season isn't at this circuit or from this grid, so the
    projection shouldn't be scored on either. Scoring it with the next race's
    features let one qualifying session reshape nine races.

    Weekend-specific features are replaced from races before this one: grid and
    qualifying with the driver's recent averages, qualifying gaps with their
    recent mean, track record with recent form overall, circuit traits with the
    median. Recent averages keep each driver in a situation the model has seen,
    unlike forcing everyone onto one grid slot.
    """
    t = race.copy()
    cols = [c for c in WEEKEND_GRID + WEEKEND_CIRCUIT if c in history.columns]
    med = history[cols].median(numeric_only=True)

    def col(name: str) -> pd.Series:
        return t[name] if name in t.columns else pd.Series(np.nan, index=t.index)

    # .fillna with a Series aligns on index; every Series here is built from
    # t's own index, so they line up by construction.
    t["quali_position"] = (
        col("drv_avg_quali_5").fillna(col("drv_avg_grid_5")).fillna(med.get("quali_position"))
    )
    t["grid"] = col("drv_avg_grid_5").fillna(t["quali_position"])

    recent = history.sort_values("race_seq").groupby("driver_id").tail(5)
    for c in ("quali_gap_to_pole_pct", "quali_gap_to_teammate_pct"):
        if c in history.columns:
            per_driver = recent.groupby("driver_id")[c].mean()
            t[c] = t["driver_id"].map(per_driver).fillna(med.get(c))

    if "drv_circuit_avg_finish" in t.columns:
        t["drv_circuit_avg_finish"] = col("drv_avg_finish_5").fillna(med.get("drv_circuit_avg_finish"))
    if "team_circuit_avg_finish" in t.columns:
        t["team_circuit_avg_finish"] = col("team_avg_finish_5").fillna(med.get("team_circuit_avg_finish"))
    for c in ("circuit_overtaking_score", "circuit_dnf_rate", "circuit_pole_win_rate", "drv_circuit_starts"):
        if c in t.columns:
            t[c] = med.get(c)
    return t
