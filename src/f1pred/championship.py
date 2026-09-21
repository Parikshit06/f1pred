"""Where the championship ends up, given how the cars are going now.

A race forecast answers one Sunday. A title projection answers the question
people actually argue about, and it is a different kind of estimate: nine races
of compounding luck, where a single retirement is worth more than a tenth of a
second of pace.

Method: take the race model's strength for each driver, then run the remaining
calendar ten thousand times. Each race uses exactly the generative process the
single-race simulator uses - the same pace variance between Sundays, the same
safety-car draw, the same per-driver retirement hazard - so a driver's title
odds are consistent with the win probability published for this weekend. A
first version drew Gumbel noise instead, which is sharper, and it returned a
title probability of 1.000. A public page should not print certainty nine races
out, and the number was an artefact of the noise model rather than a finding.

The assumption, stated plainly because it is the one that matters: current form
holds. The projection does not re-forecast each weekend, so it cannot see an
upgrade that lands in three races' time or a car that falls away. It is a
snapshot of where this season goes if nothing changes, not a prophecy.

Sprint points are not projected. The remaining sprint calendar is not in the
data, and inventing one would put fabricated points on a public page.
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
    """Returns (final totals per sim, and then per round: the mean cumulative
    points, the 10th percentile, the 90th, and expected wins per driver).

    The noise model is lifted from simulate.simulate so the two agree: pace
    varies between Sundays by 0.55 of the spread in model score, a safety car
    nearly doubles that, and cars retire at their own hazard rate. The grid
    term is absent because nine races out there is no grid to know - in the
    race simulator an unknown grid is a constant that drops out of the
    ordering anyway.

    One term exists here that has no counterpart in the single-race simulator,
    because over one afternoon it has nowhere to act: each simulated season
    gives every team a pace offset, drawn once and held for the rest of the
    year. It covers the upgrade that works, the one that does not, and - mostly
    - the plain fact that the ranker's read of the field today is not the
    field's true pace. Race-to-race noise averages out over a dozen races; that
    error does not, which is why it dominates and why leaving it out made the
    published band far too narrow. See config.SEASON_PACE_UNCERTAINTY, which is
    calibrated on whether the band actually covers.

    It is deliberately not a forecast of WHICH team improves. Nothing in the
    data supports that, and a projection that guessed would be worse than one
    that admits the spread.

    The second return value is what the progression chart draws; computing it
    here rather than extrapolating a per-race average keeps the line and the
    final standing consistent with each other.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    score_sd = float(np.std(scores)) or 1.0

    # Pace belongs to the car, so both cars in a garage move together; without a
    # team map it falls back to per-driver, which is the same magnitude applied
    # independently. Scaled by how much of the season is left: with two races to
    # go there is neither time to develop nor much left for the model to be
    # wrong about.
    drift_sd = (
        config.SEASON_PACE_UNCERTAINTY
        * float(np.clip(season_fraction_left, 0.0, 1.0))
        / config.POSITIONS_PER_SCORE_SD
    ) * score_sd
    if drift_sd > 0:
        if team_index is not None:
            per_team = rng.normal(0.0, drift_sd, size=(n_sims, int(team_index.max()) + 1))
            drift = per_team[:, team_index]
        else:
            drift = rng.normal(0.0, drift_sd, size=(n_sims, n))
    else:
        drift = np.zeros((1, n))

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
        # A car that stopped scores nothing, even if a thin field sorted it
        # into the points.
        awarded = np.where(retired, 0.0, awarded)

        np.add.at(wins, order[:, 0], 1)

        totals += awarded
        track[r] = totals.mean(axis=0)
        # The mean line alone reads as a forecast of certainty. Carrying the
        # 10th and 90th percentile at every round is what lets the chart show
        # that a leader who retires twice is inside the range, not outside it.
        lo_track[r] = np.percentile(totals, 10, axis=0)
        hi_track[r] = np.percentile(totals, 90, axis=0)

    # wins accumulates one count per simulated race, so dividing by the number
    # of simulations already gives the expected number of wins across the whole
    # remaining calendar - not a per-race rate. Multiplying by n_races again is
    # the obvious mistake and the test below pins it.
    return totals, track, lo_track, hi_track, wins / n_sims


def allocate_odds(
    probs: np.ndarray,
    clinched: np.ndarray | None = None,
    rank_by: np.ndarray | None = None,
) -> np.ndarray:
    """Round title probabilities to a tenth of a percent so they still sum to 100.

    Two things have to hold at once and naive rounding breaks both. The column
    has to add up - a reader who sees 99.9% across the whole field assumes
    something leaked. And it must not print 100% for a title that is not
    mathematically won, because a simulation cannot resolve past 1 in n_sims and
    the real uncertainty is the assumption that form holds.

    So: largest-remainder apportionment over tenths of a percent, then, if the
    leader has rounded to a clean 100% without actually clinching, one tenth is
    moved to their nearest rival. Where the title IS clinched - nobody else can
    reach the leader even winning out - 100% is the correct and honest number.
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
        # Everyone else is on a simulated zero, so ordering by probability picks
        # an arbitrary name - which is how the spare tenth once landed on the
        # ninth-placed constructor. Order by points instead when given them.
        order = np.argsort(-(rank_by if rank_by is not None else probs))
        rival = int(order[1]) if len(probs) > 1 else leader
        if rival != leader:
            tenths[leader] -= 1
            tenths[rival] += 1

    return tenths / 1000.0


def clinch_round(leader_points: float, rival_points: float, races_left: int, win: float = 25.0) -> int | None:
    """The earliest round the leader could mathematically seal the title.

    Best case for the leader, worst for the rival: the leader wins every race
    from here and the rival scores nothing. After k more races the leader has
    L + 25k and the rival can still reach R + 25(races_left - k), so the title
    is settled once L + 25k > R + 25(races_left - k).

    Returns the number of races from now, or None if it cannot be settled
    before the finale. Sprints are ignored, which makes this a lower bound.
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
        }
    ).sort_values("projected", ascending=False)

    return {
        "drivers": drivers,
        "constructors": constructors,
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
