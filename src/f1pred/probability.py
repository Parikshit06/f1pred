"""Ranking scores -> one finishing-position distribution -> every published number.

A forecast publishes win, podium, top-5, top-10 and expected finish. All of
them are read from a single matrix P[driver, position] whose rows and columns
each sum to one, so they cannot contradict each other: P(win) <= P(podium) <=
P(top 5) <= P(top 10) for every driver by construction, the field's win
chances sum to 1, its podium chances to 3.

The matrix mixes two models of the same race:

    Plackett-Luce  the ranker's scores read as strengths, exp(score / T), with
                   the temperature T fitted by log loss on earlier races only.
                   Sampled exactly via the Gumbel-max trick.
    Monte Carlo    pace noise, retirements, safety cars and, before
                   qualifying, an uncertain grid (simulate.py).

The mixing weight is fitted on the tuning seasons. A mixture of two matrices
whose rows and columns sum to one has the same property, which is what keeps
every published number consistent. (The version before this blended only
P(win) and floored the other bands afterwards, so the bands no longer summed
to 3, 5 and 10.)

A small uniform share (SMOOTHING) is mixed in last, so no driver in the race
is ever given exactly zero - Monte Carlo counts produce zeros for anyone never
seen winning, and a zero is infinitely wrong the one time it happens.

Ranking scores carry no probabilistic meaning of their own, so none of this is
assumed calibrated: reliability is measured in evaluation.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

SMOOTHING = 0.002
ADAPTIVE_WINDOW = 24  # races, roughly one season
MIN_CALIBRATION_RACES = 10
TEMPERATURE_GRID = np.concatenate([np.linspace(0.05, 2.0, 40), np.linspace(2.2, 8.0, 30)])


# ---------------------------------------------------------------------------
# Plackett-Luce
# ---------------------------------------------------------------------------
def plackett_luce(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """P(each entry finishes first), in closed form.

    Temperature controls how decisive the model is. Below 1 sharpens toward the
    favourite, above 1 flattens. Fitted on earlier races, because a ranker's
    raw score scale carries no probabilistic meaning on its own.
    """
    z = np.asarray(scores, dtype=float) / max(temperature, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sample_orders(
    rng: np.random.Generator, scores: np.ndarray, temperature: float, n_samples: int
) -> np.ndarray:
    """Full finishing orders drawn from Plackett-Luce, one row per sample.

    Adding independent Gumbel noise to score / T and sorting draws an exact
    Plackett-Luce ranking - the same distribution as picking the winner, then
    second from the rest, and so on - in one vectorised step.
    """
    z = np.asarray(scores, dtype=float) / max(temperature, 1e-6)
    noisy = z[None, :] + rng.gumbel(size=(n_samples, len(z)))
    return np.argsort(-noisy, axis=1)


def position_counts(orders: np.ndarray) -> np.ndarray:
    """Orders (sample x position -> driver) to a matrix [driver, position] of shares."""
    n_samples, n = orders.shape
    flat = orders * n + np.arange(n)[None, :]
    counts = np.bincount(flat.ravel(), minlength=n * n).reshape(n, n)
    return counts / n_samples


def pl_position_matrix(
    scores: np.ndarray, temperature: float, n_samples: int = 20_000, seed: int = config.RANDOM_SEED
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return position_counts(sample_orders(rng, scores, temperature, n_samples))


# ---------------------------------------------------------------------------
# Calibration of the temperature
# ---------------------------------------------------------------------------
def fit_temperature(
    score_groups: list[np.ndarray], winner_idx: list[int], grid: np.ndarray | None = None
) -> float:
    """The temperature that minimises log loss on observed winners."""
    grid = TEMPERATURE_GRID if grid is None else grid
    best_t, best_loss = 1.0, np.inf
    for t in grid:
        losses = []
        for scores, win in zip(score_groups, winner_idx):
            if win is None or win < 0 or win >= len(scores):
                continue
            p = plackett_luce(scores, t)
            losses.append(-np.log(max(p[win], 1e-12)))
        if losses:
            loss = float(np.mean(losses))
            if loss < best_loss:
                best_t, best_loss = float(t), loss
    return best_t


def rolling_temperature(
    score_groups: list[np.ndarray],
    winner_idx: list[int],
    fallback: float,
    window: int = ADAPTIVE_WINDOW,
) -> float:
    """Temperature fitted on a trailing window of finished races.

    One fixed value doesn't hold across eras: a season with one dominant car
    wants a sharper forecast than a close one. Refitting on the last races -
    all of them before the one being forecast - lets confidence follow the
    field. Until there are enough of them, the tuned fallback is used.
    """
    if len(score_groups) < MIN_CALIBRATION_RACES:
        return fallback
    return fit_temperature(score_groups[-window:], winner_idx[-window:])


# ---------------------------------------------------------------------------
# One distribution, everything read from it
# ---------------------------------------------------------------------------
def mix(p_model: np.ndarray, p_sim: np.ndarray, weight: float, smoothing: float = SMOOTHING) -> np.ndarray:
    """weight = trust in the closed-form model over the simulation.

    Works on win vectors or on full position matrices; either way the result
    keeps the sums of its inputs.
    """
    p_model, p_sim = np.asarray(p_model, dtype=float), np.asarray(p_sim, dtype=float)
    return smooth(weight * p_model + (1.0 - weight) * p_sim, smoothing)


def smooth(p: np.ndarray, smoothing: float = SMOOTHING) -> np.ndarray:
    """Mix in a small uniform share, so nobody in the race is ever at exactly zero."""
    p = np.asarray(p, dtype=float)
    return (1.0 - smoothing) * p + smoothing * np.full_like(p, 1.0 / p.shape[0])


def summarise(matrix: np.ndarray) -> pd.DataFrame:
    """Every published probability, from the one matrix."""
    m = np.asarray(matrix, dtype=float)
    positions = np.arange(1, m.shape[1] + 1)
    cum = np.cumsum(m, axis=1)

    def upto(k: int) -> np.ndarray:
        return cum[:, min(k, m.shape[1]) - 1]

    return pd.DataFrame(
        {
            "p_win": m[:, 0],
            "p_podium": upto(3),
            "p_top5": upto(5),
            "p_top10": upto(10),
            "exp_position": m @ positions,
        }
    )


def check_distribution(matrix: np.ndarray, atol: float = 1e-6) -> list[str]:
    """Problems with a position matrix; empty if it is a coherent distribution."""
    m = np.asarray(matrix, dtype=float)
    problems = []
    if not np.isfinite(m).all():
        problems.append("non-finite entries")
    if (m < -atol).any() or (m > 1 + atol).any():
        problems.append("entries outside [0, 1]")
    if not np.allclose(m.sum(axis=1), 1.0, atol=atol):
        problems.append("a driver's positions don't sum to 1")
    if not np.allclose(m.sum(axis=0), 1.0, atol=atol):
        problems.append("a position isn't filled exactly once")
    return problems
