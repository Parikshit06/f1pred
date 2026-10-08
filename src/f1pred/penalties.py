"""Grid penalties announced before the official starting grid exists.

A team can confirm days ahead that a car will take a new power unit and start
from the back, but the official grid that carries the penalty is only
published shortly before the race (weekend.resolve_grid). Until then the race
forecast stands on a grid it has to guess: one drawn from the qualifying
forecast, or the qualifying order once qualifying has run. Both would put the
penalised car where it qualifies.

grid_penalties.json lists the penalties confirmed so far, each with where it
was announced. They move a car down every grid the forecast guesses. They are
never applied to an official grid, which already includes them, and never to
the history the models learn from, where the real starting grid is known.
Qualifying is unaffected: a penalty moves where a car starts, not where it
qualifies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config

PATH = config.ROOT / "grid_penalties.json"
BACK = "back"
_BACK_OFFSET = 1000.0  # behind every possible grid slot


@dataclass(frozen=True)
class Penalty:
    driver_id: str
    places: int | None  # None: back of the grid
    reason: str
    source: str
    announced: str


def for_race(season: int, rnd: int, path: Path = PATH) -> list[Penalty]:
    """The penalties confirmed for one race, or none."""
    if not path.exists():
        return []
    out = []
    for row in json.loads(path.read_text()):
        if int(row["season"]) != season or int(row["round"]) != rnd:
            continue
        places = row["penalty"]
        if places != BACK and (not isinstance(places, int) or places <= 0):
            raise ValueError(
                f"penalty for {row['driver_id']} must be '{BACK}' or a positive number of places"
            )
        out.append(
            Penalty(
                driver_id=row["driver_id"],
                places=None if places == BACK else places,
                reason=row["reason"],
                source=row["source"],
                announced=row["announced"],
            )
        )
    return out


def places_for(driver_ids: list[str], penalties: list[Penalty]) -> np.ndarray:
    """Places each driver drops, in field order: 0 for none, inf for the back."""
    by_driver = {p.driver_id: (np.inf if p.places is None else float(p.places)) for p in penalties}
    return np.array([by_driver.get(d, 0.0) for d in driver_ids])


def apply(grid: np.ndarray, places: np.ndarray) -> np.ndarray:
    """The grid (or grids, one per row) after the penalties, ranked 1..n again.

    A car dropping N places lands behind the car that qualified in that slot,
    which moves up one, as the stewards apply it. Several back-of-grid starters
    keep their qualifying order among themselves.
    """
    g = np.asarray(grid, dtype=float)
    p = np.asarray(places, dtype=float)
    if not (p > 0).any():
        return g.copy()
    key = g + np.where(np.isinf(p), _BACK_OFFSET, p) + np.where(p > 0, 0.5, 0.0)
    order = np.argsort(key, axis=-1, kind="stable")
    ranks = np.broadcast_to(np.arange(1, g.shape[-1] + 1, dtype=float), g.shape)
    out = np.empty_like(g)
    np.put_along_axis(out, order, ranks, axis=-1)
    return out
