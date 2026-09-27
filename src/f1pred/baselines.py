"""Forecasts that need no model, for the model to be measured against.

The starting grid is the one that matters: it predicts a race well on its
own, so the question a model has to answer is what it adds beyond it.

Each baseline is an ordering of the field. Its probabilities are made the same
way as the model's: Plackett-Luce over -log(rank), with a temperature fitted by
log loss on earlier races only. A baseline given a fixed, hand-picked decay -
as this project once did - loses on log loss because nobody tuned it, and the
model's lead on probability is then partly an artefact.

    grid          the official starting grid, pole first
    championship  championship order going into the race
    recent_form   average finish over the last three races
    team_form     the team's average finish over the last three races
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import simulate

KINDS = ("grid", "championship", "recent_form", "team_form")
NAMES = {
    "model": "This model",
    "grid": "Grid order",
    "championship": "Championship order",
    "recent_form": "Recent driver form",
    "team_form": "Team form",
}


def order(race: pd.DataFrame, kind: str) -> pd.Series:
    """The baseline's ranking of the field (1 = predicted winner), ties averaged.

    Drivers without a value (no grid slot, no standing yet) share the back.
    """
    if kind == "grid":
        value = pd.Series(simulate.starting_grid(race["grid"]), index=race.index)
    elif kind == "championship":
        value = race["champ_position_before"]
    elif kind == "recent_form":
        value = race["drv_avg_finish_3"]
    elif kind == "team_form":
        value = race["team_avg_finish_3"]
    else:
        raise ValueError(kind)
    value = value.fillna(value.max() + 1 if value.notna().any() else 1.0)
    return pd.Series(value.rank(method="average").to_numpy(), index=race["driver_id"].to_numpy())


def scores(race: pd.DataFrame, kind: str) -> np.ndarray:
    """Strength for Plackett-Luce: P(win) ends up proportional to rank^(-1/T)."""
    return -np.log(order(race, kind).to_numpy())
