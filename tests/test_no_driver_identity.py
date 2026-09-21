"""The model must not know who anyone is.

Every feature is a number computed from prior results. No driver id, no name,
no team name reaches the estimator. These tests prove that behaviourally rather
than by reading the code, because "I checked and there's no hardcoding" is not
evidence anyone should accept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pred import features, model, simulate


def _synthetic_history(n_races: int = 60, n_drivers: int = 10, seed: int = 3) -> pd.DataFrame:
    """A made-up championship with made-up names and a known pecking order."""
    rng = np.random.default_rng(seed)
    skill = np.linspace(0.5, 9.5, n_drivers)  # driver 0 fastest
    rows = []
    for race in range(n_races):
        noise = rng.normal(0, 1.5, n_drivers)
        order = np.argsort(skill + noise)
        for finish, idx in enumerate(order, start=1):
            rows.append(
                {
                    "race_seq": race,
                    "season": 2024 + race // 24,
                    "round": race % 24 + 1,
                    "driver_id": f"driver_{idx}",
                    "constructor_id": f"team_{idx // 2}",
                    "position": float(finish),
                    "quali_position": float(finish),
                    "grid": float(finish),
                    "points": max(0.0, 26 - 2 * finish),
                    "dnf": False,
                    "finish_or_last": float(finish),
                }
            )
    return pd.DataFrame(rows)


def test_no_identity_feature_reaches_the_model():
    """Static: the feature lists must contain nothing that names anyone."""
    banned = ("driver_id", "constructor_id", "name", "code", "nationality")
    for feature in set(features.RACE_FEATURES + features.QUALI_FEATURES):
        for token in banned:
            assert token not in feature, f"{feature} carries identity into the model"


def test_source_contains_no_hardcoded_driver_or_team_names():
    """A driver named in the modelling code is a red flag whatever its intent."""
    import inspect

    from f1pred import backtest, predict, simulate as sim_mod

    names = (
        "hamilton",
        "verstappen",
        "antonelli",
        "norris",
        "russell",
        "leclerc",
        "piastri",
        "alonso",
        "perez",
        "ferrari",
        "mercedes",
        "mclaren",
        "red_bull",
    )
    for module in (features, model, sim_mod, backtest, predict):
        source = inspect.getsource(module).lower()
        for name in names:
            assert name not in source, f"{module.__name__} mentions {name!r}"


def test_swapping_two_drivers_features_swaps_their_predictions():
    """The decisive test.

    Exchange every feature value between two entries and leave the names where
    they are. If the model were keyed to identity in any way, the predictions
    would stay put. They must follow the numbers instead - exactly, and without
    disturbing anybody else.
    """
    df = _synthetic_history()
    feats = ["prior_form", "prior_quali"]
    df["prior_form"] = features._prior_rolling(df, "driver_id", "finish_or_last", 5)
    df["prior_quali"] = features._prior_rolling(df, "driver_id", "quali_position", 5)
    df["race_relevance"] = (24 - df["position"]).clip(lower=0)

    train = df[df.race_seq < 55]
    ranker = model.train(train, feats, "race_relevance", n_seeds=2)

    race = df[df.race_seq == 58].copy().reset_index(drop=True)
    before = pd.Series(ranker.score(race), index=race.driver_id.values)

    a, b = race.index[0], race.index[1]
    name_a, name_b = race.loc[a, "driver_id"], race.loc[b, "driver_id"]
    swapped = race.copy()
    tmp = swapped.loc[a, feats].copy()
    swapped.loc[a, feats] = swapped.loc[b, feats].values
    swapped.loc[b, feats] = tmp.values
    after = pd.Series(ranker.score(swapped), index=swapped.driver_id.values)

    assert after[name_a] == pytest.approx(before[name_b]), "prediction did not follow the numbers"
    assert after[name_b] == pytest.approx(before[name_a])
    for other in before.index:
        if other not in (name_a, name_b):
            assert after[other] == pytest.approx(before[other]), f"{other} moved for no reason"


def test_renaming_every_driver_changes_nothing():
    """Relabel the whole field; identical features must give identical scores."""
    df = _synthetic_history()
    feats = ["prior_form"]
    df["prior_form"] = features._prior_rolling(df, "driver_id", "finish_or_last", 5)
    df["race_relevance"] = (24 - df["position"]).clip(lower=0)

    ranker = model.train(df[df.race_seq < 55], feats, "race_relevance", n_seeds=2)
    race = df[df.race_seq == 58].copy()

    original = ranker.score(race)
    renamed = race.copy()
    renamed["driver_id"] = ["anon_" + str(i) for i in range(len(renamed))]
    renamed["constructor_id"] = "anon_team"

    assert np.allclose(original, ranker.score(renamed))


def test_a_driver_with_better_numbers_is_ranked_higher():
    """Sanity in the other direction: the numbers must actually drive the order."""
    df = _synthetic_history()
    feats = ["prior_form"]
    df["prior_form"] = features._prior_rolling(df, "driver_id", "finish_or_last", 5)
    df["race_relevance"] = (24 - df["position"]).clip(lower=0)

    ranker = model.train(df[df.race_seq < 55], feats, "race_relevance", n_seeds=2)
    race = df[df.race_seq == 58].copy().reset_index(drop=True)
    race["prior_form"] = np.linspace(1.0, 18.0, len(race))  # first entry clearly fastest

    scores = ranker.score(race)
    assert scores[0] > scores[-1], "better prior form did not produce a better score"


def test_simulation_is_also_identity_blind():
    """The Monte Carlo takes names only as labels for its output rows."""
    n = 6
    scores = np.linspace(2.0, -2.0, n)
    a = simulate.simulate(
        simulate.SimInputs(driver_ids=[f"x{i}" for i in range(n)], scores=scores, dnf_prob=np.full(n, 0.05)),
        n_sims=1500,
        seed=11,
    )
    b = simulate.simulate(
        simulate.SimInputs(
            driver_ids=[f"totally_different_{i}" for i in range(n)],
            scores=scores,
            dnf_prob=np.full(n, 0.05),
        ),
        n_sims=1500,
        seed=11,
    )
    assert np.allclose(a["p_win"].to_numpy(), b["p_win"].to_numpy())
