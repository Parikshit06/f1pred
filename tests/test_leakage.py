"""The tests that matter most.

Temporal leakage does not raise an exception; it just makes the backtest look
good and the live predictions bad. These tests construct cases where a leak
would be visible and assert it is not there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pred import features


def _frame(finishes: list[int], driver: str = "alpha") -> pd.DataFrame:
    """One driver, one result per round, in chronological order."""
    return pd.DataFrame(
        {
            "driver_id": [driver] * len(finishes),
            "season": [2024] * len(finishes),
            "round": range(1, len(finishes) + 1),
            "finish_or_last": finishes,
        }
    )


def test_prior_rolling_excludes_the_current_race():
    # A driver finishes 10th every time, then wins. The rolling average
    # computed AT the win must not know about the win.
    df = _frame([10, 10, 10, 10, 1])
    df["avg3"] = features._prior_rolling(df, "driver_id", "finish_or_last", 3)

    assert np.isnan(df["avg3"].iloc[0]), "first race has no prior form"
    assert df["avg3"].iloc[4] == pytest.approx(10.0), "the win leaked into its own feature"


def test_prior_rolling_is_not_merely_shifted_by_a_window():
    """Guards against the subtler bug: using .rolling().shift() instead of
    .shift().rolling(), which is off by the window length rather than by one."""
    df = _frame([1, 2, 3, 4, 5])
    df["avg2"] = features._prior_rolling(df, "driver_id", "finish_or_last", 2)
    # At round 3 the previous two results are 1 and 2.
    assert df["avg2"].iloc[2] == pytest.approx(1.5)


def test_prior_expanding_excludes_the_current_race():
    df = _frame([4, 4, 4, 20])
    df["exp"] = features._prior_expanding(df, "driver_id", "finish_or_last")
    assert df["exp"].iloc[3] == pytest.approx(4.0)


def test_rolling_does_not_bleed_between_drivers():
    a = _frame([1, 1, 1], "alpha")
    b = _frame([20, 20, 20], "beta")
    df = pd.concat([a, b], ignore_index=True).sort_values(["round", "driver_id"])
    df["avg3"] = features._prior_rolling(df, "driver_id", "finish_or_last", 3)

    last_alpha = df[(df.driver_id == "alpha") & (df["round"] == 3)]["avg3"].iloc[0]
    last_beta = df[(df.driver_id == "beta") & (df["round"] == 3)]["avg3"].iloc[0]
    assert last_alpha == pytest.approx(1.0)
    assert last_beta == pytest.approx(20.0)


@pytest.mark.parametrize("window", [1, 3, 5, 10])
def test_no_window_ever_sees_the_present(window):
    """Property check: make the current race an extreme outlier and confirm no
    rolling feature moves toward it."""
    rng = np.random.default_rng(0)
    finishes = list(rng.integers(8, 13, size=15)) + [1]
    df = _frame(finishes)
    df["roll"] = features._prior_rolling(df, "driver_id", "finish_or_last", window)
    assert df["roll"].iloc[-1] >= 8.0


# ---------------------------------------------------------------------------
# The bug this file originally missed
# ---------------------------------------------------------------------------
def _two_car_team(finishes_a: list[float], finishes_b: list[float]) -> pd.DataFrame:
    """One constructor, two drivers, one row each per race."""
    rows = []
    for rnd, (a, b) in enumerate(zip(finishes_a, finishes_b), start=1):
        rows.append(
            {
                "constructor_id": "team",
                "driver_id": "a",
                "race_seq": rnd - 1,
                "season": 2024,
                "round": rnd,
                "finish_or_last": a,
            }
        )
        rows.append(
            {
                "constructor_id": "team",
                "driver_id": "b",
                "race_seq": rnd - 1,
                "season": 2024,
                "round": rnd,
                "finish_or_last": b,
            }
        )
    return pd.DataFrame(rows)


def test_row_shift_helper_leaks_across_teammates():
    """Documents WHY team features may not use _prior_rolling.

    A constructor has two rows per race, so .shift(1) steps back to the
    teammate in the SAME race rather than to the previous race. This asserts
    the flaw exists, so nobody 'simplifies' the race-level helper away.
    """
    # Both cars finish 10th every race, then car A wins the last one.
    df = _two_car_team([10, 10, 10, 1], [10, 10, 10, 10])
    df["leaky"] = features._prior_rolling(df, "constructor_id", "finish_or_last", 3)

    last_b = df[(df.driver_id == "b") & (df["round"] == 4)]["leaky"].iloc[0]
    # Car A's win is in the same race, yet it has moved car B's team average.
    assert last_b < 10.0, "expected the row-shift helper to leak; it no longer does"


def test_race_level_helper_does_not_leak_across_teammates():
    df = _two_car_team([10, 10, 10, 1], [10, 10, 10, 10])
    df["safe"] = features._prior_rolling_by_race(df, "constructor_id", "finish_or_last", 3)

    for driver in ("a", "b"):
        value = df[(df.driver_id == driver) & (df["round"] == 4)]["safe"].iloc[0]
        assert value == pytest.approx(10.0), f"car {driver} saw this race's result"


def test_teammates_share_identical_team_features():
    """Prior team form belongs to the team, so both cars must read the same."""
    df = _two_car_team([3, 5, 7, 9], [12, 14, 16, 18])
    df["team_form"] = features._prior_rolling_by_race(df, "constructor_id", "finish_or_last", 3)
    per_race = df.groupby("round")["team_form"].nunique(dropna=False)
    assert (per_race <= 1).all(), "teammates disagree about their own team's history"


def test_race_level_helper_still_excludes_the_current_race():
    df = _two_car_team([4, 4, 4, 20], [4, 4, 4, 20])
    df["safe"] = features._prior_rolling_by_race(df, "constructor_id", "finish_or_last", None)
    assert df[df["round"] == 4]["safe"].iloc[0] == pytest.approx(4.0)


def test_every_team_feature_uses_the_race_level_helper():
    """Static guard: a future team_* feature added on the row-shift helper fails here."""
    import inspect

    source = inspect.getsource(features)
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped.startswith('df["team_'):
            continue
        if "_prior_rolling(" in stripped or "_prior_expanding(" in stripped:
            raise AssertionError(
                f"team feature uses a row-shift helper and will leak across teammates:\n  {stripped}"
            )


def test_tuning_is_bounded_on_both_sides():
    """The tuner must not evaluate on races the report will later claim are
    out-of-sample. Without an upper bound it walks to the end of the data."""
    import inspect

    from f1pred import backtest

    df = pd.DataFrame(
        {
            "race_seq": np.repeat(np.arange(120), 2),
            "season": np.repeat(np.where(np.arange(120) < 100, 2022, 2025), 2),
            "round": np.repeat(np.arange(120) + 1, 2),
            "position": 1.0,
        }
    )
    bounded = backtest.completed_races(df, 2022, 2024)
    assert bounded["season"].max() == 2022, "upper bound let a later season through"
    params = inspect.signature(backtest.tune).parameters
    assert "tune_start" in params and "tune_end" in params


# ---------------------------------------------------------------------------
# Retirements must not read as slowness
# ---------------------------------------------------------------------------
def _with_dnfs(finishes: list[float], dnf: list[bool]) -> pd.DataFrame:
    df = _frame([f if f is not None else 24 for f in finishes])
    df["dnf"] = dnf
    df["finish_when_running"] = df["finish_or_last"].where(~df["dnf"])
    return df


def test_pace_skips_retirements_without_consuming_the_window():
    """Three clean races either side of a retirement must average as three
    clean races, not as two plus a 22nd place."""
    df = _with_dnfs([4, 5, 22, 6], [False, False, True, False])
    df["pace"] = features._prior_pace(df, "driver_id", "finish_when_running", 3)

    # At the last race the prior CLEAN results are 4 and 5; the retirement is
    # skipped rather than averaged in and rather than eating a slot.
    assert df["pace"].iloc[3] == pytest.approx(4.5)
    # The plain helper is the thing being corrected: it reads the 22nd.
    df["naive"] = features._prior_rolling(df, "driver_id", "finish_or_last", 3)
    assert df["naive"].iloc[3] == pytest.approx((4 + 5 + 22) / 3)


def test_pace_at_a_retirement_row_uses_only_earlier_races():
    df = _with_dnfs([4, 5, 22, 6], [False, False, True, False])
    df["pace"] = features._prior_pace(df, "driver_id", "finish_when_running", 3)
    assert df["pace"].iloc[2] == pytest.approx(4.5), "retirement row saw its own race"


def test_pace_never_sees_the_current_race():
    df = _with_dnfs([10, 10, 10, 1], [False] * 4)
    df["pace"] = features._prior_pace(df, "driver_id", "finish_when_running", 3)
    assert df["pace"].iloc[3] == pytest.approx(10.0)
    assert np.isnan(df["pace"].iloc[0])


def test_pace_is_null_until_a_clean_race_exists():
    df = _with_dnfs([22, 22, 7], [True, True, False])
    df["pace"] = features._prior_pace(df, "driver_id", "finish_when_running", 5)
    assert df["pace"].isna().iloc[:3].all(), "invented pace from retirements alone"


def test_pace_does_not_bleed_between_drivers():
    a = _with_dnfs([2, 2, 2], [False] * 3).assign(driver_id="alpha")
    b = _with_dnfs([18, 18, 18], [False] * 3).assign(driver_id="beta")
    df = pd.concat([a, b], ignore_index=True).sort_values(["round", "driver_id"])
    df["pace"] = features._prior_pace(df, "driver_id", "finish_when_running", 3)
    assert df[df.driver_id == "alpha"]["pace"].iloc[-1] == pytest.approx(2.0)
    assert df[df.driver_id == "beta"]["pace"].iloc[-1] == pytest.approx(18.0)


def test_team_pace_does_not_leak_across_teammates():
    """The teammate hazard applies to the pace helper too."""
    df = _two_car_team([10, 10, 10, 1], [10, 10, 10, 10])
    df["dnf"] = False
    df["finish_when_running"] = df["finish_or_last"]
    df["pace"] = features._prior_pace_by_race(df, "constructor_id", "finish_when_running", 3)
    for driver in ("a", "b"):
        value = df[(df.driver_id == driver) & (df["round"] == 4)]["pace"].iloc[0]
        assert value == pytest.approx(10.0), f"car {driver} saw this race's result"


def test_a_retired_car_does_not_drag_its_teams_pace():
    df = _two_car_team([3, 3, 3, 3], [4, 4, 22, 4])
    df["dnf"] = [False, False, False, False, False, True, False, False]
    df["finish_when_running"] = df["finish_or_last"].where(~df["dnf"])
    df["pace"] = features._prior_pace_by_race(df, "constructor_id", "finish_when_running", 3)
    last = df[df["round"] == 4]["pace"].iloc[0]
    # Races 1 and 2 average 3.5; in race 3 only the running car counts, so 3.
    assert last == pytest.approx((3.5 + 3.5 + 3.0) / 3)
