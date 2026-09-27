"""Checks on the grader for the championship projection.

The title projection prints numbers above 99%, so the thing that grades it has
to be right or the reassurance is worthless.
"""

from __future__ import annotations

import pandas as pd
import pytest

from f1pred import title_backtest as tb


def _frame(rows):
    return pd.DataFrame(
        [
            {
                "season": s,
                "after_round": 10,
                "races_left": 8,
                "favourite": f,
                "p_favourite": p,
                "champion": c,
                "favourite_was_right": int(f == c),
                "p_on_actual_champion": p if f == c else 0.0,
            }
            for s, f, p, c in rows
        ]
    )


def test_a_favourite_who_lost_is_scored_as_wrong():
    f = _frame([(2021, "hamilton", 0.62, "max_verstappen")])
    assert f["favourite_was_right"].iloc[0] == 0


def test_calibration_buckets_by_what_was_claimed():
    f = _frame(
        [
            (2019, "a", 0.99, "a"),
            (2020, "b", 0.97, "b"),
            (2021, "c", 0.60, "d"),
            (2022, "e", 0.55, "e"),
        ]
    )
    cal = tb.calibration(f)
    top = cal[cal["bucket"] == "over 95%"].iloc[0]
    assert top["n"] == 2 and top["happened"] == pytest.approx(1.0)
    mid = cal[cal["bucket"] == "50-80%"].iloc[0]
    assert mid["n"] == 2 and mid["happened"] == pytest.approx(0.5)


def test_small_buckets_carry_a_wide_interval():
    """Sixteen out of sixteen is not proof of 99.9%. The interval is what stops
    the table being read as one."""
    f = _frame([(2000 + i, "a", 0.99, "a") for i in range(16)])
    cal = tb.calibration(f)
    row = cal[cal["bucket"] == "over 95%"].iloc[0]
    assert row["happened"] == pytest.approx(1.0)
    assert row["ci_low"] < 0.85, "a perfect record on 16 tries implies far less certainty"


def test_wilson_matches_the_other_implementation():
    from f1pred import metrics

    assert tb._wilson(16, 16) == pytest.approx(metrics.wilson(16, 16), abs=1e-3)
    assert tb._wilson(0, 0) == (0.0, 1.0)


def test_report_survives_having_nothing_to_grade():
    assert "No completed seasons" in tb.report(pd.DataFrame())


def test_checkpoints_are_spread_across_the_season():
    assert len(tb.CHECKPOINT_FRACTIONS) >= 3
    assert min(tb.CHECKPOINT_FRACTIONS) < 0.5 < max(tb.CHECKPOINT_FRACTIONS)
    assert max(tb.CHECKPOINT_FRACTIONS) < 1.0, "a checkpoint after the last race grades nothing"
