"""Tests for the spread calibration harness itself."""

from __future__ import annotations

import pandas as pd
import pytest

from f1pred import config
from f1pred import spread_calibration as sc


def test_the_fit_window_and_the_grading_window_do_not_overlap():
    """The whole claim rests on this. If a season appeared in both, the
    reported coverage would be the number the sweep was chosen to produce."""
    assert not (set(sc.FIT_SEASONS) & set(sc.GRADE_SEASONS))
    assert max(sc.FIT_SEASONS) < min(sc.GRADE_SEASONS), "the fit window sees the future"


def test_the_candidate_sweep_includes_doing_nothing():
    """Zero has to be on the grid, or the sweep cannot conclude that the term
    is unnecessary."""
    assert 0.0 in sc.CANDIDATES
    assert sc.CANDIDATES == tuple(sorted(sc.CANDIDATES))


def test_the_setting_in_use_is_one_the_sweep_could_have_picked():
    assert config.SEASON_PACE_UNCERTAINTY in sc.CANDIDATES


def test_coverage_restores_the_setting_even_when_projection_fails(monkeypatch):
    """coverage() reaches into config to swap the spread. If an exception left
    that swap in place, every later forecast in the process would silently use
    a calibration value from a diagnostic run."""

    def boom(*a, **kw):
        raise RuntimeError("no standings")

    monkeypatch.setattr(sc.championship, "project", boom)
    before = config.SEASON_PACE_UNCERTAINTY
    race = pd.DataFrame(
        {
            "driver_id": ["a"],
            "constructor_id": ["t"],
            "score": [0.0],
            "drv_dnf_rate_10": [0.1],
            "team_dnf_rate_10": [0.1],
        }
    )
    with pytest.raises(RuntimeError):
        sc.coverage([{"season": 2024, "after": 8, "elapsed": 0.35, "race": race, "actual": {}}], 99.0)
    assert config.SEASON_PACE_UNCERTAINTY == before


def test_report_says_so_rather_than_inventing_numbers_when_there_is_no_data():
    assert "not enough" in sc.report({})


def test_report_quotes_the_held_out_window_not_the_fit_window():
    frame = pd.DataFrame({"inside": [1, 0], "width": [40.0, 40.0], "elapsed": [0.35, 0.75]})
    out = sc.report(
        {
            "sweep": pd.DataFrame([{"spread": 0.0, "coverage": 0.5, "width": 38.0}]),
            "best": 6.0,
            "graded": {"before": frame, "after": frame},
            "n_checkpoints": 12,
        }
    )
    assert f"{sc.GRADE_SEASONS[0]}-{sc.GRADE_SEASONS[-1]}" in out
    assert "never seen by the sweep" in out
    assert f"{sc.TARGET_COVERAGE:.0%}" in out
