"""Which race the scheduled run forecasts. The failure worth guarding is a
"pre-race" forecast written for a race that has already started.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from f1pred.predict import next_race


def _calendar(rows: list[tuple[int, int, object, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["season", "round", "race_start_utc", "position"])


def test_it_skips_a_race_that_has_already_started():
    """jolpica can lag a race by hours, so a finished race may still show no
    results. Selecting on results alone would forecast it after the fact.
    """
    now = datetime.now(UTC)
    df = _calendar(
        [
            (2026, 15, now - timedelta(hours=8), None),  # started, results still pending
            (2026, 16, now + timedelta(days=6), None),
        ]
    )
    assert next_race(df) == (2026, 16)


def test_it_still_takes_a_race_that_is_only_minutes_away():
    """The guard is about races that have begun, not races that are close. A
    forecast published an hour before the lights is the normal case."""
    now = datetime.now(UTC)
    df = _calendar([(2026, 15, now + timedelta(minutes=40), None)])
    assert next_race(df) == (2026, 15)


def test_a_race_with_no_published_start_time_is_still_forecast():
    """Better to forecast a race we cannot date than to silently skip a round
    because the schedule was ingested without a time."""
    df = _calendar([(2026, 15, None, None)])
    assert next_race(df) == (2026, 15)


def test_finished_races_are_never_reconsidered():
    now = datetime.now(UTC)
    df = _calendar(
        [
            (2026, 14, now - timedelta(days=14), 1.0),
            (2026, 15, now + timedelta(days=4), None),
        ]
    )
    assert next_race(df) == (2026, 15)


def test_it_refuses_rather_than_guessing_when_the_season_is_over():
    """Returning the last race would republish a forecast for a race that has
    run. An error names the real problem: the next season is not ingested."""
    now = datetime.now(UTC)
    df = _calendar([(2026, 24, now - timedelta(days=2), None)])
    with pytest.raises(RuntimeError, match="upcoming race"):
        next_race(df)


def test_a_calendar_without_start_times_still_works():
    """Older ingests have no race_start_utc column at all. The guard is skipped
    rather than crashing the scheduled run."""
    df = pd.DataFrame([(2026, 15, None)], columns=["season", "round", "position"])
    assert next_race(df) == (2026, 15)


# ---------------------------------------------------------------------------
# What gets committed to predictions/
# ---------------------------------------------------------------------------
def _forecast(tmp_path, monkeypatch, *, start, grid_known=False, quali=None, practice=False):
    from f1pred import config
    from f1pred.predict import Prediction

    monkeypatch.setattr(config, "PREDICTIONS", tmp_path)
    return Prediction(
        season=2026,
        round=15,
        race_name="Azerbaijan Grand Prix",
        circuit_id="baku",
        race_start_utc=start.isoformat(),
        generated_at_utc=datetime.now(UTC).isoformat(),
        grid_known=grid_known,
        quali_start_utc=quali.isoformat() if quali else None,
        meta={"practice_data": practice},
    )


def test_a_race_week_forecast_is_committed(tmp_path, monkeypatch):
    p = _forecast(tmp_path, monkeypatch, start=datetime.now(UTC) + timedelta(days=3))
    assert p.save() is not None
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_a_forecast_a_fortnight_out_is_not_committed(tmp_path, monkeypatch):
    """The Monday run that grades a result rolls on to the next race, and on a
    weekend off that race can be two weeks away. Logging then would put the
    least informed call of the season into the record - and because the first
    file for a stage wins, it would keep the race-week call out."""
    p = _forecast(tmp_path, monkeypatch, start=datetime.now(UTC) + timedelta(days=14))
    assert p.save() is None
    assert list(tmp_path.glob("*.json")) == []


def test_the_race_week_call_still_lands_after_an_early_run_was_withheld(tmp_path, monkeypatch):
    early = _forecast(tmp_path, monkeypatch, start=datetime.now(UTC) + timedelta(days=13))
    assert early.save() is None
    later = _forecast(tmp_path, monkeypatch, start=datetime.now(UTC) + timedelta(days=2))
    assert later.save() is not None


def test_a_post_qualifying_forecast_is_never_withheld(tmp_path, monkeypatch):
    """Grid known means the weekend is underway, whatever the clock says."""
    p = _forecast(tmp_path, monkeypatch, start=datetime.now(UTC) + timedelta(days=30), grid_known=True)
    assert p.save() is not None


def test_one_file_per_race_per_stage(tmp_path, monkeypatch):
    first = _forecast(tmp_path, monkeypatch, start=datetime.now(UTC) + timedelta(days=2))
    kept = first.save()
    again = _forecast(tmp_path, monkeypatch, start=datetime.now(UTC) + timedelta(days=2))
    assert again.save() == kept, "a second run wrote a near-identical file"
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_the_pre_qualifying_call_waits_for_practice(tmp_path, monkeypatch):
    """Practice pace improves the qualifying forecast, so the logged call is
    made once it is in rather than on the first run of race week."""
    now = datetime.now(UTC)
    early = _forecast(tmp_path, monkeypatch, start=now + timedelta(days=2), quali=now + timedelta(days=1))
    assert early.save() is None
    with_practice = _forecast(
        tmp_path, monkeypatch, start=now + timedelta(days=2), quali=now + timedelta(days=1), practice=True
    )
    assert with_practice.save() is not None


def test_a_practice_outage_cannot_cost_the_record_a_race(tmp_path, monkeypatch):
    now = datetime.now(UTC)
    late = _forecast(tmp_path, monkeypatch, start=now + timedelta(hours=26), quali=now + timedelta(hours=3))
    assert late.save() is not None
