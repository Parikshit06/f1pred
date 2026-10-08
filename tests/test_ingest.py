"""Ingest bookkeeping: what counts as done, and what has to be asked again.

The parsers are pure and easy; the part that has actually gone wrong is the
part that decides not to fetch something.
"""

from __future__ import annotations

import json
from typing import ClassVar

import duckdb
import pytest

from f1pred import store
from f1pred.http_cache import RateLimitedSession
from f1pred.ingest import jolpica


@pytest.fixture
def con():
    c = duckdb.connect(":memory:")
    c.execute(store.SCHEMA)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# ingest_log
# ---------------------------------------------------------------------------
def test_status_distinguishes_ok_from_empty_from_never_tried(con):
    """`already_ingested` collapses the three, and for per-round scopes that
    collapse is the bug: an empty round of a live season is not done."""
    store.log_ingest(con, "standings", "2026:14", "ok", "20 rows")
    store.log_ingest(con, "standings", "2026:15", "empty", "0 rows")

    assert store.ingest_status(con, "standings", "2026:14") == "ok"
    assert store.ingest_status(con, "standings", "2026:15") == "empty"
    assert store.ingest_status(con, "standings", "2026:16") is None

    # Both still read as attempted, which is all the old helper could say.
    assert store.already_ingested(con, "standings", "2026:14")
    assert store.already_ingested(con, "standings", "2026:15")
    assert not store.already_ingested(con, "standings", "2026:16")


def test_log_ingest_is_idempotent(con):
    store.log_ingest(con, "standings", "2026:15", "empty", "0 rows")
    store.log_ingest(con, "standings", "2026:15", "ok", "20 rows")
    assert con.execute("SELECT count(*) FROM ingest_log").fetchone()[0] == 1
    assert store.ingest_status(con, "standings", "2026:15") == "ok"


# ---------------------------------------------------------------------------
# The response cache
# ---------------------------------------------------------------------------
class _FakeResponse:
    status_code = 200
    headers: ClassVar[dict] = {}

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _session(tmp_path, payloads):
    """A session that serves `payloads` in order and never sleeps."""
    s = RateLimitedSession(cache_dir=tmp_path, min_interval=0.0)
    calls = []

    def get(url, timeout=30):
        calls.append(url)
        return _FakeResponse(payloads[min(len(calls) - 1, len(payloads) - 1)])

    s._session.get = get
    return s, calls


def test_cached_response_costs_no_request(tmp_path):
    s, calls = _session(tmp_path, [{"MRData": {"total": "0"}}])
    s.get_json("https://example.test/a")
    s.get_json("https://example.test/a")
    assert len(calls) == 1
    assert s.cache_hits == 1


def test_refresh_goes_back_to_the_network_and_replaces_the_cache(tmp_path):
    """A 200 with an empty body is still a 200, and gets cached by URL.

    Standings for a round that has not run yet look exactly like that, so
    without a way to bypass the cache the answer stays empty for the rest of
    the season however many times the ingest runs.
    """
    empty = {"MRData": {"total": "0", "StandingsTable": {"StandingsLists": []}}}
    filled = {"MRData": {"total": "1", "StandingsTable": {"StandingsLists": [{"season": "2026"}]}}}
    s, calls = _session(tmp_path, [empty, filled])

    assert s.get_json("https://example.test/s") == empty
    assert s.get_json("https://example.test/s", refresh=True) == filled
    assert len(calls) == 2
    # The replacement is what the next plain read sees.
    assert s.get_json("https://example.test/s") == filled
    assert len(calls) == 2


def test_rate_window_survives_a_restart(tmp_path):
    s, _ = _session(tmp_path, [{"MRData": {"total": "0"}}])
    s.get_json("https://example.test/a")
    assert json.loads((tmp_path / "_rate_window.json").read_text())

    # A fresh client on the same cache directory inherits the spent budget.
    again = RateLimitedSession(cache_dir=tmp_path, min_interval=0.0)
    assert len(again._window) == 1


# ---------------------------------------------------------------------------
# Status parsing - the rule that misfiled 398 finishes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "status",
    ["Finished", "+1 Lap", "+2 Laps", "Lapped", "lapped"],
)
def test_these_all_ran_to_the_flag(status):
    assert jolpica.is_finished(status)


@pytest.mark.parametrize(
    "status",
    ["Engine", "Accident", "Collision", "Gearbox", "Retired", "Did not start", "", None],
)
def test_these_did_not(status):
    assert not jolpica.is_finished(status)


def test_lap_times_parse_in_both_spellings():
    assert jolpica.parse_lap_time_ms("1:23.456") == 83456
    assert jolpica.parse_lap_time_ms("23.456") == 23456
    assert jolpica.parse_lap_time_ms("23.4") == 23400
    assert jolpica.parse_lap_time_ms("") is None
    assert jolpica.parse_lap_time_ms("no time") is None


def test_a_practice_session_read_too_soon_after_the_flag_is_read_again(tmp_path, monkeypatch):
    """OpenF1 fills a session in as it runs. A copy taken just after the
    chequered flag is stored but marked provisional, and fetched again."""
    from f1pred import config
    from f1pred.ingest import practice

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "f1.duckdb")
    store.init_db()
    with store.connect() as c:
        store.log_ingest(c, "practice", "2026:17:FP1", "ok", "provisional, 22 drivers")
        store.log_ingest(c, "practice", "2026:16:FP1", "ok", "22 drivers")
        store.log_ingest(c, "practice", "2026:16:FP2", "failed", "will retry next run")
        store.log_ingest(c, "practice", "2026:16:FP3", "empty", "not finished yet")

    assert not practice._settled("2026:17:FP1")
    assert practice._settled("2026:16:FP1")
    assert not practice._settled("2026:16:FP2")
    assert not practice._settled("2026:16:FP3")

    with store.connect() as c:
        c.execute(
            "INSERT INTO raw_session_pace (season, round, session, driver_id, best_lap_ms) "
            "VALUES (2025, 3, 'FP1', 'norris', 90000)"
        )
    assert practice._settled("2025:3:FP1"), "history already stored is not fetched again"
    assert not practice._settled("2025:3:FP2")


def _laps(rows):
    import pandas as pd

    return pd.DataFrame(
        rows, columns=["driver_number", "lap_number", "date_start", "lap_duration", "is_pit_out_lap"]
    )


def test_practice_pace_drops_out_laps_in_laps_and_laps_under_a_yellow():
    import pandas as pd

    from f1pred.ingest import practice

    t = "2026-10-09T08:{:02d}:00+00:00"
    laps = _laps(
        [
            (1, 1, t.format(0), 100.0, True),  # out-lap
            (1, 2, t.format(2), 90.0, False),
            (1, 3, t.format(4), 90.4, False),
            (1, 4, t.format(6), 89.0, False),  # under the yellow below
            (1, 5, t.format(8), 90.2, False),
            (1, 6, t.format(10), 90.6, False),
            (1, 7, t.format(12), 90.8, False),
            (1, 8, t.format(14), 120.0, False),  # in-lap
        ]
    )
    stints = pd.DataFrame(
        [{"driver_number": 1, "stint_number": 1, "lap_start": 1, "lap_end": 8, "compound": "SOFT"}]
    )
    pits = pd.DataFrame([{"driver_number": 1, "lap_number": 8}])
    race_control = pd.DataFrame(
        [
            {
                "date": t.format(6),
                "category": "Flag",
                "flag": "YELLOW",
                "scope": "Sector",
                "sector": 4,
                "message": "",
            },
            {
                "date": "2026-10-09T08:07:00+00:00",
                "category": "Flag",
                "flag": "CLEAR",
                "scope": "Sector",
                "sector": 4,
                "message": "",
            },
        ]
    )
    spans = practice.not_clear_periods(race_control)
    row = practice.driver_pace(laps, stints, pits, spans).iloc[0]
    assert row["best_lap_ms"] == 89000, "the best lap is any timed lap"
    assert row["n_clean_laps"] == 5
    assert row["long_run_ms"] == 90400, "median of the five clean laps"
    assert row["compound_mode"] == "SOFT"


def test_a_short_stint_is_not_a_long_run():
    import pandas as pd

    from f1pred.ingest import practice

    t = "2026-10-09T08:{:02d}:00+00:00"
    laps = _laps([(4, n, t.format(2 * n), 91.0 + n / 10, False) for n in range(1, 5)])
    stints = pd.DataFrame(
        [{"driver_number": 4, "stint_number": 1, "lap_start": 1, "lap_end": 4, "compound": "MEDIUM"}]
    )
    row = practice.driver_pace(laps, stints, pd.DataFrame(), []).iloc[0]
    assert row["long_run_ms"] is None or pd.isna(row["long_run_ms"])
    assert row["best_lap_ms"] == 91100


def test_a_safety_car_or_red_flag_holds_until_the_track_is_green():
    import pandas as pd

    from f1pred.ingest import practice

    rc = pd.DataFrame(
        [
            {
                "date": "2026-10-09T08:10:00+00:00",
                "category": "Flag",
                "flag": "RED",
                "scope": "Track",
                "sector": None,
                "message": "RED FLAG",
            },
            {
                "date": "2026-10-09T08:11:00+00:00",
                "category": "Flag",
                "flag": "CLEAR",
                "scope": "Sector",
                "sector": 3,
                "message": "",
            },
            {
                "date": "2026-10-09T08:20:00+00:00",
                "category": "Flag",
                "flag": "GREEN",
                "scope": "Track",
                "sector": None,
                "message": "GREEN LIGHT",
            },
        ]
    )
    spans = practice.not_clear_periods(rc)
    assert len(spans) == 1
    assert spans[0][1] == pd.Timestamp("2026-10-09T08:20:00+00:00")
