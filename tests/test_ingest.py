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
