"""Practice pace from OpenF1: compact per-driver aggregates, nothing raw kept.

For each FP1-FP3 session, one row per driver: best lap, median clean lap,
median of long-run laps. The laps themselves stay in the HTTP cache. Only
sessions that have finished are read, so a session in progress is never
stored as if it were complete.

Pace definitions, and why:
  best_lap_ms    single fastest lap. Headline number, but noisy: one low-fuel
                 run on softs can flatter a slow car.
  median_lap_ms  median of clean green-flag laps. Much steadier signal.
  long_run_ms    median lap of stints of 5+ clean laps. The closest thing
                 practice gives to race pace, because it is run on race fuel.

A clean lap is timed, neither an out-lap nor an in-lap, run while the whole
track was clear (no yellow, red, safety car or virtual safety car at any point
of the lap, from race control's messages), and within 10% of the driver's best
clean lap. The practice history before October 2026 was fetched with FastF1
using the same definitions. Recomputed from OpenF1, best laps match it to the
millisecond and long runs to within a few hundredths of a second.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pandas as pd

from .. import config
from ..http_cache import RateLimitedSession
from ..store import connect, log_ingest, upsert
from . import openf1

log = logging.getLogger(__name__)

SOURCE = "practice"
SESSION_CODES = ("FP1", "FP2", "FP3")
MIN_STINT_LAPS = 5
# OpenF1 fills a session in as it runs. Read before this long after the
# chequered flag and the row is stored as provisional, and read again.
SETTLE = timedelta(hours=1)
_NOT_CLEAR = ("YELLOW", "DOUBLE YELLOW", "RED")


def _get(client: RateLimitedSession, endpoint: str, key: int, refresh: bool) -> pd.DataFrame:
    rows = client.get_json(
        f"{config.OPENF1_BASE}/{endpoint}?session_key={key}", refresh=refresh, not_found=[]
    )
    return pd.DataFrame(rows or [])


def _when(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format="ISO8601", utc=True)


def not_clear_periods(race_control: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Spans when any part of the track was not clear, from race control.

    A yellow in a sector lasts until that sector is cleared. A red flag or a
    safety car until the track goes green again. A span still open when the
    messages end runs to the end of the session.
    """
    if race_control.empty:
        return []
    rc = race_control.assign(t=_when(race_control["date"])).sort_values("t", kind="stable")
    active: set = set()
    spans, opened = [], None
    for m in rc.itertuples():
        was = bool(active)
        flag, scope = getattr(m, "flag", None), getattr(m, "scope", None)
        message = str(getattr(m, "message", "") or "").upper()
        if m.category == "Flag" and flag in _NOT_CLEAR:
            active.add(("sector", m.sector) if scope == "Sector" and flag != "RED" else (flag, None))
        elif m.category == "Flag" and flag == "CLEAR" and scope == "Sector":
            active.discard(("sector", m.sector))
        elif m.category == "Flag" and flag in ("CLEAR", "GREEN"):
            active.clear()
        elif m.category == "SafetyCar":
            if "DEPLOYED" in message:
                active.add(("safety car", None))
            elif any(k in message for k in ("ENDING", "IN THIS LAP", "WITHDRAWN")):
                active.discard(("safety car", None))
        if active and not was:
            opened = m.t
        elif was and not active:
            spans.append((opened, m.t))
    if active:
        spans.append((opened, pd.Timestamp.max.tz_localize("UTC")))
    return spans


def driver_pace(laps: pd.DataFrame, stints: pd.DataFrame, pits: pd.DataFrame, spans: list) -> pd.DataFrame:
    """One row per driver number: the aggregates stored in raw_session_pace."""
    laps = laps.dropna(subset=["lap_duration"]).copy()
    if laps.empty:
        return pd.DataFrame()
    start = _when(laps["date_start"])
    end = start + pd.to_timedelta(laps["lap_duration"], unit="s")
    laps["green"] = [
        pd.notna(a) and not any(lo < b and a < hi for lo, hi in spans) for a, b in zip(start, end)
    ]
    in_laps = set(zip(pits["driver_number"], pits["lap_number"])) if not pits.empty else set()
    laps["in_lap"] = [(d, n) in in_laps for d, n in zip(laps["driver_number"], laps["lap_number"])]

    rows = []
    for number, d in laps.groupby("driver_number"):
        clean = d[~d["is_pit_out_lap"].fillna(False).astype(bool) & ~d["in_lap"] & d["green"]]
        if not clean.empty:
            clean = clean[clean["lap_duration"] <= clean["lap_duration"].min() * 1.10]
        long_run, compound = None, None
        st = stints[stints["driver_number"] == number] if not stints.empty else stints
        if not clean.empty and not st.empty:
            stint_of = {}
            for s in st.itertuples():
                if pd.notna(s.lap_start) and pd.notna(s.lap_end):
                    for n in range(int(s.lap_start), int(s.lap_end) + 1):
                        stint_of[n] = (int(s.stint_number), getattr(s, "compound", None))
            tagged = clean.assign(
                stint=[stint_of.get(int(n), (None, None))[0] for n in clean["lap_number"]],
                compound=[stint_of.get(int(n), (None, None))[1] for n in clean["lap_number"]],
            )
            sized = tagged.dropna(subset=["stint"])
            sizes = sized.groupby("stint")["lap_number"].transform("size")
            long_laps = sized[sizes >= MIN_STINT_LAPS]
            if not long_laps.empty:
                long_run = round(float(long_laps["lap_duration"].median()) * 1000)
            modes = tagged["compound"].dropna().mode()
            compound = str(modes.iloc[0]) if not modes.empty else None
        rows.append(
            {
                "driver_number": number,
                "best_lap_ms": round(float(d["lap_duration"].min()) * 1000),
                "median_lap_ms": round(float(clean["lap_duration"].median()) * 1000)
                if not clean.empty
                else None,
                "long_run_ms": long_run,
                "n_laps": len(d),
                "n_clean_laps": len(clean),
                "compound_mode": compound,
            }
        )
    return pd.DataFrame(rows)


def _settled(scope: str) -> bool:
    """Stored for good: logged as complete, or already in the table from an
    earlier fetch (the history fetched with FastF1 is kept as it is)."""
    season, rnd, code = scope.split(":")
    with connect(read_only=True) as con:
        row = con.execute(
            "SELECT status, detail FROM ingest_log WHERE source = ? AND scope = ?", [SOURCE, scope]
        ).fetchone()
        stored = con.execute(
            "SELECT count(*) FROM raw_session_pace WHERE season = ? AND round = ? AND session = ?",
            [int(season), int(rnd), code],
        ).fetchone()[0]
    if row is not None:
        return row[0] == "ok" and not str(row[1]).startswith("provisional")
    return stored > 0


def _record(scope: str, status: str, detail: str) -> None:
    """Log every attempt: a run that stores nothing should say why."""
    with connect() as con:
        log_ingest(con, SOURCE, scope, status, detail[:400])
    (log.warning if status == "failed" else log.info)("practice %s: %s, %s", scope, status, detail[:200])


def ingest_session(
    client: RateLimitedSession, session: pd.Series, season: int, rnd: int, lookup: pd.DataFrame, now: datetime
) -> int:
    code = session["code"]
    scope = f"{season}:{rnd}:{code}"
    if _settled(scope):
        return 0
    ended = openf1._utc(session.get("date_end"))
    if ended is None or ended > now:
        _record(scope, "empty", "not finished yet")
        return 0
    provisional = now - ended < SETTLE
    key = int(session["session_key"])
    try:
        laps = _get(client, "laps", key, refresh=provisional)
        if laps.empty:
            _record(scope, "empty", "no laps published")
            return 0
        stints = _get(client, "stints", key, refresh=provisional)
        pits = _get(client, "pit", key, refresh=provisional)
        spans = not_clear_periods(_get(client, "race_control", key, refresh=provisional))
        drivers = openf1.fetch_session_drivers(client, session, season, lookup, refresh=provisional)
    except Exception as exc:  # noqa: BLE001 (network or rate limit: retried next run)
        _record(scope, "failed", f"will retry next run: {exc!r}")
        return 0

    pace = driver_pace(laps, stints, pits, spans)
    if pace.empty:
        _record(scope, "empty", "no timed laps")
        return 0
    ids = dict(zip(drivers["driver_number"], drivers["driver_id"]))
    pace["driver_id"] = pace["driver_number"].map(ids)
    unmapped = pace["driver_id"].isna().sum()
    if unmapped:
        log.info("practice %s: %d driver numbers not matched to a driver, left out", scope, unmapped)
    pace = pace.dropna(subset=["driver_id"]).drop(columns="driver_number")
    if pace.empty:
        _record(scope, "empty", "no laps matched to a driver")
        return 0
    pace = pace.assign(
        season=season, round=rnd, session=code, session_start_utc=openf1._utc(session["date_start"])
    )
    with connect() as con:
        n = upsert(con, "raw_session_pace", pace, ["season", "round", "session", "driver_id"])
    _record(scope, "ok", f"{'provisional, ' if provisional else ''}{n} drivers")
    return n


def _races(season: int, until: datetime) -> pd.DataFrame:
    with connect(read_only=True) as con:
        return con.execute(
            "SELECT round, race_start_utc FROM raw_races WHERE season = ? AND race_start_utc < ?::TIMESTAMP "
            "ORDER BY round",
            [season, until],
        ).fetchdf()


def ingest(seasons: list[int], next_only: bool = False) -> int:
    """Practice for every weekend already under way in these seasons, or, with
    next_only, for the race about to run. Resumable: a stored session is skipped."""
    client = openf1.http()
    now = datetime.now(UTC).replace(tzinfo=None)
    total = 0
    for season in seasons:
        if season < config.OPENF1_FIRST_SEASON:
            continue
        # A weekend's practice runs up to four days before its race.
        races = _races(season, now + timedelta(days=openf1.WEEKEND_DAYS))
        if next_only:
            races = races[pd.to_datetime(races["race_start_utc"]) > pd.Timestamp(now)].head(1)
        if races.empty:
            continue
        try:
            all_sessions = openf1.sessions(client, season, refresh=season == config.CURRENT_SEASON)
            lookup = openf1.driver_lookup(season)
        except Exception as exc:  # noqa: BLE001 (an optional source must never stop the run)
            log.warning("practice %d: sessions unavailable (%s)", season, exc)
            continue
        for race in races.itertuples():
            weekend = openf1.weekend_sessions(all_sessions, race.race_start_utc)
            got = 0
            for _, session in weekend[weekend["code"].isin(SESSION_CODES)].iterrows():
                got += ingest_session(client, session, season, int(race.round), lookup, now)
            if got:
                log.info("practice %d r%-2d: %d driver-sessions", season, int(race.round), got)
            total += got
    return total
