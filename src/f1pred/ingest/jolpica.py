"""Ingest race results, qualifying, sprints, pit stops and standings from jolpica-f1.

Why not lap-by-lap from here: a single race has ~1,200 lap records, which at
their 100-row page size is ~240 requests per race. Nine seasons of that would
take days under a 500/hour limit. FastF1 gives us the same laps in one file
per session, so laps come from there instead.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

import pandas as pd

from .. import config
from ..http_cache import RateLimitedSession
from ..store import already_ingested, connect, ingest_status, log_ingest, upsert

log = logging.getLogger(__name__)

_TIME_RE = re.compile(r"^(?:(\d+):)?(\d+)\.(\d+)$")

# Statuses that mean the car was running at the flag; anything else is a
# retirement. The flag feeds form, reliability and the simulator's retirement
# hazard. From 2023 a lapped car reads "Lapped" rather than "+1 Lap", so this is
# an explicit list and validate() cross-checks it against laps covered.
_FINISHED_PREFIXES = ("Finished", "+")
_FINISHED_EXACT = frozenset({"lapped"})


def is_finished(status: str | None) -> bool:
    """Did this entry run to the end of the race?"""
    if not status:
        return False
    text = status.strip()
    return text.startswith(_FINISHED_PREFIXES) or text.lower() in _FINISHED_EXACT


def parse_lap_time_ms(text: str | None) -> int | None:
    """'1:23.456' or '23.456' -> milliseconds. Returns None for blanks."""
    if not text:
        return None
    m = _TIME_RE.match(text.strip())
    if not m:
        return None
    minutes, seconds, frac = m.groups()
    frac_ms = int(frac.ljust(3, "0")[:3])
    return (int(minutes or 0) * 60 + int(seconds)) * 1000 + frac_ms


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _race_start_utc(date_str: str | None, time_str: str | None) -> datetime | None:
    """Session start as a naive datetime that is UTC by convention.

    The API always sends UTC and every timestamp column is stored naive, so no
    timezone is attached. A date with no time means midnight UTC.
    """
    if not date_str:
        return None
    try:
        if time_str:
            # DTZ007: naive on purpose, see above.
            return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%SZ")  # noqa: DTZ007
        return datetime.strptime(date_str, "%Y-%m-%d")  # noqa: DTZ007
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Parsers: one per endpoint, each returns a tidy DataFrame
# ---------------------------------------------------------------------------
def parse_races(races: list[dict]) -> pd.DataFrame:
    rows = []
    for r in races:
        circuit = r.get("Circuit", {})
        loc = circuit.get("Location", {})
        rows.append(
            {
                "season": _to_int(r.get("season")),
                "round": _to_int(r.get("round")),
                "race_name": r.get("raceName"),
                "circuit_id": circuit.get("circuitId"),
                "circuit_name": circuit.get("circuitName"),
                "locality": loc.get("locality"),
                "country": loc.get("country"),
                "lat": _to_float(loc.get("lat")),
                "lon": _to_float(loc.get("long")),
                "race_date": r.get("date"),
                "race_time": r.get("time"),
                "race_start_utc": _race_start_utc(r.get("date"), r.get("time")),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce").dt.date
    return df


def parse_drivers(drivers: list[dict], season: int) -> pd.DataFrame:
    rows = [
        {
            "season": season,
            "driver_id": d.get("driverId"),
            "code": d.get("code"),
            "permanent_number": _to_int(d.get("permanentNumber")),
            "given_name": d.get("givenName"),
            "family_name": d.get("familyName"),
            "dob": d.get("dateOfBirth"),
            "nationality": d.get("nationality"),
        }
        for d in drivers
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce").dt.date
    return df


def parse_results(races: list[dict]) -> pd.DataFrame:
    rows = []
    for r in races:
        season, rnd = _to_int(r.get("season")), _to_int(r.get("round"))
        for res in r.get("Results", []):
            status = res.get("status", "")
            position_text = res.get("positionText", "")
            rows.append(
                {
                    "season": season,
                    "round": rnd,
                    "driver_id": res.get("Driver", {}).get("driverId"),
                    "constructor_id": res.get("Constructor", {}).get("constructorId"),
                    "grid": _to_int(res.get("grid")),
                    # position is the classified order, retirements included - it's the ranking
                    # label. The DNF flag is separate, so position is never nulled here.
                    "position": _to_int(res.get("position")),
                    "classified": position_text.isdigit(),
                    "position_text": position_text,
                    "points": _to_float(res.get("points")) or 0.0,
                    "laps": _to_int(res.get("laps")),
                    "status": status,
                    "finished": is_finished(status),
                    "dnf": not is_finished(status),
                    "millis": _to_int((res.get("Time") or {}).get("millis")),
                    "fastest_lap_rank": _to_int((res.get("FastestLap") or {}).get("rank")),
                }
            )
    return pd.DataFrame(rows)


def parse_qualifying(races: list[dict]) -> pd.DataFrame:
    rows = []
    for r in races:
        season, rnd = _to_int(r.get("season")), _to_int(r.get("round"))
        for q in r.get("QualifyingResults", []):
            q1, q2, q3 = (parse_lap_time_ms(q.get(k)) for k in ("Q1", "Q2", "Q3"))
            best = min((t for t in (q1, q2, q3) if t is not None), default=None)
            rows.append(
                {
                    "season": season,
                    "round": rnd,
                    "driver_id": q.get("Driver", {}).get("driverId"),
                    "constructor_id": q.get("Constructor", {}).get("constructorId"),
                    "position": _to_int(q.get("position")),
                    "q1_ms": q1,
                    "q2_ms": q2,
                    "q3_ms": q3,
                    "best_ms": best,
                }
            )
    return pd.DataFrame(rows)


def parse_sprint(races: list[dict]) -> pd.DataFrame:
    rows = []
    for r in races:
        season, rnd = _to_int(r.get("season")), _to_int(r.get("round"))
        for s in r.get("SprintResults", []):
            rows.append(
                {
                    "season": season,
                    "round": rnd,
                    "driver_id": s.get("Driver", {}).get("driverId"),
                    "constructor_id": s.get("Constructor", {}).get("constructorId"),
                    "grid": _to_int(s.get("grid")),
                    "position": _to_int(s.get("position")),
                    "points": _to_float(s.get("points")) or 0.0,
                    "status": s.get("status"),
                }
            )
    return pd.DataFrame(rows)


def parse_pitstops(races: list[dict]) -> pd.DataFrame:
    rows = []
    for r in races:
        season, rnd = _to_int(r.get("season")), _to_int(r.get("round"))
        for p in r.get("PitStops", []):
            rows.append(
                {
                    "season": season,
                    "round": rnd,
                    "driver_id": p.get("driverId"),
                    "stop": _to_int(p.get("stop")),
                    "lap": _to_int(p.get("lap")),
                    "duration_s": _to_float(p.get("duration")),
                }
            )
    return pd.DataFrame(rows)


def parse_standings(standings_lists: list[dict]) -> pd.DataFrame:
    rows = []
    for sl in standings_lists:
        season, rnd = _to_int(sl.get("season")), _to_int(sl.get("round"))
        for s in sl.get("DriverStandings", []):
            constructors = s.get("Constructors", [])
            rows.append(
                {
                    "season": season,
                    "round": rnd,
                    "driver_id": s.get("Driver", {}).get("driverId"),
                    "constructor_id": constructors[-1].get("constructorId") if constructors else None,
                    "position": _to_int(s.get("position")),
                    "points": _to_float(s.get("points")) or 0.0,
                    "wins": _to_int(s.get("wins")) or 0,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Season-level driver
# ---------------------------------------------------------------------------
_RACE_TABLE = ("RaceTable", "Races")
_STANDINGS_TABLE = ("StandingsTable", "StandingsLists")

_SEASON_ENDPOINTS = [
    # (source name, url suffix, key path, parser, target table, primary keys)
    ("races", "races", _RACE_TABLE, parse_races, "raw_races", ["season", "round"]),
    ("results", "results", _RACE_TABLE, parse_results, "raw_results", ["season", "round", "driver_id"]),
    (
        "qualifying",
        "qualifying",
        _RACE_TABLE,
        parse_qualifying,
        "raw_qualifying",
        ["season", "round", "driver_id"],
    ),
    ("sprint", "sprint", _RACE_TABLE, parse_sprint, "raw_sprint", ["season", "round", "driver_id"]),
]

# These require a round in the path - Ergast rejects a season-wide query with
# a 400, so they are fetched one round at a time.
_ROUND_ENDPOINTS = [
    (
        "standings",
        "driverstandings",
        _STANDINGS_TABLE,
        parse_standings,
        "raw_standings",
        ["season", "round", "driver_id"],
    ),
    (
        "pitstops",
        "pitstops",
        _RACE_TABLE,
        parse_pitstops,
        "raw_pitstops",
        ["season", "round", "driver_id", "stop"],
    ),
]


def ingest_season(session: RateLimitedSession, season: int, force: bool = False) -> dict[str, int]:
    """Pull every season-level endpoint plus per-round standings."""
    counts: dict[str, int] = {}

    with connect() as con:
        # Drivers first - everything else can be joined once we have the code map.
        if force or not already_ingested(con, "drivers", str(season)) or season == config.CURRENT_SEASON:
            url = f"{config.JOLPICA_BASE}/{season}/drivers.json?limit={{limit}}&offset={{offset}}"
            try:
                records = session.paginate(url, ("DriverTable", "Drivers"), refresh=force)
                n = upsert(con, "raw_drivers", parse_drivers(records, season), ["season", "driver_id"])
                counts["drivers"] = n
                log_ingest(con, "drivers", str(season), "ok" if n else "empty", f"{n} rows")
                log.info("  %-11s season %d: %5d rows", "drivers", season, n)
            except Exception as exc:  # noqa: BLE001
                log_ingest(con, "drivers", str(season), "failed", repr(exc))
                log.error("  drivers season %d FAILED: %s", season, exc)

        for source, suffix, key_path, parser, table, keys in _SEASON_ENDPOINTS:
            scope = str(season)
            if not force and already_ingested(con, source, scope) and season != config.CURRENT_SEASON:
                # Past seasons never change. The current one does, so always refresh it.
                continue

            url = f"{config.JOLPICA_BASE}/{season}/{suffix}.json?limit={{limit}}&offset={{offset}}"
            try:
                # The live season is refetched every run, so it has to bypass the response
                # cache: the season-wide results URL never changes.
                records = session.paginate(url, key_path, refresh=force or season == config.CURRENT_SEASON)
                df = parser(records)
                n = upsert(con, table, df, keys)
                counts[source] = n
                log_ingest(con, source, scope, "ok" if n else "empty", f"{n} rows")
                log.info("  %-11s season %d: %5d rows", source, season, n)
            except Exception as exc:  # noqa: BLE001 - we want the run to continue
                log_ingest(con, source, scope, "failed", repr(exc))
                log.error("  %-11s season %d FAILED: %s", source, season, exc)

        # Per-round endpoints. Standings give us "championship position going
        # into this race"; pit stops give us strategy and stop-time features.
        rounds = [
            r[0]
            for r in con.execute(
                "SELECT round FROM raw_races WHERE season = ? ORDER BY round", [season]
            ).fetchall()
        ]

        for source, suffix, key_path, parser, table, keys in _ROUND_ENDPOINTS:
            total = 0
            for rnd in rounds:
                scope = f"{season}:{rnd}"
                status = ingest_status(con, source, scope)

                # Future rounds return an empty 200. In the live season that means "not yet",
                # so it isn't marked done or served from cache next time - otherwise the back
                # half of the standings never arrives and the projection comes back empty.
                stale = status == "empty" and season == config.CURRENT_SEASON
                if not force and status in ("ok", "empty") and not stale:
                    continue

                url = f"{config.JOLPICA_BASE}/{season}/{rnd}/{suffix}.json?limit={{limit}}&offset={{offset}}"
                try:
                    df = parser(session.paginate(url, key_path, refresh=force or stale))
                    n = upsert(con, table, df, keys)
                    total += n
                    # An empty result is normal, not a failure: a cancelled race
                    # or a season before pit stop timing existed has no rows.
                    log_ingest(con, source, scope, "ok" if n else "empty", f"{n} rows")
                except Exception as exc:  # noqa: BLE001
                    log_ingest(con, source, scope, "failed", repr(exc))
                    log.error("  %-11s %d r%d FAILED: %s", source, season, rnd, exc)

            if total:
                counts[source] = total
                log.info("  %-11s season %d: %5d rows", source, season, total)

    return counts


def ingest(seasons: list[int], force: bool = False) -> None:
    session = RateLimitedSession()
    for season in seasons:
        log.info("jolpica season %d", season)
        ingest_season(session, season, force=force)
    log.info("jolpica done - %s", session.stats())
