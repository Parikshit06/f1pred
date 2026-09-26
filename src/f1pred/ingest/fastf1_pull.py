"""Practice pace via FastF1: compact per-driver aggregates, nothing raw kept.

For each FP1-FP3 session, one row per driver: best lap, median clean lap,
median of long-run laps. Lap timing only - no telemetry, no weather. FastF1
caches its downloads in data/fastf1_cache (gitignored); the database keeps only
the aggregates. Resumable: each finished session is recorded in ingest_log.

The live forecast only needs the weekend in progress (ingest_next_weekend).
History is fetched only for the seasons being evaluated, because practice pace
is an experiment until the walk-forward says it helps.

Pace definitions, and why:
  best_lap_ms    single fastest lap. Headline number, but noisy: one low-fuel
                 run on softs can flatter a slow car.
  median_lap_ms  median of clean green-flag laps. Much steadier signal.
  long_run_ms    median lap of stints of 5+ laps. The closest thing practice
                 gives us to race pace, because it is run on race fuel.
"""

from __future__ import annotations

import logging
import warnings
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from .. import config
from ..store import connect, log_ingest, upsert

log = logging.getLogger(__name__)

# Practice only. Qualifying and race laps feed no feature (jolpica has the
# classifications), and loading them tripled the calls against FastF1's hourly
# limit. Sessions that don't exist on a weekend (FP2/FP3 at a sprint) are skipped.
SESSION_CODES = ["FP1", "FP2", "FP3"]

# A session that genuinely doesn't exist is settled for good. Anything else -
# a rate limit, a dropped connection - is worth another try. Recording those as
# "empty" is how every 2025 session came to be skipped forever: the first
# backfill hit FastF1's 500 calls/hour limit and each refusal was filed as
# "this session has no data".
_ABSENT = ("does not exist", "no laps")

MIN_STINT_LAPS = 5


def _setup_fastf1() -> Any:
    import fastf1

    fastf1.Cache.enable_cache(str(config.FASTF1_CACHE))
    # FastF1 is chatty about missing data for older/odd sessions; we handle it.
    warnings.filterwarnings("ignore", module="fastf1")
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    return fastf1


def _code_to_driver_id(season: int) -> dict[str, str]:
    with connect(read_only=True) as con:
        rows = con.execute(
            "SELECT code, driver_id FROM raw_drivers WHERE season = ? AND code IS NOT NULL",
            [season],
        ).fetchall()
    return {code: driver_id for code, driver_id in rows}


def _clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Green-flag, non-in/out, plausibly representative laps."""
    df = laps.copy()
    if df.empty:
        return df

    mask = df["LapTime"].notna()
    if "PitInTime" in df:
        mask &= df["PitInTime"].isna()
    if "PitOutTime" in df:
        mask &= df["PitOutTime"].isna()
    if "TrackStatus" in df:
        # '1' is all-clear. Anything else means yellow/SC/red somewhere on track.
        mask &= df["TrackStatus"].astype(str) == "1"
    if "IsAccurate" in df:
        mask &= df["IsAccurate"].fillna(False)

    df = df[mask]
    if df.empty:
        return df

    # Drop laps more than 10% slower than the driver's own best: traffic,
    # cooldown laps, deliberate backing-off. Keeps the median honest.
    best = df.groupby("Driver")["LapTime"].transform("min")
    return df[df["LapTime"] <= best * 1.10]


def _driver_pace(laps: pd.DataFrame, clean: pd.DataFrame, code: str) -> dict[str, Any]:
    all_d = laps[laps["Driver"] == code]
    cln = clean[clean["Driver"] == code] if not clean.empty else clean

    def ms(series: pd.Series) -> int | None:
        if series is None or series.empty or series.isna().all():
            return None
        return int(series.dropna().dt.total_seconds().median() * 1000)

    best_ms = None
    if not all_d.empty and all_d["LapTime"].notna().any():
        best_ms = int(all_d["LapTime"].min().total_seconds() * 1000)

    long_run_ms = None
    if not cln.empty and "Stint" in cln:
        stint_sizes = cln.groupby("Stint")["LapTime"].transform("size")
        long_laps = cln[stint_sizes >= MIN_STINT_LAPS]
        long_run_ms = ms(long_laps["LapTime"]) if not long_laps.empty else None

    compound_mode = None
    if not cln.empty and "Compound" in cln and cln["Compound"].notna().any():
        modes = cln["Compound"].mode()
        compound_mode = str(modes.iloc[0]) if not modes.empty else None

    return {
        "best_lap_ms": best_ms,
        "median_lap_ms": ms(cln["LapTime"]) if not cln.empty else None,
        "long_run_ms": long_run_ms,
        "n_laps": len(all_d),
        "n_clean_laps": len(cln),
        "compound_mode": compound_mode,
    }


def _settled(season: int, scope: str) -> bool:
    """Already fetched, or known not to exist. Never true for a transient failure.

    For the live season an empty session may just not have run yet, so it isn't
    settled either; otherwise one mid-season ingest would leave practice pace
    missing for every remaining round.
    """
    with connect(read_only=True) as con:
        row = con.execute(
            "SELECT status, detail FROM ingest_log WHERE source = 'fastf1' AND scope = ?", [scope]
        ).fetchone()
    if row is None:
        return False
    status, detail = row
    if status == "ok":
        return True
    absent = status == "empty" and any(k in (detail or "") for k in _ABSENT)
    return absent and season != config.CURRENT_SEASON


def ingest_session(fastf1: Any, season: int, rnd: int, code: str, force: bool = False) -> int:
    scope = f"{season}:{rnd}:{code}"
    if not force and _settled(season, scope):
        return 0

    try:
        session = fastf1.get_session(season, rnd, code)
    except ValueError as exc:  # "Session type 'FP3' does not exist for this event"
        with connect() as con:
            log_ingest(con, "fastf1", scope, "empty", f"unavailable: {exc!r}"[:400])
        return 0
    except Exception as exc:  # noqa: BLE001 - schedule lookup failed; try again next run
        with connect() as con:
            log_ingest(con, "fastf1", scope, "failed", repr(exc)[:400])
        return 0

    try:
        # Lap timing only: no telemetry, weather or messages. Only per-driver
        # aggregates are kept; the laps themselves stay in FastF1's cache.
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as exc:  # noqa: BLE001 - rate limit or network: retryable
        with connect() as con:
            log_ingest(con, "fastf1", scope, "failed", repr(exc)[:400])
        log.warning("fastf1 %s failed (will retry next run): %s", scope, exc)
        return 0

    try:
        laps = session.laps
    except Exception:  # noqa: BLE001 - a session not yet run "loads" but has no laps
        laps = None
    if laps is None or laps.empty:
        with connect() as con:
            log_ingest(con, "fastf1", scope, "empty", "no laps")
        return 0

    code_map = _code_to_driver_id(season)
    clean = _clean_laps(laps)
    start_utc = getattr(session, "date", None)

    rows = []
    for abbrev in sorted(laps["Driver"].dropna().unique()):
        driver_id = code_map.get(abbrev)
        if driver_id is None:
            # Unmapped abbreviation: usually a mid-season reserve driver whose
            # jolpica entry lands later. Skipped rather than guessed at.
            log.debug("no driver_id for %s in %d", abbrev, season)
            continue
        pace = _driver_pace(laps, clean, abbrev)
        rows.append(
            {
                "season": season,
                "round": rnd,
                "session": code,
                "driver_id": driver_id,
                "session_start_utc": start_utc,
                **pace,
            }
        )

    df = pd.DataFrame(rows)

    with connect() as con:
        n = upsert(con, "raw_session_pace", df, ["season", "round", "session", "driver_id"])
        log_ingest(con, "fastf1", scope, "ok" if n else "empty", f"{n} drivers" if n else "no laps mapped")

    return n


def ingest(seasons: list[int], force: bool = False, rounds: list[int] | None = None) -> None:
    fastf1 = _setup_fastf1()

    for season in seasons:
        with connect(read_only=True) as con:
            scheduled = [
                r[0]
                for r in con.execute(
                    "SELECT round FROM raw_races WHERE season = ? ORDER BY round", [season]
                ).fetchall()
            ]
        if not scheduled:
            log.warning("season %d has no races - run ingest-jolpica first", season)
            continue

        for rnd in scheduled:
            if rounds is not None and rnd not in rounds:
                continue
            got = 0
            for code in SESSION_CODES:
                got += ingest_session(fastf1, season, rnd, code, force=force)
            log.info("fastf1 %d r%-2d: %d driver-sessions", season, rnd, got)


def ingest_next_weekend() -> None:
    """Practice pace for the race about to run: the sessions held so far.

    What the live forecast needs from FastF1, at a few calls a session rather
    than a season's worth. A session that hasn't run yet comes back empty and,
    this being the live season, is tried again on the next run.
    """
    now = datetime.now(UTC).replace(tzinfo=None)
    with connect(read_only=True) as con:
        row = con.execute(
            """
            SELECT season, round FROM raw_races
            WHERE race_start_utc > ?::TIMESTAMP ORDER BY race_start_utc LIMIT 1
            """,
            [now],
        ).fetchone()
    if row is None:
        log.info("fastf1: no race scheduled")
        return
    ingest([int(row[0])], rounds=[int(row[1])])
