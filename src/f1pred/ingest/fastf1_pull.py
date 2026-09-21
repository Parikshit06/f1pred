"""Ingest practice/qualifying/race pace and session weather via FastF1.

This is the slow part of the pipeline: FastF1 downloads and caches the full
timing archive per session (a few hundred MB per season). It is resumable -
every session we finish is recorded in ingest_log and skipped next run.

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
from typing import Any

import pandas as pd

from .. import config
from ..store import already_ingested, connect, log_ingest, upsert

log = logging.getLogger(__name__)

# Attempted in order; sessions that do not exist for a given weekend are skipped.
SESSION_CODES = ["FP1", "FP2", "FP3", "SQ", "S", "Q", "R"]

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


def _weather_row(session: Any, season: int, rnd: int, code: str) -> dict[str, Any] | None:
    try:
        wx = session.weather_data
    except Exception:  # noqa: BLE001
        return None
    if wx is None or wx.empty:
        return None
    return {
        "season": season,
        "round": rnd,
        "session": code,
        "air_temp_c": float(wx["AirTemp"].mean()) if "AirTemp" in wx else None,
        "track_temp_c": float(wx["TrackTemp"].mean()) if "TrackTemp" in wx else None,
        "humidity_pct": float(wx["Humidity"].mean()) if "Humidity" in wx else None,
        "wind_speed_ms": float(wx["WindSpeed"].mean()) if "WindSpeed" in wx else None,
        "rainfall": bool(wx["Rainfall"].any()) if "Rainfall" in wx else False,
    }


def ingest_session(fastf1: Any, season: int, rnd: int, code: str, force: bool = False) -> int:
    scope = f"{season}:{rnd}:{code}"

    with connect() as con:
        if not force and already_ingested(con, "fastf1", scope):
            return 0

    try:
        session = fastf1.get_session(season, rnd, code)
        session.load(laps=True, telemetry=False, weather=True, messages=False)
    except Exception as exc:  # noqa: BLE001 - session genuinely may not exist
        with connect() as con:
            log_ingest(con, "fastf1", scope, "empty", f"unavailable: {exc!r}"[:400])
        return 0

    laps = session.laps
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
    wx_row = _weather_row(session, season, rnd, code)

    with connect() as con:
        n = upsert(con, "raw_session_pace", df, ["season", "round", "session", "driver_id"])
        if wx_row:
            upsert(con, "raw_session_weather", pd.DataFrame([wx_row]), ["season", "round", "session"])
        log_ingest(con, "fastf1", scope, "ok" if n else "empty", f"{n} drivers")

    return n


def ingest(seasons: list[int], force: bool = False) -> None:
    fastf1 = _setup_fastf1()

    for season in seasons:
        with connect(read_only=True) as con:
            rounds = [
                r[0]
                for r in con.execute(
                    "SELECT round FROM raw_races WHERE season = ? ORDER BY round", [season]
                ).fetchall()
            ]
        if not rounds:
            log.warning("season %d has no races - run ingest-jolpica first", season)
            continue

        for rnd in rounds:
            got = 0
            for code in SESSION_CODES:
                got += ingest_session(fastf1, season, rnd, code, force=force)
            log.info("fastf1 %d r%-2d: %d driver-sessions", season, rnd, got)
