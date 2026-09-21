"""Race-weekend weather from Open-Meteo (free, no API key).

Two modes, and the distinction matters for honesty:

  forecast  what we could actually have known BEFORE the session. Stamped with
            fetched_at_utc so the leakage guard can prove the forecast predates
            the session it describes.
  archive   what actually happened. Used for building historical features only,
            never for a live prediction.

FastF1 already gives us observed trackside weather for 2018+, so archive is a
fallback for sessions FastF1 has no data for.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from .. import config
from ..http_cache import RateLimitedSession
from ..store import connect, upsert

log = logging.getLogger(__name__)

_HOURLY = "temperature_2m,precipitation,precipitation_probability,wind_speed_10m,cloud_cover"


def _nearest_hour_row(hourly: dict, target: datetime) -> dict | None:
    times = hourly.get("time", [])
    if not times:
        return None

    stamps = [datetime.fromisoformat(t) for t in times]
    idx = min(range(len(stamps)), key=lambda i: abs((stamps[i] - target).total_seconds()))

    # More than 3h from the session start is not a useful forecast for it.
    if abs((stamps[idx] - target).total_seconds()) > 3 * 3600:
        return None

    def at(key: str) -> float | None:
        seq = hourly.get(key) or []
        value = seq[idx] if idx < len(seq) else None
        return float(value) if value is not None else None

    return {
        "valid_at_utc": stamps[idx],
        "temp_c": at("temperature_2m"),
        "precip_mm": at("precipitation"),
        "precip_prob_pct": at("precipitation_probability"),
        # Open-Meteo reports km/h by default; store SI to match FastF1.
        "wind_speed_ms": (at("wind_speed_10m") / 3.6) if at("wind_speed_10m") is not None else None,
        "cloud_cover_pct": at("cloud_cover"),
    }


def fetch_forecast(
    session_http: RateLimitedSession,
    season: int,
    rnd: int,
    session_code: str,
    lat: float,
    lon: float,
    session_start_utc: datetime,
) -> pd.DataFrame:
    """Forecast for one session. Safe to call repeatedly - each call is stamped
    with its own fetched_at_utc, so we keep a history of how the forecast moved."""
    days = max(1, min(16, (session_start_utc.date() - datetime.now(timezone.utc).date()).days + 2))
    url = (
        f"{config.OPEN_METEO_BASE}?latitude={lat:.4f}&longitude={lon:.4f}"
        f"&hourly={_HOURLY}&forecast_days={days}&timezone=UTC"
    )
    payload = session_http.get_json(url)
    row = _nearest_hour_row(payload.get("hourly", {}), session_start_utc)
    if row is None:
        return pd.DataFrame()

    row.update(
        {
            "season": season,
            "round": rnd,
            "session": session_code,
            "fetched_at_utc": datetime.now(timezone.utc).replace(tzinfo=None),
        }
    )
    return pd.DataFrame([row])


def fetch_archive(
    session_http: RateLimitedSession,
    season: int,
    rnd: int,
    session_code: str,
    lat: float,
    lon: float,
    session_start_utc: datetime,
) -> pd.DataFrame:
    day = session_start_utc.date().isoformat()
    url = (
        f"{config.OPEN_METEO_ARCHIVE}?latitude={lat:.4f}&longitude={lon:.4f}"
        f"&start_date={day}&end_date={day}&hourly={_HOURLY}&timezone=UTC"
    )
    payload = session_http.get_json(url)
    row = _nearest_hour_row(payload.get("hourly", {}), session_start_utc)
    if row is None:
        return pd.DataFrame()

    row.update(
        {
            "season": season,
            "round": rnd,
            "session": session_code,
            # Archive data is observed, not forecast. Stamping it with the
            # session time (not now) stops it ever passing as a valid
            # pre-session forecast in the leakage guard.
            "fetched_at_utc": session_start_utc,
        }
    )
    return pd.DataFrame([row])


def ingest_upcoming(days_ahead: int = 10) -> int:
    """Forecast every session of every race starting within the next N days."""
    http = RateLimitedSession(min_interval=0.2)
    horizon = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=days_ahead)

    with connect(read_only=True) as con:
        races = con.execute(
            """
            SELECT season, round, lat, lon, race_start_utc
            FROM raw_races
            WHERE race_start_utc IS NOT NULL
              AND race_start_utc >= now()::TIMESTAMP
              AND race_start_utc <= ?
            ORDER BY race_start_utc
            """,
            [horizon],
        ).fetchall()

    total = 0
    for season, rnd, lat, lon, start in races:
        if lat is None or lon is None:
            continue
        df = fetch_forecast(http, season, rnd, "R", lat, lon, start)
        if df.empty:
            continue
        with connect() as con:
            total += upsert(con, "raw_forecast", df, ["season", "round", "session", "fetched_at_utc"])
        log.info("forecast %d r%d: %s", season, rnd, df.iloc[0]["valid_at_utc"])

    return total
