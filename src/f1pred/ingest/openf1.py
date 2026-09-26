"""Official starting grids and weekend entry lists from OpenF1 (2023 onward).

jolpica is the primary source. OpenF1 fills two gaps in it:

  grid      jolpica's results carry the starting grid, but results only exist
            after the race, and they have arrived with the grid blank. The
            official grid - qualifying order with penalties applied - is
            published by OpenF1 against the qualifying session.
  entries   nothing in jolpica says who is driving this weekend until
            qualifying is published. OpenF1 lists the drivers in every session,
            so a replacement or a returning driver shows up from Friday.

Rows are stored as fetched, mapped onto jolpica's driver_id and constructor_id.
Nothing is invented: a driver or team that can't be matched is logged and
left out. weekend.py decides whether what is stored can be trusted.
"""

from __future__ import annotations

import logging
import unicodedata
from datetime import UTC, datetime, timedelta

import pandas as pd

from .. import config
from ..http_cache import RateLimitedSession
from ..store import connect, log_ingest, upsert

log = logging.getLogger(__name__)

SESSION_CODES = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Sprint Shootout": "SQ",  # 2023 name for sprint qualifying
    "Sprint Qualifying": "SQ",
    "Sprint": "S",
    "Qualifying": "Q",
    "Race": "R",
}

# A weekend's sessions all fall within this many days before the race.
WEEKEND_DAYS = 4


def http() -> RateLimitedSession:
    return RateLimitedSession(
        cache_dir=config.HTTP_CACHE / "openf1",
        min_interval=config.OPENF1_MIN_INTERVAL,
        hourly_limit=config.OPENF1_HOURLY_LIMIT,
    )


def _utc(value: str | None) -> datetime | None:
    """ISO timestamp with offset -> naive UTC, the convention every table uses."""
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime()


def _fold(text: str | None) -> str:
    """Lower-case, accents stripped: 'Pérez' and 'PEREZ' are the same name."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower().strip()


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
def sessions(client: RateLimitedSession, year: int, refresh: bool = False) -> pd.DataFrame:
    rows = client.get_json(f"{config.OPENF1_BASE}/sessions?year={year}", refresh=refresh, not_found=[])
    df = pd.DataFrame(rows or [])
    if df.empty:
        return df
    df["code"] = df["session_name"].map(SESSION_CODES)
    df["start_utc"] = [_utc(v) for v in df["date_start"]]
    if "is_cancelled" in df:
        df = df[~df["is_cancelled"].fillna(False).astype(bool)]
    return df.dropna(subset=["code", "start_utc"]).reset_index(drop=True)


def weekend_sessions(all_sessions: pd.DataFrame, race_start_utc: datetime) -> pd.DataFrame:
    """The sessions of the meeting whose race starts at jolpica's race start.

    Matched on the race session's start time rather than on names, which the two
    sources spell differently. Six hours of tolerance covers a late schedule
    change; the next meeting is at least a week away, so it can't be mistaken
    for this one.
    """
    if all_sessions.empty or race_start_utc is None or pd.isna(race_start_utc):
        return all_sessions.iloc[0:0]
    start = pd.Timestamp(race_start_utc)
    races = all_sessions[all_sessions["code"] == "R"]
    gap = (pd.to_datetime(races["start_utc"]) - start).abs()
    close = races[gap <= pd.Timedelta(hours=6)]
    if close.empty:
        return all_sessions.iloc[0:0]
    meeting = close.iloc[0]["meeting_key"]
    out = all_sessions[all_sessions["meeting_key"] == meeting]
    # Belt and braces: every session of the weekend precedes the race by days, not weeks.
    lo = start - pd.Timedelta(days=WEEKEND_DAYS)
    return out[(pd.to_datetime(out["start_utc"]) >= lo) & (pd.to_datetime(out["start_utc"]) <= start)]


# ---------------------------------------------------------------------------
# Mapping onto jolpica identities
# ---------------------------------------------------------------------------
def driver_lookup(season: int) -> pd.DataFrame:
    """Every driver jolpica knows, most recent season first.

    Includes earlier seasons so a driver returning after a year out still maps.
    """
    with connect(read_only=True) as con:
        return con.execute(
            """
            SELECT driver_id, code, permanent_number, family_name, max(season) AS season
            FROM raw_drivers WHERE season <= ?
            GROUP BY ALL ORDER BY season DESC
            """,
            [season],
        ).fetchdf()


def map_driver(entry: dict, lookup: pd.DataFrame, season: int) -> str | None:
    """OpenF1 driver -> jolpica driver_id, by code, then number, then surname.

    Each step must identify exactly one driver in the most recent season it
    matches, otherwise the next is tried. Numbers are not unique over time and
    codes have been reused, so the current season is preferred throughout.
    """
    if lookup.empty:
        return None
    this_season = lookup[lookup["season"] == season]
    candidates = [
        ("code", str(entry.get("name_acronym") or "").upper()),
        ("permanent_number", entry.get("driver_number")),
        ("family_name", _fold(entry.get("last_name"))),
    ]
    for pool in (this_season, lookup):
        for column, value in candidates:
            if value in (None, "", 0):
                continue
            if column == "family_name":
                hits = pool[pool["family_name"].map(_fold) == value]
            elif column == "code":
                hits = pool[pool["code"].fillna("").str.upper() == value]
            else:
                hits = pool[pool["permanent_number"] == value]
            ids = hits["driver_id"].unique()
            if len(ids) == 1:
                return str(ids[0])
    return None


def team_lookup(season: int) -> dict[str, str]:
    """driver_id -> the constructor they drove for most recently this season."""
    with connect(read_only=True) as con:
        rows = con.execute(
            """
            SELECT driver_id, arg_max(constructor_id, round) AS constructor_id
            FROM (
                SELECT season, round, driver_id, constructor_id FROM raw_results
                UNION ALL
                SELECT season, round, driver_id, constructor_id FROM raw_qualifying
            )
            WHERE season = ? AND constructor_id IS NOT NULL
            GROUP BY driver_id
            """,
            [season],
        ).fetchall()
    return {d: c for d, c in rows}


def map_teams(entries: pd.DataFrame, current_team: dict[str, str]) -> pd.Series:
    """OpenF1 team names -> constructor_id, learned from the drivers who map.

    A team name takes the constructor most of its known drivers raced for this
    season, so a reserve in a Red Bull seat maps to red_bull through the
    teammate. Nothing is hard-coded, so a renamed team still maps as long as one
    of its drivers has raced for it this season.
    """
    known = entries.assign(constructor_id=entries["driver_id"].map(current_team)).dropna(
        subset=["constructor_id", "team_name"]
    )
    if known.empty:
        return pd.Series(dtype=object)
    votes = known.groupby(["team_name", "constructor_id"]).size().reset_index(name="n")
    best = votes.sort_values("n", ascending=False).drop_duplicates("team_name")
    return best.set_index("team_name")["constructor_id"]


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch_session_drivers(
    client: RateLimitedSession, session: pd.Series, season: int, lookup: pd.DataFrame, refresh: bool
) -> pd.DataFrame:
    rows = client.get_json(
        f"{config.OPENF1_BASE}/drivers?session_key={int(session['session_key'])}",
        refresh=refresh,
        not_found=[],
    )
    out = []
    for r in rows or []:
        driver_id = map_driver(r, lookup, season)
        if driver_id is None:
            log.warning(
                "openf1: no jolpica driver for #%s %s (%s) - left out",
                r.get("driver_number"),
                r.get("name_acronym"),
                r.get("team_name"),
            )
            continue
        out.append(
            {
                "driver_id": driver_id,
                "team_name": r.get("team_name"),
                "driver_number": r.get("driver_number"),
            }
        )
    return pd.DataFrame(out, columns=["driver_id", "team_name", "driver_number"]).drop_duplicates("driver_id")


def fetch_grid(
    client: RateLimitedSession,
    weekend: pd.DataFrame,
    season: int,
    rnd: int,
    lookup: pd.DataFrame,
    refresh: bool,
) -> pd.DataFrame:
    """The official starting grid, published against the qualifying session."""
    quali = weekend[weekend["code"] == "Q"]
    if quali.empty:
        return pd.DataFrame()
    q = quali.iloc[0]
    rows = client.get_json(
        f"{config.OPENF1_BASE}/starting_grid?session_key={int(q['session_key'])}",
        refresh=refresh,
        not_found=[],
    )
    if not rows:
        return pd.DataFrame()

    drivers = fetch_session_drivers(client, q, season, lookup, refresh)
    by_number = dict(zip(drivers["driver_number"], drivers["driver_id"]))
    now = datetime.now(UTC).replace(tzinfo=None)
    out = []
    for r in rows:
        driver_id = by_number.get(r.get("driver_number"))
        if driver_id is None:
            log.warning("openf1 grid %d r%d: car #%s not mapped", season, rnd, r.get("driver_number"))
            continue
        out.append(
            {
                "season": season,
                "round": rnd,
                "driver_id": driver_id,
                "position": r.get("position"),
                "driver_number": r.get("driver_number"),
                "session_key": int(q["session_key"]),
                "session_start_utc": q["start_utc"],
                "fetched_at_utc": now,
            }
        )
    return pd.DataFrame(out)


def fetch_entries(
    client: RateLimitedSession,
    weekend: pd.DataFrame,
    season: int,
    rnd: int,
    lookup: pd.DataFrame,
    refresh: bool,
) -> pd.DataFrame:
    """Drivers in each session of the weekend that has started."""
    now = datetime.now(UTC).replace(tzinfo=None)
    current_team = team_lookup(season)
    frames = []
    for _, s in weekend.iterrows():
        if s["code"] == "R" or pd.Timestamp(s["start_utc"]) > pd.Timestamp(now):
            continue
        drivers = fetch_session_drivers(client, s, season, lookup, refresh)
        if drivers.empty:
            continue
        drivers["session"] = s["code"]
        drivers["session_start_utc"] = s["start_utc"]
        frames.append(drivers)
    if not frames:
        return pd.DataFrame()

    entries = pd.concat(frames, ignore_index=True)
    teams = map_teams(entries, current_team)
    entries["constructor_id"] = entries["team_name"].map(teams)
    unmapped = entries[entries["constructor_id"].isna()]["team_name"].dropna().unique()
    if len(unmapped):
        log.warning("openf1 %d r%d: team(s) %s not matched to a constructor", season, rnd, list(unmapped))
    entries["season"], entries["round"], entries["fetched_at_utc"] = season, rnd, now
    return entries


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
def _races_needing_a_grid(season: int, horizon_days: float) -> pd.DataFrame:
    """Completed races whose jolpica grid is missing, plus races about to run."""
    now = datetime.now(UTC).replace(tzinfo=None)
    with connect(read_only=True) as con:
        return con.execute(
            """
            SELECT ra.season, ra.round, ra.race_start_utc,
                   count(r.driver_id) AS n_results,
                   count(r.grid) AS n_grid,
                   (SELECT count(*) FROM raw_openf1_grid g
                     WHERE g.season = ra.season AND g.round = ra.round) AS n_openf1
            FROM raw_races ra
            LEFT JOIN raw_results r USING (season, round)
            WHERE ra.season = ? AND ra.race_start_utc IS NOT NULL
              AND ra.race_start_utc <= ?::TIMESTAMP + to_days(?)
            GROUP BY ALL
            ORDER BY ra.round
            """,
            [season, now, int(horizon_days)],
        ).fetchdf()


def ingest(seasons: list[int], horizon_days: float = 7.0, force: bool = False) -> dict[str, int]:
    """Fetch grids for races that lack one and entries for the race about to run.

    Past weekends are served from the response cache; the weekend under way is
    always re-fetched, because its grid changes when penalties are applied and
    its entry list grows with every session.
    """
    client = http()
    counts = {"grid": 0, "entries": 0}
    now = datetime.now(UTC).replace(tzinfo=None)

    for season in seasons:
        if season < config.OPENF1_FIRST_SEASON:
            continue
        races = _races_needing_a_grid(season, horizon_days)
        if races.empty:
            continue
        upcoming = races[pd.to_datetime(races["race_start_utc"]) > pd.Timestamp(now) - timedelta(hours=6)]
        missing = races[(races["n_results"] > 0) & (races["n_grid"] < races["n_results"])]
        targets = pd.concat([missing, upcoming.head(1)]).drop_duplicates("round")
        if not force:
            # A completed race with an OpenF1 grid already stored is settled.
            settled = (targets["n_results"] > 0) & (targets["n_openf1"] > 0)
            targets = targets[~settled | targets["round"].isin(upcoming["round"].head(1))]
        if targets.empty:
            continue

        try:
            live = not upcoming.empty
            all_sessions = sessions(client, season, refresh=live or force)
            lookup = driver_lookup(season)
        except Exception as exc:  # noqa: BLE001 - a fallback source must never stop the run
            log.warning("openf1 %d: sessions unavailable (%s)", season, exc)
            continue

        for race in targets.itertuples():
            rnd = int(race.round)
            is_upcoming = rnd in set(upcoming["round"])
            scope = f"{season}:{rnd}"
            try:
                weekend = weekend_sessions(all_sessions, race.race_start_utc)
                if weekend.empty:
                    with connect() as con:
                        log_ingest(con, "openf1", scope, "empty", "no matching meeting")
                    continue
                refresh = is_upcoming or force
                grid = fetch_grid(client, weekend, season, rnd, lookup, refresh)
                entries = (
                    fetch_entries(client, weekend, season, rnd, lookup, refresh)
                    if is_upcoming
                    else pd.DataFrame()
                )
                with connect() as con:
                    if not grid.empty:
                        # Replace, not merge: a grid re-published after penalties must
                        # not keep rows from the earlier version.
                        con.execute(
                            "DELETE FROM raw_openf1_grid WHERE season = ? AND round = ?", [season, rnd]
                        )
                        counts["grid"] += upsert(
                            con, "raw_openf1_grid", grid, ["season", "round", "driver_id"]
                        )
                    if not entries.empty:
                        cols = [
                            "season",
                            "round",
                            "session",
                            "driver_id",
                            "constructor_id",
                            "team_name",
                            "driver_number",
                            "session_start_utc",
                            "fetched_at_utc",
                        ]
                        counts["entries"] += upsert(
                            con,
                            "raw_openf1_entries",
                            entries[cols],
                            ["season", "round", "session", "driver_id"],
                        )
                    log_ingest(con, "openf1", scope, "ok", f"grid {len(grid)}, entries {len(entries)}")
                log.info(
                    "openf1 %d r%-2d: grid %d rows, entries %d rows", season, rnd, len(grid), len(entries)
                )
            except Exception as exc:  # noqa: BLE001
                with connect() as con:
                    log_ingest(con, "openf1", scope, "failed", repr(exc)[:400])
                log.warning("openf1 %d r%d failed: %s", season, rnd, exc)
    return counts
