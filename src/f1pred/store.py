"""DuckDB schema and load helpers.

One file, real SQL, no server. Every table is rebuildable from the HTTP cache,
so the database is disposable - that is why it is gitignored.

Naming rule: raw_* tables are a faithful copy of what the source gave us.
Derived tables (features, predictions) never overwrite raw_*.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import duckdb
import pandas as pd

from . import config

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_races (
    season          INTEGER NOT NULL,
    round           INTEGER NOT NULL,
    race_name       VARCHAR,
    circuit_id      VARCHAR,
    circuit_name    VARCHAR,
    locality        VARCHAR,
    country         VARCHAR,
    lat             DOUBLE,
    lon             DOUBLE,
    race_date       DATE,
    race_time       VARCHAR,
    race_start_utc  TIMESTAMP,
    PRIMARY KEY (season, round)
);

-- Needed to join FastF1 (which speaks 3-letter codes and car numbers) to
-- jolpica (which speaks driverId). Without this the two sources cannot meet.
CREATE TABLE IF NOT EXISTS raw_drivers (
    season          INTEGER NOT NULL,
    driver_id       VARCHAR NOT NULL,
    code            VARCHAR,
    permanent_number INTEGER,
    given_name      VARCHAR,
    family_name     VARCHAR,
    dob             DATE,
    nationality     VARCHAR,
    PRIMARY KEY (season, driver_id)
);

CREATE TABLE IF NOT EXISTS raw_results (
    season          INTEGER NOT NULL,
    round           INTEGER NOT NULL,
    driver_id       VARCHAR NOT NULL,
    constructor_id  VARCHAR,
    grid            INTEGER,
    position        INTEGER,          -- classification order, retirees included
    classified      BOOLEAN,          -- positionText was numeric
    position_text   VARCHAR,
    points          DOUBLE,
    laps            INTEGER,
    status          VARCHAR,
    finished        BOOLEAN,          -- running at the end
    dnf             BOOLEAN,
    millis          BIGINT,
    fastest_lap_rank INTEGER,
    PRIMARY KEY (season, round, driver_id)
);

CREATE TABLE IF NOT EXISTS raw_qualifying (
    season          INTEGER NOT NULL,
    round           INTEGER NOT NULL,
    driver_id       VARCHAR NOT NULL,
    constructor_id  VARCHAR,
    position        INTEGER,
    q1_ms           BIGINT,
    q2_ms           BIGINT,
    q3_ms           BIGINT,
    best_ms         BIGINT,
    PRIMARY KEY (season, round, driver_id)
);

CREATE TABLE IF NOT EXISTS raw_sprint (
    season          INTEGER NOT NULL,
    round           INTEGER NOT NULL,
    driver_id       VARCHAR NOT NULL,
    constructor_id  VARCHAR,
    grid            INTEGER,
    position        INTEGER,
    points          DOUBLE,
    status          VARCHAR,
    PRIMARY KEY (season, round, driver_id)
);

CREATE TABLE IF NOT EXISTS raw_pitstops (
    season          INTEGER NOT NULL,
    round           INTEGER NOT NULL,
    driver_id       VARCHAR NOT NULL,
    stop            INTEGER NOT NULL,
    lap             INTEGER,
    duration_s      DOUBLE,
    PRIMARY KEY (season, round, driver_id, stop)
);

CREATE TABLE IF NOT EXISTS raw_standings (
    season          INTEGER NOT NULL,
    round           INTEGER NOT NULL,   -- standings AFTER this round
    driver_id       VARCHAR NOT NULL,
    constructor_id  VARCHAR,
    position        INTEGER,
    points          DOUBLE,
    wins            INTEGER,
    PRIMARY KEY (season, round, driver_id)
);

-- One row per driver per practice/qualifying session, from FastF1.
CREATE TABLE IF NOT EXISTS raw_session_pace (
    season          INTEGER NOT NULL,
    round           INTEGER NOT NULL,
    session         VARCHAR NOT NULL,   -- FP1 FP2 FP3 Q SQ S R
    driver_id       VARCHAR NOT NULL,
    best_lap_ms     BIGINT,
    -- median of clean green-flag laps: a far better pace proxy than the single
    -- best lap, because it is not dominated by one low-fuel banzai run
    median_lap_ms   BIGINT,
    long_run_ms     BIGINT,             -- median of stints >= 5 laps
    n_laps          INTEGER,
    n_clean_laps    INTEGER,
    compound_mode   VARCHAR,
    session_start_utc TIMESTAMP,
    PRIMARY KEY (season, round, session, driver_id)
);

CREATE TABLE IF NOT EXISTS raw_session_weather (
    season          INTEGER NOT NULL,
    round           INTEGER NOT NULL,
    session         VARCHAR NOT NULL,
    air_temp_c      DOUBLE,
    track_temp_c    DOUBLE,
    humidity_pct    DOUBLE,
    wind_speed_ms   DOUBLE,
    rainfall        BOOLEAN,
    PRIMARY KEY (season, round, session)
);

CREATE TABLE IF NOT EXISTS raw_forecast (
    season          INTEGER NOT NULL,
    round           INTEGER NOT NULL,
    session         VARCHAR NOT NULL,
    fetched_at_utc  TIMESTAMP NOT NULL,  -- as_of stamp: critical for leakage checks
    valid_at_utc    TIMESTAMP,
    temp_c          DOUBLE,
    precip_mm       DOUBLE,
    precip_prob_pct DOUBLE,
    wind_speed_ms   DOUBLE,
    cloud_cover_pct DOUBLE,
    PRIMARY KEY (season, round, session, fetched_at_utc)
);

CREATE TABLE IF NOT EXISTS ingest_log (
    source          VARCHAR NOT NULL,
    scope           VARCHAR NOT NULL,   -- e.g. "2024" or "2024:12:FP2"
    status          VARCHAR NOT NULL,   -- ok | empty | failed
    detail          VARCHAR,
    ingested_at_utc TIMESTAMP NOT NULL,
    PRIMARY KEY (source, scope)
);
"""


def database_exists() -> bool:
    """Whether there is anything to read yet.

    A read-only connect to a missing file raises rather than creating one. That
    is right for the pipeline - `build-features` on a machine with no data
    should fail loudly and say so - but wrong for the two readers that only
    decorate a rendered page: the track record and the driver-surname lookup.
    Those have a sensible empty answer, and a fresh clone has no database
    because `data/*.duckdb` is rebuildable and therefore not committed.

    This is what CI caught on its first run: three tests rendered a page, the
    page reached for the database, and there was none.
    """
    return config.DB_PATH.exists()


@contextmanager
def connect(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    con = duckdb.connect(str(config.DB_PATH), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def init_db() -> None:
    with connect() as con:
        con.execute(SCHEMA)
    log.info("Schema ready at %s", config.DB_PATH)


def upsert(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, keys: list[str]) -> int:
    """Insert rows, replacing any existing row with the same primary key.

    DuckDB has no MERGE, so we delete-then-insert inside one transaction.
    Idempotent: running the same ingest twice leaves the table unchanged.
    """
    if df.empty:
        return 0

    cols = [c for c in df.columns]
    con.register("_incoming", df)
    try:
        con.execute("BEGIN TRANSACTION")
        join = " AND ".join(f"t.{k} = i.{k}" for k in keys)
        con.execute(f"DELETE FROM {table} t WHERE EXISTS (SELECT 1 FROM _incoming i WHERE {join})")
        con.execute(f"INSERT INTO {table} ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _incoming")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister("_incoming")

    return len(df)


def log_ingest(
    con: duckdb.DuckDBPyConnection, source: str, scope: str, status: str, detail: str = ""
) -> None:
    con.execute(
        """
        DELETE FROM ingest_log WHERE source = ? AND scope = ?;
        """,
        [source, scope],
    )
    con.execute(
        """
        INSERT INTO ingest_log (source, scope, status, detail, ingested_at_utc)
        VALUES (?, ?, ?, ?, now()::TIMESTAMP)
        """,
        [source, scope, status, detail],
    )


def ingest_status(con: duckdb.DuckDBPyConnection, source: str, scope: str) -> str | None:
    """What the last attempt at this scope recorded, or None if never tried.

    Callers need the distinction between "ok" and "empty": an empty response
    for a round of the live season usually means the round has not run yet,
    which is a reason to come back rather than a reason to stop asking.
    """
    row = con.execute(
        "SELECT status FROM ingest_log WHERE source = ? AND scope = ?", [source, scope]
    ).fetchone()
    return row[0] if row else None


def already_ingested(con: duckdb.DuckDBPyConnection, source: str, scope: str) -> bool:
    return ingest_status(con, source, scope) in ("ok", "empty")


def table_counts() -> pd.DataFrame:
    with connect(read_only=True) as con:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        rows = [(t, con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]) for t in tables]
    return pd.DataFrame(rows, columns=["table", "rows"]).sort_values("table").reset_index(drop=True)


def repair_status_flags() -> pd.DataFrame:
    """Recompute finished/dnf from the stored status text.

    The status vocabulary upstream changed in 2023 and the parser's rule went
    stale, so rows already in the database carry the wrong flag. The raw status
    text was stored, which means this is repairable in place - nine seasons do
    not need re-downloading over a 450-request hourly budget.

    Idempotent. Run it after any change to the finished/retired rule; the
    ingest path applies the same rule to new rows.
    """
    from .ingest.jolpica import is_finished

    with connect() as con:
        rows = con.execute("SELECT season, round, driver_id, status, dnf FROM raw_results").fetchdf()
        if rows.empty:
            return pd.DataFrame()

        rows["should_be_dnf"] = ~rows["status"].map(is_finished)
        wrong = rows[rows["dnf"].astype(bool) != rows["should_be_dnf"]]
        if wrong.empty:
            log.info("status flags already consistent (%d rows checked)", len(rows))
            return pd.DataFrame()

        con.execute("CREATE OR REPLACE TEMP TABLE _fix AS SELECT * FROM wrong")
        con.execute(
            """
            UPDATE raw_results AS r
            SET dnf = f.should_be_dnf, finished = NOT f.should_be_dnf
            FROM _fix AS f
            WHERE r.season = f.season AND r.round = f.round AND r.driver_id = f.driver_id
            """
        )
        log.warning("repaired %d result rows whose finished/retired flag was wrong", len(wrong))
        return (
            wrong.groupby(["status", "dnf"])
            .size()
            .reset_index(name="rows")
            .sort_values("rows", ascending=False)
        )
