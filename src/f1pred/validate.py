"""Data validation.

Runs a fixed set of checks over the raw tables and reports anything that would
quietly corrupt a model: missing rounds, duplicate entries, impossible
positions, orphaned foreign keys, results that contradict each other.

Severity:
    ERROR  the pipeline should not proceed
    WARN   worth knowing, usually a real quirk of F1 rather than a bug

Exit code is non-zero if any ERROR fires, so this is usable in CI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import duckdb
import pandas as pd

from .store import connect

log = logging.getLogger(__name__)

ERROR = "ERROR"
WARN = "WARN"


@dataclass
class Finding:
    check: str
    severity: str
    message: str
    sample: pd.DataFrame | None = None


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARN]

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self, show_samples: bool = True) -> str:
        if not self.findings:
            return "All checks passed."
        lines = []
        for f in sorted(self.findings, key=lambda x: (x.severity != ERROR, x.check)):
            lines.append(f"[{f.severity:5s}] {f.check}: {f.message}")
            if show_samples and f.sample is not None and not f.sample.empty:
                for line in f.sample.head(5).to_string(index=False).splitlines():
                    lines.append(f"          {line}")
        lines.append("")
        lines.append(f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)")
        return "\n".join(lines)


Check = Callable[[duckdb.DuckDBPyConnection], list[Finding]]
_CHECKS: list[Check] = []


def check(fn: Check) -> Check:
    _CHECKS.append(fn)
    return fn


def _q(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pd.DataFrame:
    return con.execute(sql, params or []).fetchdf()


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
@check
def tables_populated(con) -> list[Finding]:
    out = []
    required = ["raw_races", "raw_results", "raw_qualifying", "raw_drivers"]
    for table in required:
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        if n == 0:
            out.append(Finding("tables_populated", ERROR, f"{table} is empty"))
    return out


@check
def rounds_contiguous(con) -> list[Finding]:
    """Round numbers within a season must run 1..N with no holes."""
    df = _q(
        con,
        """
        SELECT season, min(round) lo, max(round) hi,
               count(DISTINCT round) n_rounds
        FROM raw_races GROUP BY season ORDER BY season
        """,
    )
    bad = df[(df.lo != 1) | (df.hi != df.n_rounds)]
    if bad.empty:
        return []
    return [
        Finding(
            "rounds_contiguous",
            ERROR,
            f"{len(bad)} season(s) have gaps in round numbering",
            bad,
        )
    ]


@check
def every_race_has_results(con) -> list[Finding]:
    df = _q(
        con,
        """
        SELECT r.season, r.round, r.race_name
        FROM raw_races r
        LEFT JOIN raw_results res USING (season, round)
        WHERE res.driver_id IS NULL
          AND r.race_date <= current_date
        ORDER BY r.season, r.round
        """,
    )
    if df.empty:
        return []
    return [
        Finding(
            "every_race_has_results",
            WARN,
            f"{len(df)} past race(s) have no results (cancelled, or ingest incomplete)",
            df,
        )
    ]


@check
def field_size_plausible(con) -> list[Finding]:
    df = _q(
        con,
        """
        SELECT season, round, count(*) n
        FROM raw_results GROUP BY 1,2 HAVING count(*) < 10 OR count(*) > 26
        ORDER BY 1,2
        """,
    )
    if df.empty:
        return []
    return [Finding("field_size_plausible", ERROR, f"{len(df)} race(s) with an odd field size", df)]


# ---------------------------------------------------------------------------
# Referential integrity
# ---------------------------------------------------------------------------
@check
def results_reference_a_race(con) -> list[Finding]:
    df = _q(
        con,
        """
        SELECT DISTINCT res.season, res.round
        FROM raw_results res
        LEFT JOIN raw_races r USING (season, round)
        WHERE r.season IS NULL
        """,
    )
    if df.empty:
        return []
    return [Finding("results_reference_a_race", ERROR, f"{len(df)} orphaned result group(s)", df)]


@check
def drivers_are_known(con) -> list[Finding]:
    """Every driver in results must exist in raw_drivers for that season.
    Unmapped drivers silently vanish from the FastF1 join."""
    df = _q(
        con,
        """
        SELECT DISTINCT res.season, res.driver_id
        FROM raw_results res
        LEFT JOIN raw_drivers d USING (season, driver_id)
        WHERE d.driver_id IS NULL
        ORDER BY 1,2
        """,
    )
    if df.empty:
        return []
    return [Finding("drivers_are_known", ERROR, f"{len(df)} driver-season(s) missing from raw_drivers", df)]


# ---------------------------------------------------------------------------
# Domain constraints
# ---------------------------------------------------------------------------
@check
def positions_in_range(con) -> list[Finding]:
    out = []
    df = _q(
        con,
        """
        SELECT season, round, driver_id, position, grid
        FROM raw_results
        WHERE position IS NOT NULL AND (position < 1 OR position > 30)
           OR grid IS NOT NULL AND (grid < 0 OR grid > 30)
        """,
    )
    if not df.empty:
        out.append(Finding("positions_in_range", ERROR, f"{len(df)} row(s) outside 1..30", df))

    neg = _q(con, "SELECT count(*) n FROM raw_results WHERE points < 0").iloc[0]["n"]
    if neg:
        out.append(Finding("positions_in_range", ERROR, f"{neg} row(s) with negative points"))
    return out


@check
def one_winner_per_race(con) -> list[Finding]:
    df = _q(
        con,
        """
        SELECT season, round, count(*) n_at_p1
        FROM raw_results WHERE position = 1
        GROUP BY 1,2 HAVING count(*) <> 1 ORDER BY 1,2
        """,
    )
    if df.empty:
        return []
    return [Finding("one_winner_per_race", ERROR, f"{len(df)} race(s) without exactly one P1", df)]


@check
def finishing_positions_unique(con) -> list[Finding]:
    df = _q(
        con,
        """
        SELECT season, round, position, count(*) n
        FROM raw_results WHERE position IS NOT NULL
        GROUP BY 1,2,3 HAVING count(*) > 1 ORDER BY 1,2,3
        """,
    )
    if df.empty:
        return []
    return [Finding("finishing_positions_unique", ERROR, f"{len(df)} duplicated position(s)", df)]


@check
def quali_positions_contiguous(con) -> list[Finding]:
    df = _q(
        con,
        """
        SELECT season, round, count(*) n, max(position) mx
        FROM raw_qualifying WHERE position IS NOT NULL
        GROUP BY 1,2 HAVING count(*) <> max(position) ORDER BY 1,2
        """,
    )
    if df.empty:
        return []
    return [Finding("quali_positions_contiguous", WARN, f"{len(df)} session(s) where positions skip", df)]


@check
def classified_implies_finished_ahead(con) -> list[Finding]:
    """A classified finisher should never be ranked behind a retirement that
    completed fewer laps. Catches parser bugs in position/classified."""
    df = _q(
        con,
        """
        WITH r AS (SELECT * FROM raw_results WHERE position IS NOT NULL AND laps IS NOT NULL)
        SELECT a.season, a.round, a.driver_id AS classified_driver,
               b.driver_id AS ranked_ahead, a.laps AS a_laps, b.laps AS b_laps
        FROM r a JOIN r b USING (season, round)
        WHERE a.classified AND NOT b.classified
          AND b.position < a.position AND b.laps < a.laps
        ORDER BY 1,2 LIMIT 50
        """,
    )
    if df.empty:
        return []
    return [
        Finding(
            "classified_implies_finished_ahead",
            WARN,
            f"{len(df)} case(s) where a retirement outranks a finisher on fewer laps",
            df,
        )
    ]


@check
def dnf_rate_plausible(con) -> list[Finding]:
    df = _q(
        con,
        """
        SELECT season, round,
               round(100.0 * sum(CASE WHEN dnf THEN 1 ELSE 0 END) / count(*), 1) AS dnf_pct
        FROM raw_results GROUP BY 1,2
        HAVING dnf_pct > 70 ORDER BY dnf_pct DESC
        """,
    )
    if df.empty:
        return []
    return [Finding("dnf_rate_plausible", WARN, f"{len(df)} race(s) with >70% retirements", df)]


@check
def points_awarded_to_finishers(con) -> list[Finding]:
    df = _q(
        con,
        """
        SELECT season, round, driver_id, position, points, status
        FROM raw_results
        WHERE points > 0 AND (position IS NULL OR position > 11)
        ORDER BY 1,2 LIMIT 50
        """,
    )
    if df.empty:
        return []
    # Legitimate in rare cases (post-race penalties reshuffling), so WARN.
    return [Finding("points_awarded_to_finishers", WARN, f"{len(df)} scorer(s) outside the top 11", df)]


# ---------------------------------------------------------------------------
# Temporal integrity - the checks that protect against leakage
# ---------------------------------------------------------------------------
@check
def races_chronological(con) -> list[Finding]:
    df = _q(
        con,
        """
        SELECT season, round, race_date,
               lag(race_date) OVER (PARTITION BY season ORDER BY round) prev
        FROM raw_races QUALIFY prev IS NOT NULL AND race_date < prev
        """,
    )
    if df.empty:
        return []
    return [Finding("races_chronological", ERROR, f"{len(df)} race(s) out of date order", df)]


@check
def forecasts_predate_their_session(con) -> list[Finding]:
    """A 'forecast' fetched after the session started is hindsight, not a
    forecast. If one of these gets into training the backtest is worthless."""
    df = _q(
        con,
        """
        SELECT f.season, f.round, f.session, f.fetched_at_utc, r.race_start_utc
        FROM raw_forecast f JOIN raw_races r USING (season, round)
        WHERE f.session = 'R' AND r.race_start_utc IS NOT NULL
          AND f.fetched_at_utc > r.race_start_utc
        ORDER BY 1,2
        """,
    )
    if df.empty:
        return []
    return [Finding("forecasts_predate_their_session", ERROR, f"{len(df)} hindsight forecast(s)", df)]


@check
def practice_predates_race(con) -> list[Finding]:
    df = _q(
        con,
        """
        SELECT p.season, p.round, p.session, p.session_start_utc, r.race_start_utc
        FROM raw_session_pace p JOIN raw_races r USING (season, round)
        WHERE p.session IN ('FP1','FP2','FP3','Q')
          AND p.session_start_utc IS NOT NULL AND r.race_start_utc IS NOT NULL
          AND p.session_start_utc > r.race_start_utc
        LIMIT 50
        """,
    )
    if df.empty:
        return []
    return [Finding("practice_predates_race", ERROR, f"{len(df)} practice session(s) after the race", df)]


# ---------------------------------------------------------------------------
# Ingest health
# ---------------------------------------------------------------------------
@check
def no_failed_ingests(con) -> list[Finding]:
    df = _q(
        con,
        """
        SELECT source, scope, substr(detail, 1, 60) AS detail
        FROM ingest_log WHERE status = 'failed' ORDER BY source, scope
        """,
    )
    if df.empty:
        return []
    return [Finding("no_failed_ingests", WARN, f"{len(df)} ingest task(s) recorded as failed", df)]


def run(con: duckdb.DuckDBPyConnection | None = None) -> Report:
    report = Report()
    if con is not None:
        for fn in _CHECKS:
            report.findings.extend(fn(con))
        return report

    with connect(read_only=True) as owned:
        for fn in _CHECKS:
            report.findings.extend(fn(owned))
    return report


@check
def retirement_flag_matches_laps(con) -> list[Finding]:
    """Cross-check the finished/retired flag against distance actually covered.

    The flag is derived from a free-text status field owned by someone else,
    and that text has already changed wording once mid-dataset ("+1 Lap"
    becoming "Lapped" in 2023), which silently turned a third of all finishes
    into retirements. Laps completed is a fact rather than a label, so it makes
    an independent witness: anyone within two laps of the winner was running at
    the flag, whatever the status string happens to say this year.
    """
    df = _q(
        con,
        """
        WITH winner AS (
            SELECT season, round, max(laps) AS race_laps
            FROM raw_results GROUP BY 1,2
        )
        SELECT r.season, r.round, r.status, count(*) AS rows
        FROM raw_results r JOIN winner w USING (season, round)
        WHERE r.dnf AND r.laps >= w.race_laps - 2
        GROUP BY 1,2,3 ORDER BY rows DESC
        """,
    )
    if df.empty:
        return []

    # Discriminate between the two things this can catch. A car that stops on
    # the last lap genuinely looks like a finisher on laps alone and is a
    # handful of rows; a changed status vocabulary hits one status string
    # hundreds of times. Only the second is a defect.
    by_status = df.groupby("status", as_index=False)["rows"].sum().sort_values("rows", ascending=False)
    systematic = by_status[by_status["rows"] >= 20]
    if systematic.empty:
        return [
            Finding(
                "retirement_flag_matches_laps",
                WARN,
                f"{int(df['rows'].sum())} entries retired within two laps of the winner "
                f"- late retirements and exclusions, which is normal",
                by_status,
            )
        ]
    statuses = ", ".join(systematic["status"].astype(str))
    return [
        Finding(
            "retirement_flag_matches_laps",
            ERROR,
            f"status '{statuses}' is marked as a retirement on {int(systematic['rows'].sum())} "
            f"entries that finished within two laps of the winner. The upstream status "
            f"vocabulary has probably changed - update ingest.jolpica.is_finished, "
            f"then run `make repair`.",
            by_status,
        )
    ]
