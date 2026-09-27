"""Where a forecast or a report came from.

Enough to reproduce it, or to know that it can't be: the code (git commit),
the model and feature definitions (short hashes), the data it was trained on
(row counts and a hash of the results table) and the library versions.
Deliberately a dictionary written into each output, not a tracking server.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata

from . import __version__, config


def _short_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def git_commit() -> str | None:
    """The commit that produced this, with '-dirty' if the tree had edits."""
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha[:12]
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=config.ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if head.returncode != 0:
            return None
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=config.ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def model_version(feature_names: list[str], params: dict, feature_version: str) -> str:
    """Changes whenever the features, their definitions or the model settings do."""
    return _short_hash(
        {"features": list(feature_names), "params": params, "feature_version": feature_version}
    )


def data_snapshot() -> dict:
    """What the database held: row counts, the last race with results, and a hash
    of the results that the labels come from."""
    from .store import connect, database_exists

    if not database_exists():
        return {}
    with connect(read_only=True) as con:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall() if r[0].startswith("raw_")]
        counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}
        last = con.execute(
            "SELECT season, round FROM raw_results ORDER BY season DESC, round DESC LIMIT 1"
        ).fetchone()
        rows = con.execute(
            "SELECT season, round, driver_id, position, grid FROM raw_results ORDER BY 1, 2, 3"
        ).fetchall()
    return {
        "rows": counts,
        "results_through": list(last) if last else None,
        "results_hash": _short_hash(rows),
    }


def software() -> dict:
    versions = {"python": platform.python_version(), "f1pred": __version__}
    for pkg in ("xgboost", "numpy", "pandas", "duckdb", "scikit-learn"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def record(**extra) -> dict:
    """The provenance block for one output."""
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "data": data_snapshot(),
        "software": software(),
        **extra,
    }
