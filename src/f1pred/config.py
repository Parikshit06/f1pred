"""Central paths and constants. Import this, don't hardcode paths anywhere else."""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HTTP_CACHE = DATA / "cache"
FASTF1_CACHE = DATA / "fastf1_cache"
DB_PATH = DATA / "f1.duckdb"
PREDICTIONS = ROOT / "predictions"
REPORTS = ROOT / "reports"

# There were two more here - CONFIG_DIR and MODELS - that nothing imported.
# MODELS was worse than unused: it was in this list, so importing f1pred at all
# created an empty models_out/ next to the source. Ranker.save() takes the path
# it writes to, so nothing needs a default one.
for _p in (DATA, HTTP_CACHE, FASTF1_CACHE, PREDICTIONS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"

# jolpica REQUIRES an identifying user agent, so they can contact or block a
# specific misbehaving client version instead of an IP range. A repo URL does
# that job without putting a personal email in every request header.
# Override with F1PRED_CONTACT if you ever need to.
REPO_URL = "https://github.com/Parikshit06/f1pred"

_CONTACT = os.environ.get("F1PRED_CONTACT", REPO_URL)
USER_AGENT = f"f1pred/0.1.0 (+{_CONTACT})" if _CONTACT else "f1pred/0.1.0"

# Their published limits are 4 req/sec burst and 500 req/hour sustained, but
# in practice the burst limiter rejects well below 4/s - the quota is shared
# across everyone on your IP, and their window is shorter than a second.
# 1.2s (~0.8 req/s) runs clean. A full 2018-2026 backfill is ~450 requests,
# so about 9 minutes. Not worth shaving.
JOLPICA_MIN_INTERVAL = 1.2
JOLPICA_HOURLY_LIMIT = 450  # margin under their 500/hr
# Every 429 permanently slows the client for the rest of the run, so it
# self-tunes to whatever the server actually tolerates today.
JOLPICA_BACKOFF_GROWTH = 1.5
JOLPICA_MAX_INTERVAL = 6.0
JOLPICA_PAGE_SIZE = 100  # their maximum

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
# FastF1 only has detailed timing from 2018. Results go back to 1950 but
# without practice pace they are not comparable, so 2018 is our floor.
FIRST_SEASON = 2018
CURRENT_SEASON = 2026

# Fitted on 2022-23 only, which is before the window the README reports, so the
# headline numbers stay out-of-sample. The tuner is bounded at both ends -
# an earlier version had no end season and walked into the reported window.
# Re-fit: f1pred.cli backtest --start-season 2024 --tune --tune-season 2022

# Weight on current-season races in training. Was briefly fitted at 4x, which
# turned out to be the model compensating for retirements being scored as last
# place; it dropped to 1.5 once pace and reliability were separated.
DEFAULT_CURRENT_SEASON_WEIGHT = 1.5

# Plackett-Luce sharpness. Below 1 = more decisive than the raw scores imply.
DEFAULT_TEMPERATURE = 0.4

# Weight on the closed-form ranking vs the simulation.
DEFAULT_BLEND_WEIGHT = 0.6

# How far a team's real pace can be from where the model has it, over the rest
# of a season. Without it the projection only carried race-to-race noise, which
# averages out, so the published 10th-90th band covered 46% of real final
# constructors' totals instead of 80%.
#
# It is not development. Development is real but small - fitted at 1.37
# finishing positions per season (95% 0.90-1.83, 283 team-checkpoints) - and
# adding exactly that much moved coverage by two points. Most of it is the
# model being wrong about a car now, which unlike race noise is carried into
# every remaining race.
#
# Calibrated by coverage: swept on 2019-22, graded on 2023-25 (46% -> 71%,
# target 80%). Re-fit: f1pred.cli calibrate-spread
SEASON_PACE_UNCERTAINTY = 6.0  # finishing positions, per full season remaining

# Positions per unit of score spread, measured inside the simulator by nudging
# one car's score and reading where it finishes. Stable across eras: 5.45-6.07.
# Lets the constant above be fitted in positions and applied in score.
POSITIONS_PER_SCORE_SD = 5.62

N_SIMULATIONS = 10_000
TOP_N = 10  # how many drivers we publish; we still model the full field

# How near a race a pre-qualifying forecast has to be before it is committed to
# predictions/. Five days reaches back to the Monday of a race week and no
# further, so the logged call is a race-week call - not one the scheduler
# happened to make a fortnight out while grading the previous result.
LOG_WINDOW_DAYS = 5.0

RANDOM_SEED = 20260913
