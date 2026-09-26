"""Paths and constants. Everything else imports these rather than hardcoding."""

from __future__ import annotations

import os
from pathlib import Path

# ---- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HTTP_CACHE = DATA / "cache"
FASTF1_CACHE = DATA / "fastf1_cache"
DB_PATH = DATA / "f1.duckdb"
PREDICTIONS = ROOT / "predictions"
REPORTS = ROOT / "reports"
# The latest season projection. Refreshed every predict run, unlike the logged
# forecast, so the championship panel moves as results come in.
SEASON_NOW = DATA / "season.json"

for _p in (DATA, HTTP_CACHE, FASTF1_CACHE, PREDICTIONS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# ---- sources ---------------------------------------------------------------
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"

# jolpica asks clients to identify themselves. A repo URL does that without
# putting an email in every request.
REPO_URL = "https://github.com/Parikshit06/f1pred"

_CONTACT = os.environ.get("F1PRED_CONTACT", REPO_URL)
USER_AGENT = f"f1pred/0.1.0 (+{_CONTACT})" if _CONTACT else "f1pred/0.1.0"

# Documented limits are 4/s and 500/hour, but the burst limiter trips well
# below 4/s in practice. ~0.8/s runs clean; each 429 slows the client further
# for the rest of the run.
JOLPICA_MIN_INTERVAL = 1.2
JOLPICA_HOURLY_LIMIT = 450
JOLPICA_BACKOFF_GROWTH = 1.5
JOLPICA_MAX_INTERVAL = 6.0
JOLPICA_PAGE_SIZE = 100

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# ---- modelling -------------------------------------------------------------
# FastF1 timing starts in 2018; earlier seasons have results but no pace data.
FIRST_SEASON = 2018
CURRENT_SEASON = 2026

# The next three are fitted on 2022-23, before the reported window:
#   make backtest   (backtest --start-season 2024 --tune --tune-season 2022)
DEFAULT_CURRENT_SEASON_WEIGHT = 2.0  # training weight on current-season races
DEFAULT_TEMPERATURE = 0.4  # Plackett-Luce sharpness; <1 is more decisive
DEFAULT_BLEND_WEIGHT = 0.6  # closed-form ranking vs simulation, for p_win

# How far a team's true pace can sit from the model's estimate over a full
# remaining season, in finishing positions. Mostly model error rather than
# development, and unlike race-to-race noise it doesn't average out.
# Chosen by coverage of the projection's 10th-90th band: make calibrate-spread
SEASON_PACE_UNCERTAINTY = 8.0
# The same for a driver against their own teammate. Chosen by coverage of the
# teammate points gap.
SEASON_DRIVER_UNCERTAINTY = 5.0

# Finishing positions per unit of score spread, measured in the simulator.
# Converts the constant above from positions into score.
POSITIONS_PER_SCORE_SD = 5.62

N_SIMULATIONS = 10_000
TOP_N = 10  # rows published per board; the full field is still modelled

# A pre-qualifying forecast is only logged within this many days of the race,
# so the record holds race-week calls rather than whatever the scheduler ran
# while grading the previous weekend.
LOG_WINDOW_DAYS = 5.0

RANDOM_SEED = 20260913
