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

# OpenF1: official starting grids and per-session driver lists, 2023 onward.
# The free tier allows about 30 requests a minute; one every 2.1s stays under.
OPENF1_BASE = "https://api.openf1.org/v1"
OPENF1_FIRST_SEASON = 2023
OPENF1_MIN_INTERVAL = 2.1
OPENF1_HOURLY_LIMIT = 1500

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# ---- modelling -------------------------------------------------------------
# FastF1 timing starts in 2018; earlier seasons have results but no pace data.
FIRST_SEASON = 2018
CURRENT_SEASON = 2026

# Fallbacks only: `make backtest` fits these on 2022-23 and writes them to
# reports/backtest.json, which the live forecast reads (backtest.load_settings).
DEFAULT_CURRENT_SEASON_WEIGHT = 1.5  # training weight on current-season races
DEFAULT_TEMPERATURE = 0.35  # Plackett-Luce sharpness; <1 is more decisive
DEFAULT_BLEND_WEIGHT = 0.6  # Plackett-Luce vs simulation in the mixture
DEFAULT_QUALI_TEMPERATURE = 0.5  # the qualifying model's own Plackett-Luce temperature

# How far a team's true pace can sit from the model's estimate over a full
# remaining season, in finishing positions. Mostly model error rather than
# development, and unlike race-to-race noise it doesn't average out.
# Chosen by coverage of the projection's 10th-90th band: make calibrate-spread
SEASON_PACE_UNCERTAINTY = 6.0
# The same for a driver against their own teammate. Chosen by coverage of the
# teammate points gap on 2019-22, where scoring each driver on their own recent
# weekends (championship.season_strength) already covers it: the sweep picks 0.
SEASON_DRIVER_UNCERTAINTY = 0.0

# Finishing positions per unit of score spread, measured in the simulator.
# Converts the constant above from positions into score.
POSITIONS_PER_SCORE_SD = 5.62

N_SIMULATIONS = 10_000
TOP_N = 10  # rows published per board; the full field is still modelled

# A pre-qualifying forecast is only logged within this many days of the race,
# so the record holds race-week calls rather than whatever the scheduler ran
# while grading the previous weekend.
LOG_WINDOW_DAYS = 5.0
# Within that window, the pre-qualifying call waits for the weekend's practice
# pace - it measurably improves the qualifying forecast - but never beyond this
# many hours before qualifying, so a practice-data outage can't cost the record
# a race.
PRACTICE_WAIT_HOURS = 6.0

RANDOM_SEED = 20260913
