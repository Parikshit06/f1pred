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
CONFIG_DIR = ROOT / "config"
PREDICTIONS = ROOT / "predictions"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models_out"

for _p in (DATA, HTTP_CACHE, FASTF1_CACHE, PREDICTIONS, REPORTS, MODELS):
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

# Every number below is fitted, never chosen. They come from a walk-forward
# over 2022-2023 only - the seasons BEFORE the window the README reports - so
# the headline table stays out-of-sample. An earlier version tuned from 2022
# with no upper bound, which quietly walked through 2024-2026 and fitted the
# settings on the very races it then reported; see tune_* in backtest.py.
#
# Re-fit with: f1pred.cli backtest --start-season 2024 --tune --tune-season 2022

# How much heavier a current-season race counts in training. The reel this is
# modelled on used 3x because that felt about right; the tuner says 1.5.
#
# It briefly said 4x, which was the model compensating for a defect rather than
# learning about F1: recent form was being poisoned by retirements scored as
# last place, so the only way to get a usable signal was to lean hard on the
# newest races. Once pace and reliability were separated the need for that
# largely went away. A settings value that moves a long way when a bug is fixed
# is usually a symptom, not a finding.
DEFAULT_CURRENT_SEASON_WEIGHT = 1.5

# Plackett-Luce sharpness. Below 1 means the model is more decisive than raw
# ranking scores imply; it is fitted by log loss on held-out races.
DEFAULT_TEMPERATURE = 0.4

# How much to trust the closed-form ranking over the simulation. The rest is
# the Monte Carlo, which is what prices in retirements and safety cars - and it
# earns more of the weight now that the retirement rates it samples from are
# correct.
DEFAULT_BLEND_WEIGHT = 0.6

# How far a team's real pace can end up from where the model has it, over the
# rest of a season. This is the term that stops a title projection printing a
# range it cannot honour.
#
# The projection used to hold every team's pace fixed at whatever the ranker
# said this weekend, so the only uncertainty in the remaining season was
# race-to-race noise - and noise averages out over a dozen races. The published
# 10th-90th band was therefore very narrow, and graded against real final
# constructors' standings it contained the truth 45% of the time instead of 80%.
#
# Three things can move a season away from the projection, and it is worth being
# clear which one this is, because the obvious answer turned out to be wrong:
#
#   race-to-race noise  already modelled, and largely self-cancelling
#   development         real - fitted at 1.37 finishing positions over a full
#                       season remaining (95% 0.90-1.83, 283 team-checkpoints,
#                       decaying to nothing by the last few races) - but far too
#                       small to explain the gap. Adding it alone moved coverage
#                       from 45% to 47%.
#   being wrong now     the ranker's read of the field at round 8 is not the
#                       field's true pace, and unlike race noise that error does
#                       not average out. It is carried into every remaining race.
#
# The third dominates, so this constant stands for all three together rather
# than pretending to isolate development. Each simulated season draws one pace
# offset per team, held for the rest of the year, at this size scaled by how
# much of the season is left.
#
# Calibrated the only way a predictive interval honestly can be - by whether it
# covers. Swept on 2019-2022, where 6.0 put coverage at 82%, then graded on
# 2023-2025, which the sweep never saw:
#
#                        coverage   mean band width
#   fixed pace              45.0%              38.6
#   calibrated (6.0)        70.8%              89.5
#   target                  80.0%
#
# Still short of 80, and the method page says so rather than rounding it off.
# The gain is concentrated early in the season, where the old band was worst:
# with a third of the season run, coverage went from 30% to 90%.
#
# Re-fit with: f1pred.cli calibrate-spread
SEASON_PACE_UNCERTAINTY = 6.0  # finishing positions, per full season remaining

# The exchange rate between the two scales, so the figure above can be fitted in
# finishing positions - where it means something to a reader - and applied in
# model score, where the simulator works. Measured inside the simulator by
# nudging one car's score and reading off where it finishes: 5.62 positions per
# unit of score spread, and stable across eras at 5.45 to 6.07.
POSITIONS_PER_SCORE_SD = 5.62

N_SIMULATIONS = 10_000
TOP_N = 10  # how many drivers we publish; we still model the full field

RANDOM_SEED = 20260913
