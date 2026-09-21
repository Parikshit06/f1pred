.PHONY: setup setup-uv which-python data data-jolpica data-fastf1 validate \
        features build-features repair backtest predict title-backtest \
        calibrate-spread dashboard bias verify test lint all clean

SEASONS ?= 2018-2026

# The reported window, defined once. `make backtest` has to reproduce the table
# in the README and on the method page, and it did not: the target ran
# --start-season 2022 with no tuning, while the published numbers come from a
# run started at 2024 with the settings fitted on 2022-23. Two different
# answers under one command name.
#
# START is the first season REPORTED on. TUNE is the season the temperature,
# blend and recency weight are fitted from, and it must be earlier - fitting
# and reporting on the same races makes the settings part of the answer. The
# CLI refuses the combination if it is not.
START ?= 2024
TUNE  ?= 2022

# Which Python runs everything. Checked in this order so a fresh clone works
# whatever the person has installed:
#
#   .venv/     the project's own environment, if it has been created
#   uv         if it is on PATH
#   python3    last resort, assuming the package is already installed
#
# This used to be hardcoded to `uv run python`, which failed at the first
# target with "make: uv: No such file or directory" on any machine where uv was
# not on PATH - including one that already had a perfectly good .venv sitting
# in the project directory.
ifneq ($(wildcard .venv/bin/python),)
  PY     := .venv/bin/python
  PYTEST := .venv/bin/pytest
  RUFF   := .venv/bin/ruff
else ifneq ($(shell command -v uv 2>/dev/null),)
  PY     := uv run python
  PYTEST := uv run pytest
  RUFF   := uv run ruff
else
  PY     := python3
  PYTEST := python3 -m pytest
  RUFF   := python3 -m ruff
endif

# Creates .venv and installs the project into it. Uses uv when available
# because it is much faster, and falls back to the standard library otherwise.
setup:
	@if command -v uv >/dev/null 2>&1; then \
	  uv sync; \
	else \
	  python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"; \
	fi
	@echo "Environment ready. Next: make data"

# Explicit uv path, for when you know you want it.
setup-uv:
	uv sync
	@echo "Environment ready. Next: make data"

# Prints which interpreter the other targets will use. Worth having when a
# command behaves differently from the same command run by hand.
which-python:
	@echo $(PY)
	@$(PY) -c "import sys; print(sys.executable); import f1pred; print('f1pred ok')"

# ---- data ------------------------------------------------------------------
# Runs on YOUR machine (needs internet). Resumable - safe to ctrl-C and rerun.
data: data-jolpica data-fastf1

data-jolpica:
	$(PY) -m f1pred.cli ingest-jolpica --seasons $(SEASONS)

data-fastf1:
	$(PY) -m f1pred.cli ingest-fastf1 --seasons $(SEASONS)

# ---- pipeline --------------------------------------------------------------
# Data quality gate. Run it before trusting anything downstream - it is what
# caught 398 finishes being stored as retirements.
validate:
	$(PY) -m f1pred.cli validate

features:
	$(PY) -m f1pred.cli build-features

# Alias, because the CLI subcommand is spelled build-features and muscle memory
# reaches for that.
build-features: features

# Recompute finished/retired flags from stored status text. Needed after any
# change to the retirement rule; does not re-download anything.
repair:
	$(PY) -m f1pred.cli repair

# There is no `train` target. There was one, and it called a CLI subcommand
# that does not exist, so it failed on the spot and took `make all` down with
# it. Nothing in this project loads a model off disk: predict, backtest and
# verify each fit one on exactly the races they are allowed to see, which is
# the whole point of a walk-forward. A model saved on Friday would only be a
# way to accidentally use one that had seen too much.

# Reproduces the published accuracy table. Slow: it refits per race.
backtest:
	$(PY) -m f1pred.cli backtest --start-season $(START) --tune --tune-season $(TUNE)

predict:
	$(PY) -m f1pred.cli predict --next

# Grades the championship projection's favourite against completed seasons.
title-backtest:
	$(PY) -m f1pred.cli title-backtest

# Re-fits how wide the projection's range has to be to keep its promise, then
# grades that choice on seasons the fit never saw.
calibrate-spread:
	$(PY) -m f1pred.cli calibrate-spread

# Renders both pages: the forecast and the evidence behind it.
dashboard:
	$(PY) -m f1pred.cli dashboard

# ---- diagnostics -----------------------------------------------------------
# Per-driver bias: is the model just favouring the famous names? The raw column
# says yes and is an artifact; the corrected one is the answer.
bias:
	$(PY) -m f1pred.cli bias

# Seven audits in one pass: leakage, input coverage, feature weighting, driver
# bias, accuracy against baselines, calibration, prediction sanity.
verify:
	$(PY) -m f1pred.cli verify --start-season $(START)

# ---- quality ---------------------------------------------------------------
test:
	$(PYTEST) -q

lint:
	$(RUFF) check src tests
	$(RUFF) format --check src tests

all: features backtest predict dashboard

clean:
	rm -rf data/*.duckdb data/*.duckdb.wal
	@echo "Dropped the database. Caches kept - 'make features' will rebuild fast."
