.PHONY: setup setup-uv which-python data data-jolpica data-fastf1 validate \
        features build-features repair train backtest predict title-backtest \
        calibrate-spread dashboard test lint all clean

SEASONS ?= 2018-2026
START   ?= 2022   # first season the backtest reports on

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

train:
	$(PY) -m f1pred.cli train

backtest:
	$(PY) -m f1pred.cli backtest --start-season $(START)

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

# ---- quality ---------------------------------------------------------------
test:
	$(PYTEST) -q

lint:
	$(RUFF) check src tests
	$(RUFF) format --check src tests

all: features train backtest predict dashboard

clean:
	rm -rf data/*.duckdb data/*.duckdb.wal
	@echo "Dropped the database. Caches kept - 'make features' will rebuild fast."
