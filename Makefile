.PHONY: setup data data-jolpica data-fastf1 validate repair features predict dashboard \
        backtest title-backtest calibrate-spread bias verify test lint all clean

SEASONS ?= 2018-2026
# START is the first season reported on; TUNE is the season the settings are
# fitted on, and must come before it.
START   ?= 2024
TUNE    ?= 2022

# .venv if it exists, then uv, then whatever python3 is on PATH.
ifneq ($(wildcard .venv/bin/python),)
  PY := .venv/bin/python
else ifneq ($(shell command -v uv 2>/dev/null),)
  PY := uv run python
else
  PY := python3
endif

setup:
	@if command -v uv >/dev/null 2>&1; then uv sync --extra dev; \
	else python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"; fi

# ---- data (resumable) --------------------------------------------------------
data: data-jolpica data-fastf1

data-jolpica:
	$(PY) -m f1pred.cli ingest-jolpica --seasons $(SEASONS)

data-fastf1:
	$(PY) -m f1pred.cli ingest-fastf1 --seasons $(SEASONS)

validate:
	$(PY) -m f1pred.cli validate

# Recompute finished/retired flags from the stored status text.
repair:
	$(PY) -m f1pred.cli repair

# ---- forecast ----------------------------------------------------------------
features:
	$(PY) -m f1pred.cli build-features

predict:
	$(PY) -m f1pred.cli predict --next

dashboard:
	$(PY) -m f1pred.cli dashboard

# ---- measured figures (reports/*.json) --------------------------------------
backtest:
	$(PY) -m f1pred.cli backtest --start-season $(START) --tune --tune-season $(TUNE)

title-backtest:
	$(PY) -m f1pred.cli title-backtest

calibrate-spread:
	$(PY) -m f1pred.cli calibrate-spread

bias:
	$(PY) -m f1pred.cli bias

verify:
	$(PY) -m f1pred.cli verify --start-season $(START)

# ---- checks ------------------------------------------------------------------
test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

all: features backtest predict dashboard

clean:
	rm -f data/*.duckdb data/*.duckdb.wal
