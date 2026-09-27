.PHONY: setup demo data data-jolpica data-openf1 data-practice validate repair features predict dashboard \
        backtest experiments title-backtest calibrate-spread bias verify test coverage lint all clean

SEASONS ?= 2018-2026
# START is the first season reported on; TUNE is the first season the settings
# are fitted on (TUNE..START-1), and must come before it.
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

# Installs the exact versions in uv.lock (what CI and the reports used).
setup:
	@if command -v uv >/dev/null 2>&1; then uv sync --locked --extra dev; \
	else python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"; fi

# The whole pipeline on a synthetic championship: no network, about a minute.
demo:
	$(PY) -m f1pred.cli demo

# ---- data (resumable) --------------------------------------------------------
data: data-jolpica data-openf1

data-jolpica:
	$(PY) -m f1pred.cli ingest-jolpica --seasons $(SEASONS)

# Official starting grids and weekend entry lists (2023+).
data-openf1:
	$(PY) -m f1pred.cli ingest-openf1 --seasons 2026

# Practice pace for the race weekend in progress (FP1-FP3 aggregates only).
data-practice:
	$(PY) -m f1pred.cli ingest-fastf1 --next

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
	$(PY) -m f1pred.cli backtest --start-season $(START) --tune-season $(TUNE)

experiments:
	$(PY) -m f1pred.cli experiments --start-season $(START) --tune-season $(TUNE)

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

coverage:
	$(PY) -m pytest -q --cov --cov-report=term-missing

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

all: features backtest predict dashboard

clean:
	rm -f data/*.duckdb data/*.duckdb.wal
	rm -rf data/demo
