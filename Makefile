.PHONY: setup setup-uv data data-jolpica data-fastf1 forecast status validate features backtest predict bias verify dashboard test lint all clean

VENV    := .venv
PY      := $(shell if [ -x "$(VENV)/bin/python" ]; then echo "$(VENV)/bin/python"; \
                   elif command -v uv >/dev/null 2>&1; then echo "uv run python"; \
                   else echo "python3"; fi)
SEASONS ?= 2018-2026
START   ?= 2024
# make backtest TUNE=1   fits temperature/blend/recency on TUNE_SEASON first
TUNE       ?=
TUNE_SEASON ?= $(shell expr $(START) - 2)
TUNE_FLAGS := $(if $(TUNE),--tune --tune-season $(TUNE_SEASON),)
SIMS    ?= 3000

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e ".[dev]"
	@echo ""
	@echo "macOS note: XGBoost needs OpenMP. If 'make predict' fails to load"
	@echo "libxgboost.dylib, run:  brew install libomp"

setup-uv:
	uv venv && uv pip install -e ".[dev]"

# ---- data (needs internet; resumable) --------------------------------------
data: data-jolpica data-fastf1

data-jolpica:
	$(PY) -m f1pred.cli ingest-jolpica --seasons $(SEASONS)

data-fastf1:
	$(PY) -m f1pred.cli ingest-fastf1 --seasons $(SEASONS)

forecast:
	$(PY) -m f1pred.cli forecast

status:
	$(PY) -m f1pred.cli status

# ---- pipeline --------------------------------------------------------------
validate:
	$(PY) -m f1pred.cli validate

features:
	$(PY) -m f1pred.cli build-features

backtest:
	$(PY) -m f1pred.cli backtest --start-season $(START) --sims $(SIMS) $(TUNE_FLAGS)

predict:
	$(PY) -m f1pred.cli predict --next

# Per-driver bias report: is the model favouring particular drivers?
bias:
	$(PY) -m f1pred.cli bias --start-season $(START)

# Seven audits: leakage, inputs, weighting, bias, accuracy, calibration,
# prediction sanity. Exits non-zero on any failure.
verify:
	$(PY) -m f1pred.cli verify --start-season $(START) --sims $(SIMS)

dashboard:
	$(PY) -m f1pred.cli dashboard

all: validate features backtest predict dashboard

# ---- quality ---------------------------------------------------------------
test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

clean:
	rm -f data/f1.duckdb data/f1.duckdb.wal
	@echo "Database dropped. HTTP cache kept, so re-ingest costs no requests."
