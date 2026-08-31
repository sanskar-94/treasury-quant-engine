# Treasury Quant Engine - developer entry points
PY := ./.venv/bin/python
PIP := ./.venv/bin/pip

.PHONY: help venv install data curve features train backtest report test lint fmt typecheck serve trade clean all

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the virtualenv
	python3 -m venv .venv && $(PIP) install --upgrade pip setuptools wheel

install: ## Install the package and all extras in editable mode
	$(PIP) install -e ".[all]"

data:  ## Pull Treasury + FRED history into the local cache
	$(PY) -m tqe.cli data pull

curve: ## Fit the Nelson-Siegel-Svensson curve over all history
	$(PY) -m tqe.cli curve fit

features: ## Build the model feature matrix
	$(PY) -m tqe.cli features build

train: ## Walk-forward train the ensemble
	$(PY) -m tqe.cli train --walk-forward

backtest: ## Run the backtest with full transaction costs
	$(PY) -m tqe.cli backtest

report: ## Render the tearsheet from the latest backtest
	$(PY) -m tqe.cli report

trade: ## Dry-run one live trading session (no orders are sent)
	$(PY) -m tqe.cli trade --dry-run

serve: ## Start the FastAPI service on :8000
	$(PY) -m tqe.cli serve --port 8000

all: data curve features train backtest report  ## Full pipeline end to end

test:  ## Run the test suite
	$(PY) -m pytest -q

test-cov: ## Tests with a coverage report
	$(PY) -m pytest -q --cov=tqe --cov-report=term-missing

lint:  ## Lint with ruff
	$(PY) -m ruff check src tests

fmt:   ## Auto-format and fix lint
	$(PY) -m ruff format src tests && $(PY) -m ruff check --fix src tests

typecheck: ## Static type check
	$(PY) -m mypy src/tqe --ignore-missing-imports

clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info htmlcov .coverage
