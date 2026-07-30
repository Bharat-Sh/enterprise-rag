# Convenience targets. `make` is not installed on Windows by default — either
# use Git Bash with make, or run the underlying commands directly (they are all
# one-liners and listed in the README).

.DEFAULT_GOAL := help
.PHONY: help install dev lint format typecheck arch test check up down logs clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install runtime + dev dependencies
	uv venv
	uv pip install -e ".[dev]"

dev: ## Run the API with auto-reload
	uv run uvicorn rag.api.main:app --reload --host 0.0.0.0 --port 8000

lint: ## Check style without modifying files
	uv run ruff check .
	uv run ruff format --check .

format: ## Apply formatting and auto-fixable lint rules
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Run mypy in strict mode
	uv run mypy

arch: ## Verify the hexagonal import contracts
	uv run lint-imports

test: ## Run the test suite with coverage
	uv run pytest --cov=rag --cov-report=term-missing

check: lint typecheck arch test ## Everything CI runs, locally

up: ## Start the local stack (postgres, redis, qdrant, api)
	docker compose -f docker/compose.yml up -d --build

down: ## Stop the stack, keeping volumes
	docker compose -f docker/compose.yml down

logs: ## Tail the API logs
	docker compose -f docker/compose.yml logs -f api

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
