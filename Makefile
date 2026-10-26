.PHONY: sync lint format typecheck test preflight migrate api worker up down logs

# Keep tooling from dropping caches and bytecode into the working tree.
export PYTHONDONTWRITEBYTECODE := 1
export RUFF_NO_CACHE := true

sync:
	uv sync

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

typecheck:
	uv run mypy src

test:
	uv run pytest

preflight: lint typecheck test

migrate:
	uv run alembic upgrade head

api:
	uv run atlas-api

worker:
	uv run arq atlas_worker.main.WorkerSettings

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api worker
