.PHONY: sync lint format typecheck test preflight migrate api worker up down logs bench eval-db eval-ingest eval-queries eval eval-pipeline

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

# Seed a throwaway database with synthetic vectors and compare exact scans
# against halfvec HNSW. Requires the atlas-postgres container to be running.
bench:
	uv run --with "psycopg[binary]" --with numpy python scripts/bench_indexing.py

# Retrieval evaluation harness. Uses a separate database so production data is
# never touched; `make eval-db` creates and migrates it when it does not exist.
EVAL_DB ?= postgresql+asyncpg://atlas:atlas@localhost:5432/atlas_eval

eval-db:
	@docker exec atlas-postgres-1 psql -U atlas -d postgres -tAc \
		"SELECT 1 FROM pg_database WHERE datname='atlas_eval'" | grep -q 1 \
		|| docker exec atlas-postgres-1 createdb -U atlas atlas_eval
	POSTGRES__URL="$(EVAL_DB)" uv run alembic upgrade head

eval-ingest:
	POSTGRES__URL="$(EVAL_DB)" uv run python scripts/eval_retrieval.py ingest \
		--path specs --path README.md --path CONVENTIONS.md

eval-queries:
	POSTGRES__URL="$(EVAL_DB)" uv run python scripts/eval_retrieval.py generate \
		--sample 40 --out specs/eval/queries-atlas-repo.jsonl --overwrite

eval:
	POSTGRES__URL="$(EVAL_DB)" uv run python scripts/eval_retrieval.py run \
		--queries specs/eval/queries-atlas-repo.jsonl --arms full4096,trunc2048

# Production pipeline: dense SQL vs RRF vs RRF + rerank at several depths.
eval-pipeline:
	POSTGRES__URL="$(EVAL_DB)" uv run python scripts/eval_retrieval.py run \
		--queries specs/eval/queries-atlas-repo.jsonl \
		--arms full4096,hybrid,hybrid-rerank@50,hybrid-rerank@24,hybrid-rerank@12 \
		--label hybrid
