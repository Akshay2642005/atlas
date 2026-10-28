.PHONY: sync lint format typecheck test preflight migrate api worker up down logs bench eval-db eval-ingest eval-queries eval-warm eval eval-pipeline \
	eval-corpus-fetch eval-ingest-docs eval-queries-docs eval-docs

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

# mypy's configured import handling hides misuse of third-party APIs; basedpyright
# reports it. `src` only: the test fakes are deliberately duck-typed.
pyright:
	uv run basedpyright src

test:
	uv run pytest

preflight: lint typecheck pyright test

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

# Fill the embedding cache first; the hosted 8B embedder 504s for minutes.
eval-warm:
	POSTGRES__URL="$(EVAL_DB)" uv run python scripts/eval_retrieval.py warm \
		--queries specs/eval/queries-atlas-repo.jsonl

# Public-docs corpus: ~10x the repository corpus, with real exact identifiers.
# The fetcher pins the upstream tags, so re-running it reproduces the corpus.
EVAL_DOCS_TENANT ?= atlas-eval-docs
EVAL_DOCS_QUERIES ?= specs/eval/queries-public-docs.jsonl

eval-corpus-fetch:
	uv run python scripts/fetch_eval_corpus.py

eval-ingest-docs: eval-db
	POSTGRES__URL="$(EVAL_DB)" uv run python scripts/eval_retrieval.py ingest \
		--tenant $(EVAL_DOCS_TENANT) \
		--path .eval-corpora/fastapi --path .eval-corpora/django \
		--path .eval-corpora/pydantic --path .eval-corpora/pgvector

eval-queries-docs:
	POSTGRES__URL="$(EVAL_DB)" uv run python scripts/eval_retrieval.py generate \
		--tenant $(EVAL_DOCS_TENANT) --sample 120 --seed 7 \
		--out $(EVAL_DOCS_QUERIES) --overwrite

eval-warm-docs:
	POSTGRES__URL="$(EVAL_DB)" uv run python scripts/eval_retrieval.py warm \
		--tenant $(EVAL_DOCS_TENANT) --queries $(EVAL_DOCS_QUERIES)

# Width and fusion on the public-docs corpus: the evidence ADR 0003 waits on.
eval-docs:
	POSTGRES__URL="$(EVAL_DB)" uv run python scripts/eval_retrieval.py run \
		--tenant $(EVAL_DOCS_TENANT) --queries $(EVAL_DOCS_QUERIES) \
		--arms full4096,trunc2048,reembed:nvidia/nemotron-3.5-embed-1b@2048,hybrid,hybrid-rerank@24,dense-rerank@24 \
		--label public-docs

# Production pipeline: dense SQL vs RRF vs rerank depth, plus rerank without
# the lexical branch. Reproduces the decision table in ADR 0003.
eval-pipeline:
	POSTGRES__URL="$(EVAL_DB)" uv run python scripts/eval_retrieval.py run \
		--queries specs/eval/queries-atlas-repo.jsonl \
		--arms full4096,hybrid,hybrid-rerank@50,hybrid-rerank@24,hybrid-rerank@12,dense-rerank@24 \
		--label hybrid
