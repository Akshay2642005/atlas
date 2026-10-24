# Atlas — AI Agents

Read CONVENTIONS.md before any GitHub or git operation.

<!-- BEGIN bigpowers:project -->

## Project
Atlas — strictly-typed, DI-first enterprise RAG platform (Groq + NVIDIA + pgVector)
Stack: Python 3.12, FastAPI, SQLAlchemy, Pydantic, Dishka (Frontend TBD)

## Commands

| Action | Command |
|--------|---------|
| Run    | `uv run atlas-api` |
| Test   | `uv run pytest tests/` |
| Build  | (none) |
| Lint   | `uv run ruff check src/` |
| Format | `uv run ruff format src/` |
| Type Check | `uv run mypy src/` |
| Preflight | `uv run pytest tests/ && uv run ruff check src/ && uv run mypy src/` |
| CI     | `gh pr checks` (when a PR is open) |

## Architecture
Modular RAG platform with hexagonal ports: atlas_core (config, domain, security), atlas_llm (Groq, NVIDIA), atlas_vector (pgVector), atlas_ingest (pipeline), atlas_retrieval (answerer), atlas_api (FastAPI), atlas_worker (jobs). DI container wires all providers.

## Conventions
- Package structure: src/ with subdirectories (atlas_core, atlas_llm, atlas_vector, atlas_ingest, atlas_retrieval, atlas_api, atlas_worker)
- Line length: 100 characters
- Strict type checking with mypy (Python 3.12)
- Use Dishka for dependency injection
- All planning documents go to specs/

## Never
- Never commit secrets or API keys
- Never modify .env file with real credentials
- Never change the DI container structure without understanding the dependency graph
- Never dismiss reproducible gate failures as pre-existing or out of scope
- Never proceed on red Preflight or red CI — invoke quick-fix or fix-bug first

## Agent Rules
- **Workflow Mandate:** You MUST use the bigpowers skills (e.g. `plan-work`, `develop-tdd`, `orchestrate-project`) to perform tasks. DO NOT write code directly in response to a user prompt like "build this feature".
- **Always Green:** Preflight and CI must be green before forward work. Reproducible gate failures require **fix-or-log** (quick-fix → fix-bug) per CONVENTIONS § Discovered Defects.
- Read specs/ before writing code.
- All planning and specifications MUST be written to `specs/` (`product/SCOPE_LATEST.yaml`, `release-plan.yaml`, `epics/`) before any code is generated.
- Write the minimum code that solves the stated problem. Nothing extra.
- Run tests after every change. Show evidence before declaring done.
- One clarifying question beats a wrong assumption baked into 200 lines.

<!-- END bigpowers:project -->

<!-- BEGIN bigpowers:context-routing -->

## Context Routing (Glob → Sub-AGENTS.md)

| Glob pattern | Sub-AGENTS.md location |
|-------------|------------------------|
| (none configured) | — |

<!-- END bigpowers:context-routing -->

<!-- BEGIN bigpowers:learned-preferences -->

## Learned User Preferences
(Empty - populated by session-state)

## Workspace Facts
(Empty - populated by session-state)

<!-- END bigpowers:learned-preferences -->
