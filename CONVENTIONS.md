# Atlas Conventions

Shared rules for all AI agents working on this project.

## Always Green / Shift Left

**1-10-100 Rule:** Fix a defect in tests (1), locally (10), or production (100). The cost grows exponentially. Shift left: catch issues early via Preflight and CI.

**Preflight:** MUST run before any commit. If red, stop. Fix it or log it (quick-fix → fix-bug). Never commit on red.

**CI:** MUST pass before merging. If red, stop. The same fix-or-log ladder applies.

**Green definition:** All tests pass, lint clean, type check passes, build succeeds. No warnings or errors.

## Discovered Defects

When a gate fails (Preflight or CI) and the defect was not introduced by your changes:

1. **Quick-fix path:** If the fix is trivial (data-only, no logic risk), use `quick-fix`. No TDD, no branch ceremony.
2. **Fix-bug path:** If the fix requires logic changes, use `fix-bug`. Full TDD cycle with investigation → develop → validate.
3. **Log it:** If you cannot fix it now, create a `specs/bugs/BUG-*.md` file and register it in `specs/bugs/registry.yaml`. Do not dismiss it.

**Separate commits:** Discovered fixes MUST be separate commits from feature work. Use Conventional Commits: `fix(scope): description`.

## Banned Dismissive Phrases

| Phrase | Why banned |
|--------|------------|
| "pre-existing" | Doesn't matter — it blocks you now |
| "out of scope" | If it blocks your work, it's in scope |
| "not introduced by my changes" | Doesn't matter — it blocks you now |
| "I'll fix it later" | Later never comes; fix or log now |
| "ignoring this for now" | Never ignore red gates |

## Specs Convention

All planning and specifications MUST be written to `specs/` at the project root:

- `specs/product/SCOPE_LATEST.yaml` — What's in/out of scope
- `specs/product/VISION_LATEST.yaml` — Product vision
- `specs/product/GLOSSARY_LATEST.yaml` — Domain language
- `specs/release-plan.yaml` — Release index and epic list
- `specs/epics/eNN-*/` — Epic capsules with stories and tasks
- `specs/tech-architecture/tech-stack.md` — Technology stack
- `specs/tech-architecture/TEST_PLAN_LATEST.md` — Test architecture
- `specs/tech-architecture/SECURITY_PLAN_LATEST.md` — Security plan
- `specs/bugs/BUG-*.md` — Bug investigations
- `specs/bugs/registry.yaml` — Bug registry

Never write planning documents outside `specs/`.

## Git Conventions

**Branch protection:** Main/master is protected. Work on feature branches via `kickoff-branch`.

**Commit messages:** MUST follow Conventional Commits: `type(scope): description`. Use `commit-message` skill to draft messages.

**Never force push** unless explicitly requested by the user. Use `release-branch` for merging.

## Defensive Code Categories

Atlas implements these defensive patterns:

- **Rate limiting:** Token bucket algorithm per tenant (Redis-backed)
- **Retry:** Provider fallback with exponential backoff (tenacity)
- **Timeout:** Request timeout for all LLM and embedding calls
- **Graceful degradation:** Fallback to secondary providers on failure

When adding new external calls, follow these patterns:
- Add timeout configuration
- Add retry logic with backoff
- Add rate limiting where appropriate
- Add circuit breaker for external dependencies

## Testing Conventions

**TDD preferred:** Use `develop-tdd` for feature work. Red-green-refactor cycle.

**Test coverage:** All new code must have tests. Aim for >80% coverage.

**Async tests:** Use pytest-asyncio with `asyncio_mode = "auto"`.

**Test naming:** Use descriptive test names that explain what is being tested.

## Type Safety

**Strict mode:** Use mypy strict mode. All new code must type-check.

**Pydantic models:** Use Pydantic for all configuration and data models.

**Protocol interfaces:** Use Protocol for ports (hexagonal architecture).

## Dependency Injection

**Dishka container:** All providers MUST be registered in the DI container.

**Never import concrete classes:** Import only ports (Protocol interfaces) in application code.

**Test ports:** Implement test ports for unit testing.

## Code Style

**Line length:** 100 characters (configured in ruff).

**Import order:** Use isort with known-first-party packages configured.

**Formatting:** Use ruff format before committing.

## Security

**Never commit secrets:** API keys, passwords, tokens go in .env only (never in git).

**PII handling:** Use Presidio for PII detection in ingestion pipeline.

**Tenant isolation:** All queries MUST include tenant_id filter.

## Documentation

**Docstrings:** Add docstrings to all public functions and classes.

**Type hints:** Use type hints for all function parameters and return values.

**Comments:** Only add comments to explain "why", not "what".
