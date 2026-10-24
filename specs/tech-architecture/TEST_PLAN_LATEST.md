# Atlas Test Plan

## Test Strategy

Atlas follows TDD (Test-Driven Development) using the `develop-tdd` skill.

## Test Levels

### Unit Tests
- Test individual functions and classes in isolation
- Use pytest fixtures for setup
- Mock external dependencies (LLM providers, database)
- Test ports (Protocol interfaces) with test implementations

### Integration Tests
- Test interaction between modules
- Use test database (PostgreSQL with pgVector)
- Test Redis integration
- Test DI container wiring

### End-to-End Tests
- Test full RAG pipeline
- Test API endpoints
- Test multi-tenant isolation
- Test document ingestion and querying

## Test Coverage

Aim for >80% code coverage across all modules.

## Async Testing

Use pytest-asyncio with `asyncio_mode = "auto"` for async tests.

## Test Data

Use fixtures for test data:
- Sample documents
- Sample queries
- Test tenants

## Continuous Testing

Run tests as part of Preflight:
```bash
uv run pytest tests/
```
