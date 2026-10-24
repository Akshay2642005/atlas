# Atlas Security Plan

## PII Detection

Atlas uses Presidio for PII detection and anonymization in the ingestion pipeline.

**Supported Entity Types:**
- PERSON (names)
- EMAIL_ADDRESS
- PHONE_NUMBER
- IBAN_CODE
- CREDIT_CARD
- IP_ADDRESS
- LOCATION
- DATE_TIME
- URL
- US_SSN
- UK_NINO
- PASSPORT

**PII Policies:**
- `off`: No detection
- `mask`: Replace with placeholders (e.g., `[NAME]`, `[EMAIL]`)
- `block`: Reject documents with PII
- `anonymize`: Replace with synthetic data

## Tenant Isolation

All data is isolated at the tenant level:

- Row-level security in PostgreSQL
- All queries include `WHERE tenant_id = ?`
- Partition pruning for performance
- API key validation on every request

## Encryption

**At rest:**
- PostgreSQL: AES-256 encryption (managed by cloud provider or LUKS)
- Redis: Optional TLS encryption

**In transit:**
- TLS 1.3 for all API endpoints
- TLS for PostgreSQL connections
- TLS for Redis connections

## Audit Logging

All operations are logged with:
- Timestamp
- Tenant ID
- User ID
- Operation type
- Resource ID
- IP address
- User agent
- Trace ID

## Rate Limiting

Token bucket algorithm per tenant:
- Configurable requests per minute
- Redis-backed for distributed scenarios
- 429 HTTP response on limit exceeded

## Authentication

API key authentication:
- Per-tenant API keys
- Key rotation support
- Revocation capability

## Compliance

Atlas is designed to support:
- SOC 2 Type II (with proper process documentation)
- HIPAA (with BAA)
- GDPR (data residency, right to deletion)
