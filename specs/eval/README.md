# Evaluation query sets

One JSON object per line. Query sets are committed so a retrieval change can be
re-scored against the same cases.

```json
{
  "id": "q-1a2b3c4d5e6f",
  "query": "How does Atlas fuse dense and lexical rankings?",
  "tenant_id": "atlas-eval",
  "relevant": [{ "checksum": "<sha256 of the source document>", "seq": 3 }],
  "source_uri": "specs/ATLAS_PRODUCT_SPEC.md",
  "generator_model": "groq:openai/gpt-oss-120b",
  "generated_at": "2026-10-26T13:19:18+00:00"
}
```

## Why checksum + seq

Chunk UUIDs change when a corpus is re-ingested. Ground truth is therefore
stored as the document's SHA-256 checksum plus the chunk's `seq`, and the
harness resolves those to live chunk ids at run time. A case whose references
no longer resolve is skipped and counted in the report rather than scored
against partial ground truth.

## Generating cases

```bash
make eval-queries
```

This samples chunks (deterministic `md5(id || seed)` order) and asks the
configured chat model for one standalone question per chunk. The source chunk is
the sole relevant passage. That makes the judgment strict: a different chunk
that also answers the question counts as a miss.

## Hand-written cases

Append records with the same shape. Multiple `relevant` entries are supported
and use binary gains for nDCG. Keep `tenant_id` pointing at the corpus the case
was written against; `run --tenant` selects the corpus and resolves the
references.
