"""Pure retrieval-evaluation logic: cases, arm specs, metrics, and uncertainty.

No I/O lives here. `scripts/eval_retrieval.py` wires these helpers to Postgres,
the embedders, and the chat model so the maths stays testable without network.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

_WS_RE = re.compile(r"\s+")
_BULLET_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s+")
_QUOTES = "\"'“”‘’`"
_ARM_RE = re.compile(r"^reembed:(?P<model>[A-Za-z0-9._/-]+)@(?P<dims>[0-9]+)(?P<ann>:hnsw)?$")
_HYBRID_RE = re.compile(r"^hybrid(?P<rerank>-rerank)?(?:@(?P<depth>[0-9]+))?$")
_DENSE_RERANK_RE = re.compile(r"^dense-rerank(?:@(?P<depth>[0-9]+))?$")


@dataclass(frozen=True, slots=True)
class RelevanceRef:
    """Ground truth that survives re-ingestion: a document checksum + chunk seq."""

    checksum: str
    seq: int


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One judged query. `relevant` is chunk-level ground truth."""

    id: str
    query: str
    relevant: tuple[RelevanceRef, ...]
    tenant_id: str = ""
    source_uri: str = ""
    generator_model: str = ""


@dataclass(frozen=True, slots=True)
class QueryScore:
    recall: float
    mrr: float
    ndcg: float
    hit_at_1: float
    rank: int | None


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    count: int
    recall: float
    mrr: float
    ndcg: float
    hit_at_1: float
    p50_ms: float
    p95_ms: float


@dataclass(frozen=True, slots=True)
class DeltaCI:
    """Paired bootstrap of `arm - baseline`: mean and 95% interval."""

    mean: float
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """One retrieval configuration to evaluate.

    `existing` and `reembed` arms are dense-only SQL. `hybrid`,
    `hybrid-rerank`, and `dense-rerank` run the production retriever (dense +
    lexical RRF, optionally reranked; `dense-rerank` disables the lexical
    branch), so they measure the pipeline users actually hit.
    """

    name: str
    kind: Literal["existing", "reembed", "hybrid", "hybrid-rerank", "dense-rerank"]
    dims: int
    ann: bool = False
    model: str | None = None
    # `<arm>@<n>` overrides the candidate depth sent to the reranker.
    rerank_top_k: int | None = None

    @property
    def is_pipeline(self) -> bool:
        """True when the arm runs `Retriever` instead of a raw SQL query."""
        return self.kind in {"hybrid", "hybrid-rerank", "dense-rerank"}

    @property
    def reranks(self) -> bool:
        """True when the arm is supposed to call the hosted reranker."""
        return self.kind in {"hybrid-rerank", "dense-rerank"}


def truncate_embedding(values: Sequence[float], dims: int) -> list[float]:
    """Matryoshka truncation: keep the leading dims, then restore unit length.

    Nemotron embeddings are L2-normalised, so a raw prefix is shorter than unit
    length. Cosine distance is scale-invariant, but renormalising keeps query
    and passage vectors consistent with how the model was trained.
    """
    if dims <= 0:
        raise ValueError("dims must be positive")
    if dims > len(values):
        raise ValueError(f"cannot truncate {len(values)} dims to {dims}")
    head = list(values[:dims])
    norm = math.sqrt(sum(value * value for value in head))
    if norm == 0.0:
        return head
    return [value / norm for value in head]


def clean_question(raw: str) -> str:
    """Extract a single question from a model response, rejecting chatter."""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty question")
    stripped = [_BULLET_RE.sub("", line).strip().strip(_QUOTES).strip() for line in lines]
    candidates = [line for line in stripped if line.endswith("?")]
    text = candidates[0] if candidates else stripped[0]
    text = _WS_RE.sub(" ", text)
    if not text:
        raise ValueError("empty question")
    return text


def normalize_question(text: str) -> str:
    """Case/punctuation-insensitive key for near-duplicate detection."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def score_ranking(ranked: Sequence[str], relevant: Collection[str], *, k: int) -> QueryScore:
    """Score one ranked id list against a relevant set (binary gains)."""
    if not relevant:
        raise ValueError("relevant set cannot be empty")
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = set(relevant)
    top = list(ranked[:k])
    rank = next(
        (position for position, cid in enumerate(top, start=1) if cid in relevant_set), None
    )
    if rank is None:
        return QueryScore(recall=0.0, mrr=0.0, ndcg=0.0, hit_at_1=0.0, rank=None)

    recall = len(relevant_set.intersection(top)) / len(relevant_set)
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, cid in enumerate(top, start=1)
        if cid in relevant_set
    )
    ideal = sum(
        1.0 / math.log2(position + 1) for position in range(1, min(len(relevant_set), k) + 1)
    )
    return QueryScore(
        recall=recall,
        mrr=1.0 / rank,
        ndcg=dcg / ideal,
        hit_at_1=1.0 if rank == 1 else 0.0,
        rank=rank,
    )


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile; `p` is 0-100."""
    if not values:
        return 0.0
    if not 0.0 <= p <= 100.0:
        raise ValueError("p must be in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (p / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(scores: Sequence[QueryScore], latencies_ms: Sequence[float]) -> ScoreSummary:
    """Mean quality metrics plus latency percentiles for one arm."""
    if not scores:
        raise ValueError("no scores to summarize")
    count = len(scores)
    return ScoreSummary(
        count=count,
        recall=sum(score.recall for score in scores) / count,
        mrr=sum(score.mrr for score in scores) / count,
        ndcg=sum(score.ndcg for score in scores) / count,
        hit_at_1=sum(score.hit_at_1 for score in scores) / count,
        p50_ms=percentile(latencies_ms, 50.0),
        p95_ms=percentile(latencies_ms, 95.0),
    )


def bootstrap_delta(
    baseline: Sequence[float],
    arm: Sequence[float],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> DeltaCI:
    """Paired bootstrap of the mean difference `arm - baseline`.

    Returns the observed mean plus the 2.5th/97.5th percentiles of the resampled
    means. A CI that straddles zero means the difference is not distinguishable
    from noise at this query-set size.
    """
    if len(baseline) != len(arm):
        raise ValueError("paired samples must have equal length")
    if not baseline:
        raise ValueError("no samples")
    differences = [
        arm_value - base_value for base_value, arm_value in zip(baseline, arm, strict=True)
    ]
    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(len(differences)):
            total += differences[rng.randrange(len(differences))]
        means.append(total / len(differences))
    return DeltaCI(mean=observed, low=percentile(means, 2.5), high=percentile(means, 97.5))


def _rerank_depth(raw: str | None, *, arm: str) -> int | None:
    if raw is None:
        return None
    depth = int(raw)
    if depth < 1:
        raise ValueError(f"{arm} depth must be at least 1")
    return depth


def parse_arm(spec: str) -> ArmSpec:
    """Parse a dense arm or a production pipeline arm.

    Pipeline arms are `hybrid`, `hybrid-rerank[@depth]`, and
    `dense-rerank[@depth]` (the last drops the lexical branch, so it measures
    whether fusion earns its keep once reranking is in place).
    """
    compact = spec.strip()
    if compact == "full4096":
        return ArmSpec(name=compact, kind="existing", dims=4096)
    if compact in {"trunc2048", "trunc2048-hnsw"}:
        return ArmSpec(name=compact, kind="existing", dims=2048, ann=compact.endswith("-hnsw"))
    hybrid = _HYBRID_RE.match(compact)
    if hybrid is not None:
        rerank = hybrid.group("rerank") is not None
        raw_depth = hybrid.group("depth")
        if raw_depth is not None and not rerank:
            raise ValueError("only hybrid-rerank accepts a candidate depth, e.g. hybrid-rerank@24")
        return ArmSpec(
            name=compact,
            kind="hybrid-rerank" if rerank else "hybrid",
            dims=0,
            rerank_top_k=_rerank_depth(raw_depth, arm="hybrid-rerank"),
        )
    dense_rerank = _DENSE_RERANK_RE.match(compact)
    if dense_rerank is not None:
        return ArmSpec(
            name=compact,
            kind="dense-rerank",
            dims=0,
            rerank_top_k=_rerank_depth(dense_rerank.group("depth"), arm="dense-rerank"),
        )
    match = _ARM_RE.match(compact)
    if match is None:
        raise ValueError(
            f"unknown arm {spec!r}; expected full4096, trunc2048, trunc2048-hnsw, "
            "hybrid, hybrid-rerank[@<depth>], dense-rerank[@<depth>], "
            "or reembed:<model>@<dims>[:hnsw]"
        )
    dims = int(match.group("dims"))
    if not 1 <= dims <= 16000:
        raise ValueError(f"dims must be between 1 and 16000, got {dims}")
    model = match.group("model")
    ann = match.group("ann") is not None
    if ann and dims > 4000:
        raise ValueError("pgvector caps halfvec HNSW indexes at 4,000 dimensions")
    return ArmSpec(name=compact, kind="reembed", dims=dims, ann=ann, model=model)


def parse_arms(spec: str) -> list[ArmSpec]:
    """Parse a comma-separated arm list, rejecting duplicates and 4096-dim ANN."""
    arms = [parse_arm(part) for part in spec.split(",") if part.strip()]
    if not arms:
        raise ValueError("at least one arm is required")
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        raise ValueError("duplicate arms")
    for arm in arms:
        if arm.ann and arm.dims > 4000:
            raise ValueError(f"arm {arm.name!r} requests ANN above pgvector's 4,000-dim cap")
    return arms


def resolve_case(
    case: EvalCase, index: Mapping[tuple[str, int], str]
) -> tuple[tuple[str, ...], int]:
    """Map checksum+seq references to live chunk ids.

    Returns the resolved ids and how many references were missing. Missing
    references mean the corpus changed under the query set; callers should skip
    the case rather than score it against partial ground truth.
    """
    resolved: list[str] = []
    missing = 0
    for ref in case.relevant:
        chunk_id = index.get((ref.checksum, ref.seq))
        if chunk_id is None:
            missing += 1
        else:
            resolved.append(chunk_id)
    return tuple(resolved), missing
