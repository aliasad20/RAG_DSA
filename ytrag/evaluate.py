"""Measures retrieval against eval/golden.tsv and eval/offtopic.txt.

A hit means a retrieved chunk comes from one of the question's accepted videos.
Labels are at video level, so this says nothing about timestamp precision.
"""

import json
import statistics
import time

from ytrag.config import MIN_SCORE, PROJECT_DIR
from ytrag.store import search

EVAL_DIR = PROJECT_DIR / "eval"
K_VALUES = (1, 3, 5, 10)
THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


def load_golden() -> list[tuple[str, set[str]]]:
    rows = []
    for line in (EVAL_DIR / "golden.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            question, ids = line.split("\t")
            rows.append((question, set(ids.split(","))))
    return rows


def load_offtopic() -> list[str]:
    return [q for q in (EVAL_DIR / "offtopic.txt").read_text(encoding="utf-8").splitlines() if q.strip()]


def first_hit_rank(video_ids: list[str], accepted: set[str]) -> int | None:
    """1-based rank of the first result from an accepted video, or None."""
    return next((i for i, v in enumerate(video_ids, 1) if v in accepted), None)


def threshold_sweep(in_scores: list[float], off_scores: list[float]) -> list[dict]:
    """What each score floor would do, judged on each question's best score."""
    return [
        {
            "threshold": t,
            "in_domain_answered": sum(s >= t for s in in_scores) / len(in_scores),
            "off_topic_refused": sum(s < t for s in off_scores) / len(off_scores),
        }
        for t in THRESHOLDS
    ]


def run() -> dict:
    golden, offtopic = load_golden(), load_offtopic()
    search("warm up the embedding model", 1)  # so model loading isn't counted as latency

    ranks, in_scores, misses, latencies = [], [], [], []
    for question, accepted in golden:
        start = time.perf_counter()
        hits = search(question, max(K_VALUES))
        latencies.append(time.perf_counter() - start)
        rank = first_hit_rank([c.video_id for c, _ in hits], accepted)
        ranks.append(rank)
        in_scores.append(hits[0][1])
        if rank is None or rank > 5:
            misses.append({"question": question, "rank": rank, "top_result": hits[0][0].title})

    off_scores = [search(q, 1)[0][1] for q in offtopic]

    return {
        "questions": len(golden),
        "off_topic_questions": len(offtopic),
        "recall_at_k": {k: sum(r is not None and r <= k for r in ranks) / len(ranks) for k in K_VALUES},
        "mrr_at_10": sum(1 / r for r in ranks if r) / len(ranks),
        "median_search_ms": statistics.median(latencies) * 1000,
        "in_domain_top_score": {"min": min(in_scores), "median": statistics.median(in_scores), "max": max(in_scores)},
        "off_topic_top_score": {"min": min(off_scores), "median": statistics.median(off_scores), "max": max(off_scores)},
        "current_threshold": MIN_SCORE,
        "threshold_sweep": threshold_sweep(in_scores, off_scores),
        "misses_at_5": misses,
    }


def save(results: dict) -> None:
    (EVAL_DIR / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
