"""Search data for the static website, so visitors' browsers can search without a server.

vectors.bin holds one row of EMBED_DIM int8 values per chunk (a normalised vector
times 127). meta.json holds the lecture list plus each chunk's lecture, time and
a short preview. Full transcript text is deliberately left out: it stays private.
"""

import json

import numpy as np

from ytrag.chunk import chunk_transcript
from ytrag.config import EMBED_MODEL, MIN_SCORE, PROJECT_DIR
from ytrag.extract import cached_transcripts
from ytrag.store import embed

OUT_DIR = PROJECT_DIR / "frontend" / "index-data"
PREVIEW_CHARS = 180


def quantize(vectors: np.ndarray) -> np.ndarray:
    return np.clip(np.round(vectors * 127), -127, 127).astype(np.int8)


def export() -> tuple[int, int, int]:
    transcripts = cached_transcripts()
    videos, rows, texts = [], [], []
    for t in transcripts:
        chunks = chunk_transcript(t)
        if not chunks:
            continue
        for c in chunks:
            body = c.text.split("\n\n", 1)[-1]
            rows.append([len(videos), c.link_seconds, body[:PREVIEW_CHARS]])
            texts.append(c.text)
        videos.append({"id": t["video_id"], "title": t["title"]})

    vectors = quantize(np.asarray(embed(texts), dtype=np.float32))
    meta = {"model": EMBED_MODEL, "dim": vectors.shape[1], "min_score": MIN_SCORE, "videos": videos, "chunks": rows}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "vectors.bin").write_bytes(vectors.tobytes())
    meta_path = OUT_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return len(videos), len(rows), vectors.size + meta_path.stat().st_size
