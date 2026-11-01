"""Embeddings (sentence-transformers) + local vector store (Qdrant on disk)."""

from qdrant_client import QdrantClient, models

from ytrag.config import COLLECTION, EMBED_MODEL, QDRANT_API_KEY, QDRANT_DIR, QDRANT_URL
from ytrag.models import Chunk

_embedder = None
_client = None


def embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def client() -> QdrantClient:
    global _client
    if _client is None:
        # Hosted Qdrant when configured, otherwise the local folder.
        _client = (
            QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, timeout=60)
            if QDRANT_URL
            else QdrantClient(path=str(QDRANT_DIR))
        )
    return _client


def embed(texts: list[str]) -> list[list[float]]:
    return embedder().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False).tolist()


def _ensure_collection(dim: int) -> None:
    c = client()
    if not c.collection_exists(COLLECTION):
        c.create_collection(COLLECTION, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        if QDRANT_URL:  # hosted clusters can require an index on filtered fields; local mode has none
            c.create_payload_index(COLLECTION, "video_id", models.PayloadSchemaType.KEYWORD)


def _video_filter(video_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id))])


def upsert_video(chunks: list[Chunk]) -> int:
    """Replace all chunks of one video (so re-chunking doesn't leave stale points)."""
    if not chunks:
        return 0
    vectors = embed([c.text for c in chunks])
    _ensure_collection(len(vectors[0]))
    c = client()
    c.delete(COLLECTION, points_selector=models.FilterSelector(filter=_video_filter(chunks[0].video_id)))
    c.upsert(
        COLLECTION,
        points=[models.PointStruct(id=ch.point_id, vector=v, payload=ch.to_payload()) for ch, v in zip(chunks, vectors)],
    )
    return len(chunks)


def search(query: str, top_k: int, video_id: str | None = None) -> list[tuple[Chunk, float]]:
    return search_vector(embed([query])[0], top_k, video_id)


def search_vector(vector: list[float], top_k: int, video_id: str | None = None) -> list[tuple[Chunk, float]]:
    """Search with an embedding computed elsewhere: the website's browser sends its own."""
    if not client().collection_exists(COLLECTION):
        return []
    res = client().query_points(
        COLLECTION,
        query=vector,
        limit=top_k,
        query_filter=_video_filter(video_id) if video_id else None,
        with_payload=True,
    )
    return [(Chunk(**p.payload), p.score) for p in res.points]


def chunk_count() -> int:
    return client().count(COLLECTION).count


def indexed_video_ids() -> set[str]:
    if not client().collection_exists(COLLECTION):
        return set()
    ids, offset = set(), None
    while True:
        points, offset = client().scroll(COLLECTION, limit=1000, offset=offset, with_payload=["video_id"])
        ids.update(p.payload["video_id"] for p in points)
        if offset is None:
            return ids
