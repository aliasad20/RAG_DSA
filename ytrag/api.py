"""Answer server for the website. Search itself runs in the visitor's browser.

    .venv/bin/python -m uvicorn ytrag.api:app --port 8000

The browser sends the question plus its embedding, so this server needs no AI
model and fits a small free host. It holds the Groq and Qdrant keys.
"""

import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ytrag.config import (
    ALLOWED_ORIGINS,
    ASK_LIMIT_PER_HOUR,
    ASK_TOTAL_LIMIT_PER_HOUR,
    EMBED_DIM,
    GROQ_API_KEY,
)
from ytrag.rag import ask as rag_ask
from ytrag.store import chunk_count

MAX_QUESTION_CHARS = 300
HOUR = 3600

app = FastAPI(title="Lecture answers")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

_visitor_times: dict[str, deque] = defaultdict(deque)
_all_times: deque = deque()


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    vector: list[float] = Field(min_length=EMBED_DIM, max_length=EMBED_DIM)


def _over_limit(times: deque, now: float, limit: int) -> bool:
    while times and now - times[0] > HOUR:
        times.popleft()
    return len(times) >= limit


def _check_rate_limit(request: Request) -> None:
    """Protects the free LLM quota. The total limit also covers faked visitor addresses."""
    forwarded = request.headers.get("x-forwarded-for", "")
    visitor = forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    now, times = time.monotonic(), _visitor_times[visitor]
    if _over_limit(times, now, ASK_LIMIT_PER_HOUR) or _over_limit(_all_times, now, ASK_TOTAL_LIMIT_PER_HOUR):
        raise HTTPException(429, "Too many questions this hour. Search still works without a limit.")
    times.append(now)
    _all_times.append(now)


@app.get("/api/status")
def status() -> dict:
    return {"answers_available": bool(GROQ_API_KEY)}


@app.get("/api/ping")
def ping() -> dict:
    """Called weekly so the free Qdrant cluster counts as in use."""
    try:
        return {"chunks": chunk_count()}
    except Exception as exc:
        print(f"[ping] failed: {exc}")
        raise HTTPException(502, "Database unavailable.") from exc


@app.post("/api/ask")
def ask(body: AskBody, request: Request) -> dict:
    if not GROQ_API_KEY:
        raise HTTPException(503, "Written answers aren't switched on here. Search still works.")
    _check_rate_limit(request)
    try:
        return rag_ask(body.question, vector=body.vector)
    except Exception as exc:  # provider outage, quota, bad key: never show details to visitors
        print(f"[ask] failed: {exc}")
        raise HTTPException(502, "The answer service is unavailable right now. Search still works.") from exc
