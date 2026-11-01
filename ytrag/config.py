"""All settings in one place, overridable through .env or environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

DATA_DIR = Path(os.getenv("YTRAG_DATA_DIR", PROJECT_DIR / "data"))
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
QDRANT_DIR = DATA_DIR / "qdrant"
for _d in (AUDIO_DIR, TRANSCRIPT_DIR, QDRANT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Speech-to-text
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE") or None  # None = auto-detect
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")      # auto | cuda | cpu
WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "16"))

# Chunking (in seconds of video, not characters)
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "60"))
CHUNK_OVERLAP_SECONDS = int(os.getenv("CHUNK_OVERLAP_SECONDS", "15"))
MIN_CHUNK_WORDS = int(os.getenv("MIN_CHUNK_WORDS", "12"))
SUBSEGMENT_SECONDS = float(os.getenv("SUBSEGMENT_SECONDS", "8"))  # long Whisper segments are split to this

# Embeddings + vector store
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # small enough to run in a browser
EMBED_DIM = 384
COLLECTION = os.getenv("QDRANT_COLLECTION", "youtube_chunks_minilm")
QDRANT_URL = os.getenv("QDRANT_URL", "")          # blank = local folder in data/qdrant
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.40"))  # cosine floor, chosen from the sweep in eval/results.json
LINK_REWIND_SECONDS = 3  # land the link slightly before the chunk starts

# LLM for grounded answers
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
REFUSAL = "This isn't covered in the indexed videos."
ASK_LIMIT_PER_HOUR = int(os.getenv("ASK_LIMIT_PER_HOUR", "20"))  # per visitor, on the website
ASK_TOTAL_LIMIT_PER_HOUR = int(os.getenv("ASK_TOTAL_LIMIT_PER_HOUR", "200"))  # all visitors together
# Websites allowed to call the answer server from a browser, comma separated.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:8080,http://localhost:8080").split(",")
