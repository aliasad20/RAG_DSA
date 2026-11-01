"""Question -> retrieved chunks -> grounded LLM answer with timestamp citations."""

import re

from ytrag.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    MIN_SCORE,
    REFUSAL,
    TOP_K,
)
from ytrag.models import Chunk
from ytrag.store import search, search_vector

SYSTEM_PROMPT = f"""Answer the question using ONLY the numbered transcript excerpts provided.
The transcripts are auto-generated, so read past obvious transcription mistakes.
- If the excerpts don't contain the answer, reply exactly: "{REFUSAL}"
- Cite excerpts inline with plain square brackets: [1], [2]. Every claim needs a citation.
- Reply in the same language and script as the question (a Hinglish question written in English letters gets a Hinglish answer in English letters).
- Keep it concise (under ~6 sentences). Never invent timestamps or video titles."""


# Models also write 【2】, 【2†L1】 and [1, 2]. A plain [n] straight after a word is code (arr[1]).
_CITATION_FORMS = re.compile(r"\s*(?:【|(?<![\w)])\[)\s*(\d+(?:\s*[,，]\s*\d+)*)(?:†[^\]】]*)?\s*[\]】]")
_CITATION = re.compile(r"(?<![\w)])\[(\d+)\]")


def _normalize_citations(text: str) -> str:
    return _CITATION_FORMS.sub(lambda m: " " + "".join(f"[{n}]" for n in re.findall(r"\d+", m[1])), text).strip()


def retrieve(
    question: str, top_k: int = TOP_K, video_id: str | None = None, vector: list[float] | None = None
) -> list[tuple[Chunk, float]]:
    hits = search_vector(vector, top_k, video_id) if vector else search(question, top_k, video_id)
    return [(c, s) for c, s in hits if s >= MIN_SCORE]


def _llm(system: str, user: str) -> str:
    if GROQ_API_KEY:
        from groq import Groq
        r = Groq(api_key=GROQ_API_KEY).chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        return (r.choices[0].message.content or "").strip()
    if GEMINI_API_KEY:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)  # kept in a variable: a dropped client closes itself
        r = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0.2),
        )
        return (r.text or "").strip()
    raise RuntimeError("Set GROQ_API_KEY or GEMINI_API_KEY in .env to use `ask` (or use `search`).")


def ask(question: str, top_k: int = TOP_K, video_id: str | None = None, vector: list[float] | None = None) -> dict:
    hits = retrieve(question, top_k, video_id, vector)
    # Nothing relevant retrieved: refuse without calling the LLM, so it can't
    # answer from its own knowledge and attach unrelated timestamps.
    if not hits:
        return {"answer": REFUSAL, "sources": []}

    context = "\n\n".join(f'[{i}] "{c.title}" @ {c.timestamp}\n{c.text}' for i, (c, _) in enumerate(hits, 1))
    text = _llm(SYSTEM_PROMPT, f"EXCERPTS\n{context}\n\nQUESTION: {question}")
    if REFUSAL.lower() in text.lower():
        return {"answer": REFUSAL, "sources": []}

    # Keep only the sources the model actually cited, renumbered 1..N.
    text = _normalize_citations(text)
    used = list(dict.fromkeys(int(n) for n in _CITATION.findall(text) if 1 <= int(n) <= len(hits)))
    # No valid citation means nothing ties the answer to the lectures, so don't show it.
    if not used:
        return {"answer": REFUSAL, "sources": []}
    remap = {old: new for new, old in enumerate(used, 1)}
    text = _CITATION.sub(lambda m: f"[{remap[int(m[1])]}]" if int(m[1]) in remap else "", text)
    sources = [{"title": hits[i - 1][0].title, "timestamp": hits[i - 1][0].timestamp, "url": hits[i - 1][0].url} for i in used]
    return {"answer": text, "sources": sources}
