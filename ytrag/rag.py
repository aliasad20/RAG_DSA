"""Question -> retrieved chunks -> grounded LLM answer with timestamp citations."""

import re
from collections.abc import Iterator

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
# End of a reply still being written: a citation may be cut off inside its brackets.
_UNFINISHED = re.compile(r"\s*(?:[\[【][^\]】]*)?$")


def _normalize_citations(text: str) -> str:
    return _CITATION_FORMS.sub(lambda m: " " + "".join(f"[{n}]" for n in re.findall(r"\d+", m[1])), text).strip()


def retrieve(
    question: str, top_k: int = TOP_K, video_id: str | None = None, vector: list[float] | None = None
) -> list[tuple[Chunk, float]]:
    hits = search_vector(vector, top_k, video_id) if vector else search(question, top_k, video_id)
    return [(c, s) for c, s in hits if s >= MIN_SCORE]


def _llm_stream(system: str, user: str) -> Iterator[str]:
    if GROQ_API_KEY:
        from groq import Groq
        stream = Groq(api_key=GROQ_API_KEY).chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            stream=True,
        )
        for part in stream:
            if part.choices and part.choices[0].delta.content:
                yield part.choices[0].delta.content
        return
    if GEMINI_API_KEY:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)  # kept in a variable: a dropped client closes itself
        stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=user,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0.2),
        )
        for part in stream:
            if part.text:
                yield part.text
        return
    raise RuntimeError("Set GROQ_API_KEY or GEMINI_API_KEY in .env to use `ask` (or use `search`).")


def _llm(system: str, user: str) -> str:
    return "".join(_llm_stream(system, user)).strip()


def _user_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    context = "\n\n".join(f'[{i}] "{c.title}" @ {c.timestamp}\n{c.text}' for i, (c, _) in enumerate(hits, 1))
    return f"EXCERPTS\n{context}\n\nQUESTION: {question}"


def _cite(text: str, hits: list[tuple[Chunk, float]]) -> tuple[str, list[dict]]:
    """Keep only the sources the model actually cited, renumbered 1..N by first use."""
    text = _normalize_citations(text)
    used = list(dict.fromkeys(int(n) for n in _CITATION.findall(text) if 1 <= int(n) <= len(hits)))
    remap = {old: new for new, old in enumerate(used, 1)}
    text = _CITATION.sub(lambda m: f"[{remap[int(m[1])]}]" if int(m[1]) in remap else "", text)
    sources = [{"title": hits[i - 1][0].title, "timestamp": hits[i - 1][0].timestamp, "url": hits[i - 1][0].url} for i in used]
    return text, sources


def _final_answer(reply: str, hits: list[tuple[Chunk, float]]) -> dict:
    if REFUSAL.lower() in reply.lower():
        return {"answer": REFUSAL, "sources": []}
    text, sources = _cite(reply, hits)
    # No valid citation means nothing ties the answer to the lectures, so don't show it.
    return {"answer": text, "sources": sources} if sources else {"answer": REFUSAL, "sources": []}


def ask(question: str, top_k: int = TOP_K, video_id: str | None = None, vector: list[float] | None = None) -> dict:
    hits = retrieve(question, top_k, video_id, vector)
    # Nothing relevant retrieved: refuse without calling the LLM, so it can't
    # answer from its own knowledge and attach unrelated timestamps.
    if not hits:
        return {"answer": REFUSAL, "sources": []}
    return _final_answer(_llm(SYSTEM_PROMPT, _user_prompt(question, hits)), hits)


def ask_stream(question: str, top_k: int = TOP_K, video_id: str | None = None, vector: list[float] | None = None) -> Iterator[dict]:
    """Yields {"text", "sources"?} as the answer grows, then {"done", "answer", "sources"}.

    Nothing is sent before the first valid citation, so a refusal or an uncited
    reply never appears on screen. The last event is the checked answer.
    """
    hits = retrieve(question, top_k, video_id, vector)
    if not hits:
        yield {"done": True, "answer": REFUSAL, "sources": []}
        return
    reply, sent, sent_sources = "", "", 0
    for piece in _llm_stream(SYSTEM_PROMPT, _user_prompt(question, hits)):
        reply += piece
        text, sources = _cite(_UNFINISHED.sub("", reply), hits)
        if not sources or not text.startswith(sent) or len(text) == len(sent) or REFUSAL.lower() in reply.lower():
            continue
        event = {"text": text[len(sent):]}
        if len(sources) > sent_sources:
            event["sources"] = sources
        yield event
        sent, sent_sources = text, len(sources)
    yield {"done": True, **_final_answer(reply, hits)}
