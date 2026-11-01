"""Transcript segments -> overlapping time-window chunks.

Chunking is done on the segment list (time domain) rather than on a flat
string, so every chunk keeps a start time for its YouTube link.

The batched Whisper pipeline returns segments of about 29 s, too long to overlap
inside a 60 s window, so they are first split into pieces of a few seconds. Whisper
gives no times inside a segment; piece times are estimated from word counts.
"""

import re
from collections import Counter

from ytrag.config import CHUNK_OVERLAP_SECONDS, CHUNK_SECONDS, MIN_CHUNK_WORDS, SUBSEGMENT_SECONDS
from ytrag.models import Chunk, Segment

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _looks_like_hallucination(text: str) -> bool:
    """Whisper sometimes loops one phrase over silence; such chunks match everything."""
    parts = [p.strip().lower() for p in text.replace("?", ".").replace("!", ".").split(".") if p.strip()]
    if len(parts) < 3:
        return False
    return Counter(parts).most_common(1)[0][1] / len(parts) >= 0.6


def _split_segment(seg: Segment) -> list[Segment]:
    """Split at sentence ends, then by word count, sharing the time out by words."""
    total_words = len(seg.text.split())
    duration = seg.end - seg.start
    if duration <= SUBSEGMENT_SECONDS or total_words < 2:
        return [seg]

    seconds_per_word = duration / total_words
    max_words = max(1, int(SUBSEGMENT_SECONDS / seconds_per_word))
    pieces, cursor = [], seg.start
    for sentence in _SENTENCE_END.split(seg.text):
        words = sentence.split()
        for k in range(0, len(words), max_words):
            part = words[k : k + max_words]
            end = cursor + len(part) * seconds_per_word
            pieces.append(Segment(cursor, end, " ".join(part)))
            cursor = end
    pieces[-1].end = seg.end  # no rounding drift past the real segment end
    return pieces


def chunk_transcript(transcript: dict) -> list[Chunk]:
    segs = [
        piece
        for s in transcript["segments"]
        if s["text"].strip()
        for piece in _split_segment(Segment(**s))
    ]
    title, vid = transcript["title"], transcript["video_id"]
    chunks: list[Chunk] = []

    i = 0
    while i < len(segs):
        start = segs[i].start
        j = i
        while j + 1 < len(segs) and segs[j + 1].end - start <= CHUNK_SECONDS:
            j += 1
        window = segs[i : j + 1]
        body = " ".join(s.text for s in window)

        if len(body.split()) >= MIN_CHUNK_WORDS and not _looks_like_hallucination(body):
            # Prefixing the title helps retrieval: mid-video text rarely names its topic.
            chunks.append(Chunk(vid, title, int(window[0].start), int(window[-1].end), f"{title}\n\n{body}"))

        if j == len(segs) - 1:
            break
        # Start the next window at the first segment inside the overlap region,
        # always moving forward by at least one segment.
        overlap_from = segs[j].end - CHUNK_OVERLAP_SECONDS
        nxt = next((k for k in range(i + 1, j + 1) if segs[k].start >= overlap_from), j + 1)
        i = max(nxt, i + 1)

    return chunks
