"""Data types passed between pipeline stages."""

import uuid
from dataclasses import asdict, dataclass

from ytrag.config import LINK_REWIND_SECONDS

_ID_NAMESPACE = uuid.UUID("2b1f4a0e-6c55-4f0e-9d3a-7a1c9e5b8d21")


def hms(seconds: float) -> str:
    """65 -> '1:05', 3725 -> '1:02:05'."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


@dataclass
class Video:
    video_id: str
    title: str
    duration: int = 0

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Chunk:
    video_id: str
    title: str
    start: int
    end: int
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.video_id}:{self.start}"

    @property
    def point_id(self) -> str:
        # Deterministic, so re-indexing a video overwrites instead of duplicating.
        return str(uuid.uuid5(_ID_NAMESPACE, self.chunk_id))

    @property
    def link_seconds(self) -> int:
        return max(0, self.start - LINK_REWIND_SECONDS)

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}&t={self.link_seconds}s"

    @property
    def timestamp(self) -> str:
        return hms(self.link_seconds)

    def to_payload(self) -> dict:
        return asdict(self)
