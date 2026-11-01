"""YouTube URL -> audio (yt-dlp) -> timestamped transcript (faster-whisper).

Transcripts are cached as JSON in data/transcripts/<video_id>.json, so a
video is only ever transcribed once unless --force is passed.
"""

import json
import os
from pathlib import Path

import yt_dlp

from ytrag.config import (
    AUDIO_DIR,
    TRANSCRIPT_DIR,
    WHISPER_BATCH_SIZE,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
)
from ytrag.models import Segment, Video, hms

_YDL_QUIET = {"quiet": True, "no_warnings": True, "noprogress": True}


# ── Listing ──────────────────────────────────────────────────────────────

def list_videos(url: str) -> list[Video]:
    """Accepts a single video, a playlist, or a channel URL."""
    opts = {**_YDL_QUIET, "extract_flat": "in_playlist", "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    entries = info.get("entries") if info.get("_type") == "playlist" else [info]
    return [
        Video(e["id"], e.get("title") or e["id"], int(e.get("duration") or 0))
        for e in entries or []
        if e and e.get("id")
    ]


# ── Audio ────────────────────────────────────────────────────────────────

def _audio_file(video_id: str) -> Path | None:
    for p in AUDIO_DIR.glob(f"{video_id}.*"):
        if p.suffix not in {".part", ".ytdl", ".tmp"}:
            return p
    return None


def download_audio(video: Video) -> Path:
    """Best audio stream only. faster-whisper decodes m4a/webm itself, so no ffmpeg step."""
    if existing := _audio_file(video.video_id):
        return existing
    opts = {
        **_YDL_QUIET,
        "format": "bestaudio/best",
        "outtmpl": str(AUDIO_DIR / f"{video.video_id}.%(ext)s"),
        "retries": 5,
        "sleep_interval": 2,
        "max_sleep_interval": 6,  # be polite on long playlists to avoid bot checks
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([video.url])
    path = _audio_file(video.video_id)
    if path is None:
        raise RuntimeError(f"Download finished but no audio file found for {video.video_id}")
    return path


# ── Whisper ──────────────────────────────────────────────────────────────

_whisper = None


def _load_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        device = WHISPER_DEVICE
        if device == "auto":
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        try:
            model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)
        except Exception as exc:  # missing CUDA libs etc.
            if device == "cpu":
                raise
            print(f"[whisper] GPU load failed ({exc}); using CPU.")
            model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        _whisper = BatchedInferencePipeline(model=model)
    return _whisper


def transcript_path(video_id: str) -> Path:
    return TRANSCRIPT_DIR / f"{video_id}.json"


def load_transcript(video_id: str) -> dict | None:
    path = transcript_path(video_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if "segments" in data else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def transcribe(video: Video, force: bool = False, keep_audio: bool = False) -> dict:
    """Return the transcript dict for a video, transcribing only if not cached."""
    if not force and (cached := load_transcript(video.video_id)):
        return cached

    audio = download_audio(video)
    # vad_filter skips silence/music, which is where Whisper tends to hallucinate.
    raw, info = _load_whisper().transcribe(
        str(audio),
        language=WHISPER_LANGUAGE,
        batch_size=WHISPER_BATCH_SIZE,
        vad_filter=True,
    )
    segments = [Segment(s.start, s.end, s.text.strip()) for s in raw if s.text.strip()]

    data = {
        "video_id": video.video_id,
        "title": video.title,
        "url": video.url,
        "duration": video.duration or int(info.duration),
        "language": info.language,
        "model": WHISPER_MODEL,
        "segments": [vars(s) for s in segments],
    }
    # Write to a temp file then rename, so an interrupted run never leaves a
    # half-written transcript that later looks like a valid cache hit.
    tmp = transcript_path(video.video_id).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, transcript_path(video.video_id))

    # Also write a human-readable .txt next to the JSON.
    txt = transcript_path(video.video_id).with_suffix(".txt")
    txt.write_text(
        f"{video.title}\n{video.url}\n\n" + "\n".join(f"[{hms(s.start)}] {s.text}" for s in segments),
        encoding="utf-8",
    )

    if not keep_audio:
        audio.unlink(missing_ok=True)
    return data


def cached_transcripts() -> list[dict]:
    return [t for p in sorted(TRANSCRIPT_DIR.glob("*.json")) if (t := load_transcript(p.stem))]
