"""Command line: extract -> index -> search / ask.

    python -m ytrag.cli extract <video|playlist|channel URL>
    python -m ytrag.cli index
    python -m ytrag.cli search "how does binary search work"
    python -m ytrag.cli ask "explain sliding window"
"""

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from ytrag.config import TOP_K, TRANSCRIPT_DIR

app = typer.Typer(add_completion=False, help="YouTube -> transcript -> RAG")
console = Console()


@app.command()
def extract(
    url: str = typer.Argument(..., help="YouTube video, playlist or channel URL"),
    force: bool = typer.Option(False, help="Re-transcribe even if cached"),
    keep_audio: bool = typer.Option(False, help="Keep downloaded audio files"),
    limit: int = typer.Option(0, help="Only process the first N videos (0 = all)"),
    shard: str = typer.Option("0/1", help="i/n: process only every n-th video starting at i, "
                                          "so n workers (e.g. one per GPU) split a playlist"),
):
    """Download audio and transcribe each video with Whisper (cached)."""
    from ytrag.extract import list_videos, load_transcript, transcribe

    shard_i, shard_n = (int(x) for x in shard.split("/"))
    if not 0 <= shard_i < shard_n:
        raise typer.BadParameter("shard must be i/n with 0 <= i < n", param_hint="--shard")

    videos = list_videos(url)
    if limit:
        videos = videos[:limit]
    videos = videos[shard_i::shard_n]
    console.print(f"Found [bold]{len(videos)}[/] video(s).")

    failed = []
    with Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
                  TimeElapsedColumn(), console=console) as bar:
        task = bar.add_task("Transcribing", total=len(videos))
        for v in videos:
            cached = not force and load_transcript(v.video_id) is not None
            bar.update(task, description=f"{'cached ' if cached else ''}{v.title[:50]}")
            try:
                transcribe(v, force=force, keep_audio=keep_audio)
            except Exception as exc:  # keep going; one bad video shouldn't kill a playlist run
                failed.append((v, exc))
                console.print(f"[red]Failed[/] {v.video_id} {v.title}: {exc}")
            bar.advance(task)

    console.print(f"Transcripts in [cyan]{TRANSCRIPT_DIR}[/] ({len(videos) - len(failed)} ok, {len(failed)} failed).")
    if failed:
        console.print("Re-run the same command to retry failed videos; finished ones are skipped.")


@app.command()
def index(all_: bool = typer.Option(False, "--all", help="Re-index videos already in the store")):
    """Chunk cached transcripts, embed them, and store them in Qdrant."""
    from ytrag.chunk import chunk_transcript
    from ytrag.extract import cached_transcripts
    from ytrag.store import indexed_video_ids, upsert_video

    transcripts = cached_transcripts()
    done = set() if all_ else indexed_video_ids()
    todo = [t for t in transcripts if t["video_id"] not in done]
    console.print(f"{len(transcripts)} transcript(s), {len(todo)} to index. Loading embedding model...")

    total = 0
    for t in todo:
        n = upsert_video(chunk_transcript(t))
        total += n
        console.print(f"  {n:4d} chunks  {t['title'][:70]}")
    console.print(f"[green]Indexed {total} chunks.[/]")


@app.command()
def search(query: str, k: int = typer.Option(TOP_K, help="Number of results")):
    """Retrieval only: ranked timestamp links, no LLM needed."""
    from ytrag.store import search as vsearch

    for i, (c, score) in enumerate(vsearch(query, k), 1):
        preview = c.text.split("\n\n", 1)[-1][:200]
        console.print(f"[bold]{i}. {c.title}[/] @ {c.timestamp}  (score {score:.3f})\n   [link={c.url}]{c.url}[/link]\n   {preview}...\n")


@app.command()
def ask(question: str, k: int = typer.Option(TOP_K, help="Chunks to retrieve")):
    """Grounded answer from the transcripts, with timestamp citations."""
    from ytrag.rag import ask as rag_ask

    result = rag_ask(question, k)
    console.print(f"\n{result['answer']}\n")
    for i, s in enumerate(result["sources"], 1):
        console.print(f"[{i}] {s['title']} @ {s['timestamp']}  {s['url']}")


@app.command("export-static")
def export_static():
    """Write the search data the website's browser downloads to frontend/index-data/."""
    from ytrag.export_static import export

    videos, chunks, size = export()
    console.print(f"[green]Exported {chunks} chunks from {videos} videos ({size / 1e6:.1f} MB).[/]")


@app.command()
def evaluate():
    """Measure retrieval on the test questions in eval/ and save eval/results.json."""
    from ytrag.evaluate import run, save

    results = run()
    save(results)
    console.print_json(data=results)


if __name__ == "__main__":
    app()
