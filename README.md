# YouTube Lecture RAG

Search what is said across a YouTube playlist and jump to the moment an idea is explained, or get a short written answer that cites those moments. Built over a 315-video, 114-hour DSA course.

The approach follows the public walkthrough ["I converted my 60 Hour DSA Playlist to a RAG!"](https://youtu.be/Z9CI_OVhXNs) by Padho with Pratyush. This is a separate implementation for a different playlist.

## How it works

- **Offline, on a GPU machine:** yt-dlp downloads audio → Whisper large-v3 transcribes it with timestamps → transcripts are cut into 60 s overlapping chunks → all-MiniLM-L6-v2 embeds them.
- **Search, in the visitor's browser:** the same model runs in the page through Transformers.js and compares the question against a 3.5 MB vector file. No server is involved.
- **Written answers, on a small server:** the browser sends the question and its embedding. FastAPI finds the closest chunks in Qdrant and streams an answer from an LLM on Groq that may use only those chunks. Nothing is shown until the first valid citation, and an answer without one is refused.

The server holds the API keys and needs no AI model, so it fits a free host at about 115 MB of memory.

## Results

On the 68-question test set in `eval/`:

| Metric | Result |
|---|---|
| Recall@5 on the website, with at most 2 results per lecture | 95.6% |
| Recall@5 without that limit | 91.2% |
| MRR@10 | 0.86 |
| Off-topic questions refused at score floor 0.40 | 12 of 12 |
| Real questions still answered at that floor | 100% |
| Search time in the browser | about 25 ms |

The set is small, labels are per video rather than per timestamp, and the score floor and per-lecture limit were chosen on these same questions, so treat these as optimistic. The earlier, much larger bge-m3 model scored 91.2% Recall@5 and 0.81 MRR on the same set.

## Run it locally

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env

.venv/bin/python -m ytrag.cli extract "PLAYLIST_URL"   # add --shard 0/2 and --shard 1/2 to split across GPUs
.venv/bin/python -m ytrag.cli index
.venv/bin/python -m ytrag.cli export-static            # search data for the website
.venv/bin/python -m ytrag.cli evaluate
.venv/bin/python -m unittest discover tests

.venv/bin/python -m uvicorn ytrag.api:app --port 8000  # answer server
python -m http.server 8080 --directory frontend        # website at http://127.0.0.1:8080
```

## Host it for free

**Answer server on Render**

1. Put this project in a GitHub repository.
2. In [Render](https://dashboard.render.com), choose **New → Blueprint** and select the repository. It reads `render.yaml`.
3. Fill in `QDRANT_URL`, `QDRANT_API_KEY` and `GROQ_API_KEY`, and set `ALLOWED_ORIGINS` to the website's address, such as `https://your-project.netlify.app`.
4. Check that `https://YOUR-SERVICE.onrender.com/api/ping` shows the chunk count.
5. In the GitHub repository's **Settings → Secrets and variables → Actions → Variables**, add `ANSWER_SERVER_URL` with the Render address. A weekly job then touches the database, because free Qdrant clusters are suspended after a week without use.

**Website on Netlify**

1. Set `apiBase` in `frontend/config.js` to the Render address and push.
2. In [Netlify](https://app.netlify.com), choose **Add new project → Import an existing project** and select the repository. It reads `netlify.toml`, which publishes `frontend/` with no build step.

The free Render server sleeps after 15 minutes without traffic and takes about a minute to wake. Search keeps working meanwhile. Netlify's free plan pauses all of an account's projects once its monthly credits run out.

## Content

The lectures belong to their creator. The website links back to the original videos, and its search data holds only short previews. Full transcripts stay private.
