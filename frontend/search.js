// Semantic search in the visitor's browser: a small model plus a downloaded vector file.

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";
const BROWSER_MODEL = "Xenova/all-MiniLM-L6-v2"; // same model as the index, converted for browsers
const CANDIDATES_PER_RESULT = 6;

let ready = null;

async function fetchChecked(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path}`);
  return response;
}

async function load(onProgress) {
  const [{ pipeline, env }, meta, vectorBytes] = await Promise.all([
    import(TRANSFORMERS_URL),
    fetchChecked("index-data/meta.json").then((r) => r.json()),
    fetchChecked("index-data/vectors.bin").then((r) => r.arrayBuffer()),
  ]);
  env.allowLocalModels = false;

  const files = {};
  const embedder = await pipeline("feature-extraction", BROWSER_MODEL, {
    dtype: "q8",
    progress_callback: (p) => {
      if (p.status !== "progress") return;
      files[p.file] = { loaded: p.loaded, total: p.total };
      const parts = Object.values(files);
      const total = parts.reduce((sum, f) => sum + f.total, 0);
      onProgress(parts.reduce((sum, f) => sum + f.loaded, 0) / total);
    },
  });
  return { embedder, meta, vectors: new Int8Array(vectorBytes) };
}

export function prepare(onProgress = () => {}) {
  if (!ready) {
    ready = load(onProgress).catch((error) => {
      ready = null; // let the next search try again
      throw error;
    });
  }
  return ready;
}

export function formatTime(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = String(seconds % 60).padStart(2, "0");
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${secs}` : `${minutes}:${secs}`;
}

export function toMoment(videoId, title, start, preview = "") {
  return {
    key: `${videoId}:${start}`,
    video_id: videoId,
    title,
    start,
    preview,
    timestamp: formatTime(start),
    url: `https://www.youtube.com/watch?v=${videoId}&t=${start}s`,
  };
}

export async function embedQuestion(text) {
  const { embedder } = await prepare();
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

// Best chunks above the score floor. perLecture stops one video filling the whole list.
export async function search(text, { limit = 6, perLecture = 2 } = {}) {
  const { meta, vectors } = await prepare();
  const query = await embedQuestion(text);
  const dim = meta.dim;
  const count = vectors.length / dim;

  const scored = [];
  for (let i = 0; i < count; i++) {
    let dot = 0;
    const offset = i * dim;
    for (let d = 0; d < dim; d++) dot += query[d] * vectors[offset + d];
    const score = dot / 127;
    if (score >= meta.min_score) scored.push([score, i]);
  }
  scored.sort((a, b) => b[0] - a[0]);

  const shown = new Map();
  const results = [];
  for (const [score, i] of scored.slice(0, limit * CANDIDATES_PER_RESULT)) {
    const [videoIndex, start, preview] = meta.chunks[i];
    const used = shown.get(videoIndex) || 0;
    if (used >= perLecture) continue;
    shown.set(videoIndex, used + 1);
    const video = meta.videos[videoIndex];
    results.push({ ...toMoment(video.id, video.title, start, preview), score });
    if (results.length === limit) break;
  }
  return results;
}
