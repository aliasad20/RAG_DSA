import { embedQuestion, prepare, search, toMoment } from "./search.js";

const { useEffect, useRef, useState } = React;
const html = htm.bind(React.createElement);

const API_BASE = (window.APP_CONFIG && window.APP_CONFIG.apiBase) || "";
const ASK_TIMEOUT_MS = 90000;
const WAKE_NOTICE_MS = 6000;
const TYPE_INTERVAL_MS = 25;
const TYPE_CHARS = 3;
const TYPE_CATCH_UP = 100; // long answers start faster, so none takes more than a few seconds
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const EXAMPLES = [
  "How does the tortoise and hare method find a cycle?",
  "Difference between memoization and tabulation",
  "Why use a priority queue for shortest paths?",
  "sliding window ka template kya hota hai",
];

const STOP_WORDS = new Set(["what", "when", "where", "which", "does", "from", "that", "this", "with", "have", "kaise", "kyun", "hota", "hain"]);

function highlight(text, query) {
  const terms = query.toLowerCase().match(/[\p{L}\p{N}]{4,}/gu)?.filter((t) => !STOP_WORDS.has(t)) || [];
  if (!terms.length) return text;
  const parts = text.split(new RegExp(`(${terms.join("|")})`, "giu"));
  return parts.map((part, i) => (i % 2 ? html`<mark key=${i}>${part}</mark>` : part));
}

// The LLM writes light markdown and [n] citations. Built as elements, never as raw HTML.
function renderAnswer(text, sources, onPick) {
  return text.split(/(\[\d+\]|\*\*[^*\n]+\*\*|\*[^*\n]+\*|`[^`\n]+`)/g).map((part, i) => {
    const citation = part.match(/^\[(\d+)\]$/);
    if (citation && sources[citation[1] - 1]) {
      const source = sources[citation[1] - 1];
      return html`<button key=${i} type="button" className="cite" title=${`${source.title} at ${source.timestamp}`} onClick=${() => onPick(source)}>${citation[1]}</button>`;
    }
    if (part.startsWith("**")) return html`<strong key=${i}>${part.slice(2, -2)}</strong>`;
    if (part.startsWith("*") && part.length > 2) return html`<em key=${i}>${part.slice(1, -1)}</em>`;
    if (part.startsWith("`")) return html`<code key=${i}>${part.slice(1, -1)}</code>`;
    return part;
  });
}

// A half-revealed citation or bold marker stays hidden until the rest of it is on screen.
function hideUnfinished(text) {
  let out = text.replace(/\[\d*$/, "");
  for (const marker of ["**", "`"]) {
    if ((out.split(marker).length - 1) % 2) out = out.slice(0, out.lastIndexOf(marker));
  }
  return out;
}

// Groq can finish an answer in under a second, so the text is revealed at a readable pace.
function useTypewriter(text) {
  const [count, setCount] = useState(0);
  useEffect(() => {
    if (!text) { setCount(0); return; }
    if (count >= text.length) return;
    if (REDUCED_MOTION) { setCount(text.length); return; }
    const timer = setTimeout(() => {
      setCount((c) => Math.min(text.length, c + Math.max(TYPE_CHARS, Math.ceil((text.length - c) / TYPE_CATCH_UP))));
    }, TYPE_INTERVAL_MS);
    return () => clearTimeout(timer);
  }, [text, count]);
  return text.slice(0, count);
}

async function askServer(question, onEvent, onSlow, isStale) {
  const vector = await embedQuestion(question);
  const controller = new AbortController();
  const abort = setTimeout(() => controller.abort(), ASK_TIMEOUT_MS);
  const slow = setTimeout(onSlow, WAKE_NOTICE_MS);
  try {
    const response = await fetch(`${API_BASE}/api/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, vector }),
      signal: controller.signal,
    });
    clearTimeout(slow);
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(typeof data.detail === "string" ? data.detail : "The answer service had a problem.");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) throw new Error("The answer was cut off. Please try again.");
      if (isStale()) { // a newer search replaced this one, so stop using the LLM quota on it
        controller.abort();
        return null;
      }
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const raw of events) {
        const event = JSON.parse(raw.replace(/^data: /, ""));
        if (event.error) throw new Error(event.error);
        if (event.done) return event;
        onEvent(event);
      }
    }
  } catch (error) {
    if (error.name === "AbortError" || error.name === "TypeError") {
      throw new Error("Couldn't reach the answer server. Finding moments still works, or try again in a minute.");
    }
    throw error;
  } finally {
    clearTimeout(abort);
    clearTimeout(slow);
  }
}

function sourceToMoment(source) {
  const url = new URL(source.url);
  return toMoment(url.searchParams.get("v"), source.title, parseInt(url.searchParams.get("t"), 10) || 0);
}

function Player({ moment }) {
  if (!moment) {
    return html`<div className="player-empty"><p>Pick a moment to watch it here.</p></div>`;
  }
  // The key reloads the iframe, so choosing another moment seeks to it.
  const src = `https://www.youtube-nocookie.com/embed/${moment.video_id}?start=${moment.start}&autoplay=1&rel=0`;
  return html`<div>
    <div className="player">
      <iframe key=${src} src=${src} title=${moment.title} allow="autoplay; encrypted-media; picture-in-picture" allowFullScreen></iframe>
    </div>
    <p className="now-title">${moment.title}</p>
    <a className="open" href=${moment.url} target="_blank" rel="noopener">Open at ${moment.timestamp} on YouTube</a>
  </div>`;
}

function MomentList({ moments, active, onPick, query, numbered }) {
  return html`<ol className="moments">
    ${moments.map((m, i) => html`<li key=${m.key}>
      <button type="button" className=${"moment" + (active && active.key === m.key ? " active" : "")} aria-pressed=${Boolean(active && active.key === m.key)} onClick=${() => onPick(m)}>
        <span className="thumb">
          <img src=${`https://i.ytimg.com/vi/${m.video_id}/mqdefault.jpg`} alt="" loading="lazy" width="320" height="180" />
          <span className="time">${m.timestamp}</span>
        </span>
        <span className="moment-text">
          <span className="title">${numbered && html`<span className="num">${i + 1}</span>`}${m.title}</span>
          ${m.preview && html`<span className="preview">${highlight(m.preview, query)}…</span>`}
        </span>
      </button>
    </li>`)}
  </ol>`;
}

function App() {
  const params = new URLSearchParams(location.search);
  const startMode = params.get("mode") === "answer" && API_BASE ? "answer" : "search";
  const [query, setQuery] = useState(params.get("q") || "");
  const [mode, setMode] = useState(startMode);
  const [state, setState] = useState({ phase: "idle" });
  const [progress, setProgress] = useState(null);
  const [active, setActive] = useState(null);
  const input = useRef(null);
  const runId = useRef(0);

  async function run(text, chosenMode) {
    const question = text.trim();
    if (!question) return;
    const id = ++runId.current; // a newer search replaces any answer still on its way
    setQuery(question);
    setActive(null);
    setState({ phase: "loading", question, mode: chosenMode });
    const url = new URLSearchParams({ q: question });
    if (chosenMode === "answer") url.set("mode", "answer");
    history.replaceState(null, "", `?${url}`);
    try {
      await prepare((p) => setProgress(p));
      setProgress(null);
      let next;
      if (chosenMode === "answer") {
        const isStale = () => id !== runId.current;
        const onSlow = () => { if (!isStale()) setState((s) => ({ ...s, waking: true })); };
        const onEvent = (event) => {
          if (isStale()) return;
          setState((s) => ({
            ...s,
            waking: false,
            grounded: true,
            answer: (s.answer || "") + event.text,
            moments: event.sources ? event.sources.map(sourceToMoment) : s.moments,
          }));
        };
        const data = await askServer(question, onEvent, onSlow, isStale);
        if (!data) return;
        next = { answer: data.answer, moments: data.sources.map(sourceToMoment), grounded: data.sources.length > 0 };
      } else {
        next = { moments: await search(question) };
      }
      if (id === runId.current) setState({ phase: "done", question, mode: chosenMode, ...next });
    } catch (error) {
      setProgress(null);
      if (id === runId.current) setState({ phase: "error", message: error.message || "Search couldn't start. Please reload the page." });
    }
  }

  useEffect(() => {
    if (query) run(query, startMode);
    if (API_BASE) fetch(`${API_BASE}/api/status`).catch(() => {}); // start waking the server early
    prepare((p) => setProgress(p)).then(() => setProgress(null)).catch(() => setProgress(null));
    const onKey = (e) => {
      if (e.key === "/" && document.activeElement !== input.current) {
        e.preventDefault();
        input.current.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  function chooseMode(next) {
    setMode(next);
    if (state.phase === "done" && state.question === query.trim() && state.mode !== next) run(query, next);
  }

  const busy = state.phase === "loading";
  const started = state.phase !== "idle";
  const moments = state.moments || [];
  const nothing = state.phase === "done" && !state.answer && moments.length === 0;
  const fullAnswer = state.answer || "";
  const typed = useTypewriter(fullAnswer);
  const typing = busy || typed.length < fullAnswer.length;
  const shownAnswer = typing ? hideUnfinished(typed) : fullAnswer;
  // Numbered by first use, so the highest citation on screen says how many sources to list.
  const citedSoFar = Math.max(0, ...[...shownAnswer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])));
  const shownMoments = fullAnswer && typing ? moments.slice(0, citedSoFar) : moments;

  return html`<div className=${"page" + (started ? " started" : "")}>
    <header>
      <h1>Find the moment it was explained.</h1>
      <p className="intro">Search what is said across 315 DSA lectures, 114 hours in all, and jump straight to that point in the video.</p>
    </header>

    <form className="search" onSubmit=${(e) => { e.preventDefault(); run(query, mode); }}>
      <input ref=${input} type="search" value=${query} maxLength="300" aria-label="Your question"
        placeholder="Ask about any DSA topic" onChange=${(e) => setQuery(e.target.value)} />
      <button className="primary" disabled=${busy || !query.trim()}>${mode === "answer" ? "Ask" : "Search"}</button>
    </form>

    <div className="controls">
      ${API_BASE && html`<div className="tabs" role="group" aria-label="What to show">
        <button type="button" aria-pressed=${mode === "search"} onClick=${() => chooseMode("search")}>Find moments</button>
        <button type="button" aria-pressed=${mode === "answer"} onClick=${() => chooseMode("answer")}>Get a written answer</button>
      </div>`}
      ${progress !== null && html`<div className="prep" role="status">
        <span>Preparing search, first visit only</span>
        <span className="bar"><span style=${{ width: `${Math.round(progress * 100)}%` }}></span></span>
      </div>`}
    </div>

    ${!started && html`<div className="examples">
      ${EXAMPLES.map((q) => html`<button type="button" key=${q} onClick=${() => run(q, mode)}>${q}</button>`)}
    </div>`}

    ${busy && !fullAnswer && html`<p className="status" role="status">${
      state.mode !== "answer" ? "Searching the lectures…"
      : state.waking ? "The answer server was asleep. Waking it can take up to a minute…"
      : "Writing an answer from the lectures…"}</p>`}
    ${state.phase === "error" && html`<p className="error" role="alert">${state.message}</p>`}
    ${nothing && html`<p className="status">Nothing in these lectures looks like a good match. Try describing it another way.</p>`}

    ${fullAnswer && html`<section className=${"answer" + (state.grounded ? "" : " plain") + (typing ? " typing" : "")} aria-busy=${typing}>
      <p>${renderAnswer(shownAnswer, moments, setActive)}</p>
      ${state.grounded && !typing && html`<p className="note">Written by AI from the lecture transcripts, which can contain errors. Check the sources.</p>`}
    </section>`}

    ${shownMoments.length > 0 && html`<div className="workspace">
      <${MomentList} moments=${shownMoments} active=${active} onPick=${setActive} query=${state.question} numbered=${Boolean(state.answer)} />
      <aside className="watch"><${Player} moment=${active} /></aside>
    </div>`}

    <footer>
      Search runs in your browser. Written answers send your question to this project's server and to Groq.
      Lectures belong to their creator, from the Strivers A2Z DSA Course on YouTube. An independent student project, not affiliated with the course.
    </footer>
  </div>`;
}

ReactDOM.createRoot(document.getElementById("root")).render(html`<${App} />`);
