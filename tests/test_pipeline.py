"""Run from the project root: .venv/bin/python -m unittest discover tests"""

import json
import unittest
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

from ytrag import api, rag
from ytrag.chunk import _split_segment, chunk_transcript
from ytrag.config import EMBED_DIM, REFUSAL
from ytrag.evaluate import first_hit_rank, threshold_sweep
from ytrag.export_static import quantize
from ytrag.models import Chunk, Segment, hms


def transcript(segments):
    return {
        "video_id": "vid",
        "title": "Title",
        "segments": [{"start": a, "end": b, "text": t} for a, b, t in segments],
    }


def words(n, word="word"):
    return " ".join(f"{word}{i}" for i in range(n))


def hit(n, score=0.9):
    return Chunk(f"v{n}", f"Video {n}", n * 100, n * 100 + 60, f"text {n}"), score


class ChunkTests(unittest.TestCase):
    def test_windows_overlap_and_keep_start_times(self):
        segs = [(i * 20.0, (i + 1) * 20.0, words(20)) for i in range(6)]
        chunks = chunk_transcript(transcript(segs))
        self.assertGreater(len(chunks), 2)
        self.assertEqual(chunks[0].start, 0)
        self.assertEqual(chunks[-1].end, 120)
        for a, b in zip(chunks, chunks[1:]):
            self.assertGreater(a.end, b.start)  # really overlaps
            self.assertLessEqual(a.end - b.start, 16)  # by about the configured 15 s
            self.assertLessEqual(a.end - a.start, 60)

    def test_split_keeps_text_and_time_range(self):
        text = words(10) + ". " + words(30, "other")
        pieces = _split_segment(Segment(100.0, 140.0, text))
        self.assertEqual(" ".join(p.text for p in pieces), text)
        self.assertEqual((pieces[0].start, pieces[-1].end), (100.0, 140.0))
        self.assertTrue(all(p.end - p.start <= 8.01 for p in pieces))
        second_sentence = next(p for p in pieces if p.text.startswith("other0"))
        self.assertAlmostEqual(second_sentence.start, 110.0)  # 10 of 40 words = 10 of 40 s

    def test_short_segment_is_not_split(self):
        seg = Segment(0.0, 5.0, words(10))
        self.assertEqual(_split_segment(seg), [seg])

    def test_title_is_prefixed(self):
        chunk = chunk_transcript(transcript([(0.0, 10.0, words(20))]))[0]
        self.assertTrue(chunk.text.startswith("Title\n\n"))

    def test_short_chunks_are_dropped(self):
        self.assertEqual(chunk_transcript(transcript([(0.0, 10.0, "too short")])), [])

    def test_repetition_loops_are_dropped(self):
        loop = "Thanks for watching. " * 8
        self.assertEqual(chunk_transcript(transcript([(0.0, 10.0, loop)])), [])

    def test_piece_longer_than_window_still_terminates(self):
        segs = [(0.0, 500.0, words(2)), (500.0, 900.0, words(2))]
        self.assertEqual(chunk_transcript(transcript(segs)), [])  # too few words, but it returns


class ModelTests(unittest.TestCase):
    def test_timestamp_formatting(self):
        self.assertEqual(hms(65), "1:05")
        self.assertEqual(hms(3725), "1:02:05")

    def test_link_rewinds_but_not_below_zero(self):
        self.assertTrue(Chunk("v", "t", 100, 160, "x").url.endswith("&t=97s"))
        self.assertTrue(Chunk("v", "t", 1, 60, "x").url.endswith("&t=0s"))

    def test_point_id_is_deterministic(self):
        self.assertEqual(Chunk("v", "a", 5, 9, "x").point_id, Chunk("v", "b", 5, 99, "y").point_id)
        self.assertNotEqual(Chunk("v", "a", 5, 9, "x").point_id, Chunk("v", "a", 6, 9, "x").point_id)


class AskGuardTests(unittest.TestCase):
    def ask(self, hits, llm_reply=None):
        with mock.patch.object(rag, "search", return_value=hits), \
             mock.patch.object(rag, "_llm", return_value=llm_reply) as llm:
            return rag.ask("question"), llm

    def test_refuses_without_calling_llm_when_scores_are_low(self):
        result, llm = self.ask([hit(1, score=0.1)])
        self.assertEqual(result, {"answer": REFUSAL, "sources": []})
        llm.assert_not_called()

    def test_model_refusal_returns_no_sources(self):
        result, _ = self.ask([hit(1)], llm_reply=REFUSAL)
        self.assertEqual(result, {"answer": REFUSAL, "sources": []})

    def test_keeps_only_cited_sources_and_renumbers(self):
        result, _ = self.ask([hit(1), hit(2), hit(3)], llm_reply="First [3], again [3], then [1].")
        self.assertEqual(result["answer"], "First [1], again [1], then [2].")
        self.assertEqual([s["title"] for s in result["sources"]], ["Video 3", "Video 1"])

    def test_invented_citation_is_removed(self):
        result, _ = self.ask([hit(1)], llm_reply="Claim [1] and made up [7].")
        self.assertEqual(result["answer"], "Claim [1] and made up .")
        self.assertEqual(len(result["sources"]), 1)


    def test_other_citation_styles_are_understood(self):
        result, _ = self.ask([hit(1), hit(2)], llm_reply="One【2】 two【1†L3】 both [1, 2].")
        self.assertEqual(result["answer"], "One [1] two [2] both [2][1].")
        self.assertEqual([s["title"] for s in result["sources"]], ["Video 2", "Video 1"])

    def test_array_index_is_not_a_citation(self):
        result, _ = self.ask([hit(1), hit(2)], llm_reply="Set arr[2] to zero [1].")
        self.assertEqual(result["answer"], "Set arr[2] to zero [1].")
        self.assertEqual(len(result["sources"]), 1)

    def test_code_is_left_alone(self):
        reply = "Use `x = [1]` here [2]:\n```python\nitems = [1, 2]\nfirst = [2]\n```\nDone [1]."
        result, _ = self.ask([hit(1), hit(2)], llm_reply=reply)
        self.assertEqual(result["answer"], "Use `x = [1]` here [1]:\n```python\nitems = [1, 2]\nfirst = [2]\n```\nDone [2].")
        self.assertEqual([s["title"] for s in result["sources"]], ["Video 2", "Video 1"])

    def test_citation_only_inside_code_is_refused(self):
        result, _ = self.ask([hit(1)], llm_reply="Trust me: `x = [1]`.")
        self.assertEqual(result, {"answer": REFUSAL, "sources": []})

    def test_answer_without_citations_is_refused(self):
        result, _ = self.ask([hit(1)], llm_reply="Trust me, this is how it works.")
        self.assertEqual(result, {"answer": REFUSAL, "sources": []})

    def test_browser_vector_is_used_instead_of_embedding(self):
        with mock.patch.object(rag, "search") as text_search, \
             mock.patch.object(rag, "search_vector", return_value=[hit(1)]) as vector_search, \
             mock.patch.object(rag, "_llm", return_value="Claim [1]."):
            result = rag.ask("question", vector=[0.1, 0.2])
        text_search.assert_not_called()
        self.assertEqual(vector_search.call_args.args[0], [0.1, 0.2])
        self.assertEqual(len(result["sources"]), 1)


class AskStreamTests(unittest.TestCase):
    def stream(self, hits, pieces):
        with mock.patch.object(rag, "search", return_value=hits), \
             mock.patch.object(rag, "_llm_stream", return_value=iter(pieces)) as llm:
            return list(rag.ask_stream("question")), llm

    def typed(self, events):
        return "".join(e.get("text", "") for e in events)

    def test_typed_text_matches_the_checked_answer(self):
        events, _ = self.stream([hit(1), hit(2), hit(3)], ["First [", "3], again", " [3], then", " [1", "]."])
        self.assertEqual(self.typed(events), "First [1], again [1], then [2].")
        self.assertEqual(events[-1]["answer"], "First [1], again [1], then [2].")
        self.assertEqual([s["title"] for s in events[-1]["sources"]], ["Video 3", "Video 1"])

    def test_source_arrives_before_or_with_its_citation(self):
        events, _ = self.stream([hit(1), hit(2)], ["One [2] ", "and more ", "two [1]."])
        known = 0
        for event in events[:-1]:
            known = len(event.get("sources", [])) or known
            cited = [int(n) for n in rag._CITATION.findall(event["text"])]
            self.assertLessEqual(max(cited, default=0), known)
        self.assertEqual(known, 2)

    def test_citation_styles_split_across_pieces(self):
        events, _ = self.stream([hit(1), hit(2)], ["One", "【", "2†L", "1】 two ", "【1】."])
        self.assertEqual(self.typed(events), "One [1] two [2].")

    def test_array_index_split_across_pieces_is_kept(self):
        events, _ = self.stream([hit(1), hit(2)], ["Set arr", "[2", "] to zero [", "1]."])
        self.assertEqual(self.typed(events), "Set arr[2] to zero [1].")

    def test_code_block_split_across_pieces_stays_intact(self):
        pieces = ["Like this [1]:\n", "```py\nitems = [", "1, 2]\n", "first = [2]\n``", "`\nDone [", "2]."]
        events, _ = self.stream([hit(1), hit(2)], pieces)
        expected = "Like this [1]:\n```py\nitems = [1, 2]\nfirst = [2]\n```\nDone [2]."
        self.assertEqual(self.typed(events), expected)
        self.assertEqual(events[-1]["answer"], expected)

    def test_refusal_and_uncited_reply_are_never_typed(self):
        for pieces in ([REFUSAL[:9], REFUSAL[9:]], ["Trust me, ", "this works."]):
            events, _ = self.stream([hit(1)], pieces)
            self.assertEqual(self.typed(events), "")
            self.assertEqual(events[-1], {"done": True, "answer": REFUSAL, "sources": []})

    def test_refuses_without_calling_llm_when_scores_are_low(self):
        events, llm = self.stream([hit(1, score=0.1)], [])
        self.assertEqual(events, [{"done": True, "answer": REFUSAL, "sources": []}])
        llm.assert_not_called()


class ExportTests(unittest.TestCase):
    def test_quantize_keeps_range_and_values(self):
        out = quantize(np.array([[1.0, -1.0, 0.5, 0.0]], dtype=np.float32))
        self.assertEqual(out.dtype, np.int8)
        self.assertEqual(out.tolist(), [[127, -127, 64, 0]])


class ApiTests(unittest.TestCase):
    vector = [0.1] * EMBED_DIM

    def setUp(self):
        api._visitor_times.clear()
        api._all_times.clear()
        self.client = TestClient(api.app)

    def post(self, body):
        with mock.patch.object(api, "GROQ_API_KEY", "set"), \
             mock.patch.object(api, "rag_ask", return_value={"answer": "a", "sources": []}):
            return self.client.post("/api/ask", json=body)

    def test_ask_passes(self):
        self.assertEqual(self.post({"question": "q", "vector": self.vector}).status_code, 200)

    def test_rejects_wrong_vector_size_and_long_question(self):
        self.assertEqual(self.post({"question": "q", "vector": [0.1]}).status_code, 422)
        self.assertEqual(self.post({"question": "q" * 301, "vector": self.vector}).status_code, 422)

    def test_visitor_limit(self):
        for _ in range(api.ASK_LIMIT_PER_HOUR):
            self.assertEqual(self.post({"question": "q", "vector": self.vector}).status_code, 200)
        self.assertEqual(self.post({"question": "q", "vector": self.vector}).status_code, 429)

    def test_total_limit_cannot_be_dodged_with_faked_addresses(self):
        with mock.patch.object(api, "ASK_TOTAL_LIMIT_PER_HOUR", 2), mock.patch.object(api, "GROQ_API_KEY", "set"), \
             mock.patch.object(api, "rag_ask", return_value={"answer": "a", "sources": []}):
            codes = [
                self.client.post("/api/ask", json={"question": "q", "vector": self.vector},
                                 headers={"x-forwarded-for": f"10.0.0.{i}"}).status_code
                for i in range(3)
            ]
        self.assertEqual(codes, [200, 200, 429])

    def test_only_allowed_sites_get_cors_access(self):
        allowed = self.client.options("/api/ask", headers={"Origin": api.ALLOWED_ORIGINS[0], "Access-Control-Request-Method": "POST"})
        other = self.client.options("/api/ask", headers={"Origin": "https://example.com", "Access-Control-Request-Method": "POST"})
        self.assertIn("access-control-allow-origin", allowed.headers)
        self.assertNotIn("access-control-allow-origin", other.headers)

    def test_ping_reports_count_and_hides_failures(self):
        with mock.patch.object(api, "chunk_count", return_value=7):
            self.assertEqual(self.client.get("/api/ping").json(), {"chunks": 7})
        with mock.patch.object(api, "chunk_count", side_effect=RuntimeError("secret detail")), mock.patch("builtins.print"):
            response = self.client.get("/api/ping")
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("secret", response.text)

    def test_llm_failure_hides_details(self):
        with mock.patch.object(api, "GROQ_API_KEY", "set"), \
             mock.patch.object(api, "rag_ask", side_effect=RuntimeError("secret detail")), mock.patch("builtins.print"):
            response = self.client.post("/api/ask", json={"question": "q", "vector": self.vector})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("secret", response.text)


class EvaluateTests(unittest.TestCase):
    def test_first_hit_rank(self):
        self.assertEqual(first_hit_rank(["a", "b", "c"], {"b", "c"}), 2)
        self.assertIsNone(first_hit_rank(["a"], {"z"}))

    def test_threshold_sweep(self):
        row = next(r for r in threshold_sweep([0.8, 0.4], [0.6, 0.3]) if r["threshold"] == 0.5)
        self.assertEqual(row["in_domain_answered"], 0.5)
        self.assertEqual(row["off_topic_refused"], 0.5)


if __name__ == "__main__":
    unittest.main()

    def stream(self, reply):
        with mock.patch.object(api, "GROQ_API_KEY", "set"), mock.patch.object(api, "rag_ask_stream", reply), mock.patch("builtins.print"):
            return self.client.post("/api/ask/stream", json={"question": "q", "vector": self.vector})

    def test_stream_sends_events(self):
        response = self.stream(lambda *a, **k: iter([{"text": "Hi [1]"}, {"done": True, "answer": "Hi [1]", "sources": []}]))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        events = [json.loads(line[6:]) for line in response.text.split("\n\n") if line]
        self.assertEqual(events[0], {"text": "Hi [1]"})
        self.assertTrue(events[-1]["done"])

    def test_stream_failure_hides_details(self):
        def fail(*a, **k):
            yield {"text": "Hi"}
            raise RuntimeError("secret detail")
        response = self.stream(fail)
        self.assertNotIn("secret", response.text)
        self.assertIn('"error"', response.text)

    def test_stream_shares_the_rate_limit(self):
        reply = lambda *a, **k: iter([])
        for _ in range(api.ASK_LIMIT_PER_HOUR):
            self.assertEqual(self.stream(reply).status_code, 200)
        self.assertEqual(self.stream(reply).status_code, 429)
