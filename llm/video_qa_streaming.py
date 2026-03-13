"""
llm/video_qa_streaming.py

Streaming extension of VideoQA for the OpenAI-compatible /v1/chat/completions
endpoint.  Uses HuggingFace TextIteratorStreamer so that generated tokens are
yielded one-by-one instead of waiting for the full answer.
"""

import json
import time
import threading
from typing import Iterator, Optional, Dict, List

import torch
from transformers import TextIteratorStreamer

from llm.video_qa import VideoQA
from search.semantic_search import SearchResult
from sqlalchemy.orm import Session


class StreamingVideoQA(VideoQA):
    """
    Extends VideoQA with a streaming `stream_ask` generator.

    Yields OpenAI SSE-compatible JSON strings:
        data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"..."}}]}

    A `---\\n**Sources**` block listing citations is streamed at the very end.
    """

    def stream_ask(
        self,
        question: str,
        video_filter: Optional[str] = None,
        top_k: int = 5,
        request_id: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> Iterator[str]:
        """
        Generator that yields SSE data lines (already formatted as
        `data: {...}\n\n` strings) for SSE streaming.

        Args:
            question: User's question
            video_filter: Optional video filename to restrict search
            top_k: How many retrieved segments to use as RAG context
            request_id: Optional request ID for SSE IDs
            max_new_tokens: Max tokens to generate
        """
        rid = request_id or f"chatcmpl-{int(time.time())}"

        def _chunk(content: str) -> str:
            payload = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "ATLAS",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None,
                    }
                ],
            }
            return f"data: {json.dumps(payload)}\n\n"

        def _done() -> str:
            payload = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "ATLAS",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n"

        # ── 1. Semantic search ──────────────────────────────────────────
        search_results: List[SearchResult] = self.search_engine.search(
            query=question,
            top_k=top_k,
            video_filter=video_filter,
            semantic_weight=0.7,
            text_weight=0.3,
        )

        if not search_results:
            msg = (
                "I couldn't find relevant information in the video library "
                "to answer your question. Try rephrasing or using different keywords."
            )
            yield _chunk(msg)
            yield _done()
            return

        # ── 2. Build RAG context ────────────────────────────────────────
        context_parts = []
        for i, res in enumerate(search_results):
            ts = _fmt_ts(res.start_time)
            context_parts.append(
                f"[Source {i + 1} | {res.video_filename} @ {ts}]\n{res.text}"
            )
        context_text = "\n\n".join(context_parts)

        # ── 3. Build prompt ─────────────────────────────────────────────
        prompt = self._build_prompt(question, context_text)

        # ── 4. Stream generation ────────────────────────────────────────
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0,
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "repetition_penalty": 1.1,
            "streamer": streamer,
        }

        # Run generation in a background thread so we can yield from main thread
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        # Yield tokens as they arrive
        for token_text in streamer:
            if token_text:
                yield _chunk(token_text)

        thread.join()

        # ── 5. Append formatted source citations ────────────────────────
        citations = _build_citation_block(search_results)
        yield _chunk(citations)
        yield _done()

    def ask_sync(
        self,
        question: str,
        video_filter: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict:
        """
        Synchronous (non-streaming) version — aggregates the stream and
        returns an OpenAI-style complete response dict.
        """
        full_text = ""
        for chunk in self.stream_ask(question, video_filter=video_filter, top_k=top_k):
            if chunk.startswith("data: [DONE]") or not chunk.startswith("data: "):
                continue
            try:
                payload = json.loads(chunk[6:])
                content = payload["choices"][0]["delta"].get("content", "")
                full_text += content
            except Exception:
                pass

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "ATLAS",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    s = int(seconds or 0)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _build_citation_block(results: List[SearchResult]) -> str:
    """Build a Markdown citations block appended after the main answer."""
    if not results:
        return ""

    lines = ["\n\n---\n**📹 Sources**\n"]
    seen = set()
    for i, r in enumerate(results):
        key = (r.video_filename, int(r.start_time))
        if key in seen:
            continue
        seen.add(key)
        ts = _fmt_ts(r.start_time)
        snippet = (r.text or "")[:100].replace("\n", " ").strip()
        score_pct = int((r.score or 0) * 100)
        lines.append(
            f"{i + 1}. **{r.video_filename}** @ `{ts}` ({score_pct}% match)  \n"
            f"   > {snippet}…\n"
        )

    return "".join(lines)


# ── Singleton ──────────────────────────────────────────────────────────────────

_streaming_qa: Optional[StreamingVideoQA] = None


def get_streaming_qa(db: Session) -> StreamingVideoQA:
    """Get or create the global StreamingVideoQA instance."""
    global _streaming_qa
    if _streaming_qa is None:
        _streaming_qa = StreamingVideoQA(db=db)
    return _streaming_qa


def reset_streaming_qa():
    """Force recreation of the singleton (e.g. after DB reconnect)."""
    global _streaming_qa
    _streaming_qa = None
