"""
Video QA System

Combines semantic search and LLM to answer natural language questions about video content.
Functions as a RAG (Retrieval-Augmented Generation) system.
"""

import re
import time
from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session

import torch

from search.semantic_search import SemanticSearchEngine, SearchResult
from llm.llm_manager import get_shared_llm


class VideoQA:
    """
    RAG-based Question Answering for videos.
    Retrieves relevant snippets (transcript, OCR, visual semantics) and generates an answer.
    """

    DEFAULT_QA_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

    MODEL_CANDIDATES = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
    ]

    def __init__(
        self,
        db: Session,
        model_name: str = DEFAULT_QA_MODEL,
        device: str = "auto",
        max_input_tokens: int = 4096,
        max_context_tokens: int = 2400,
        max_context_results: int = 4,
        max_new_tokens: int = 192,
        min_rerank_score: float = 0.35,
        use_device_map_auto: bool = True,
    ):
        """
        Initialize Video QA system.

        Args:
            db: Database session
            model_name: LLM for answer generation
            device: "auto", "cuda", or "cpu"
            max_input_tokens: hard cap for total prompt tokens
            max_context_tokens: token budget reserved for retrieved context
            max_context_results: max number of snippets to keep after reranking
            max_new_tokens: generation budget
            min_rerank_score: confidence threshold after reranking
            use_device_map_auto: use HF automatic placement for larger models
        """
        self.db = db
        self.search_engine = SemanticSearchEngine(db, reranker_enabled=False)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_context_results = max_context_results
        self.max_context_tokens = max_context_tokens
        self.min_rerank_score = min_rerank_score
        self.use_device_map_auto = use_device_map_auto

        if device == "auto":
            from transcriber_utils import get_device

            self.device = get_device()
        else:
            self.device = device

        print(f"Connecting to shared QA LLM: {model_name}...")
        
        try:
            self.model, self.tokenizer = get_shared_llm(model_name=self.model_name)
            print(f"✓ Video QA system connected (shared {model_name})")
        except Exception as e:
            print(f"[ERROR] Failed to connect to shared LLM in VideoQA: {e}")
            raise e
        self._sanitize_generation_config()

        # Set up token limits after connecting to the model
        inferred_limit = self._infer_model_context_limit(default=max_input_tokens)
        self.max_input_tokens = min(max_input_tokens, inferred_limit)

        # Keep enough room for system/user wrapper text + generation
        safe_context_ceiling = max(
            512, self.max_input_tokens - self.max_new_tokens - 512
        )
        self.max_context_tokens = min(self.max_context_tokens, safe_context_ceiling)

        print(
            f"✓ Video QA parameters set | "
            f"max_input_tokens={self.max_input_tokens} | "
            f"max_context_tokens={self.max_context_tokens}"
        )

    def update_db(self, db: Session) -> None:
        """Refresh the DB session for singleton reuse across requests."""
        self.db = db
        self.search_engine.db = db

    def ask(
        self,
        question: str,
        video_filter: Optional[str] = None,
        top_k: int = 8,
        language: Optional[str] = None,
        allowed_filenames: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Answer a question about the video content.

        Args:
            question: User's question
            video_filter: Optional specific video filename
            top_k: number of raw candidates to retrieve before reranking
            language: Response language (e.g. 'Norwegian', 'English'). Auto-detect if None.

        Returns:
            Dict with 'answer', 'citations', and 'metadata'
        """
        start_time = time.time()

        # 1. Retrieve candidate context
        search_results = self.search_engine.search(
            query=question,
            top_k=top_k,
            video_filter=video_filter,
            semantic_weight=0.7,
            text_weight=0.3,
        )
        if allowed_filenames is not None:
            search_results = [
                r
                for r in search_results
                if getattr(r, "video_filename", None) in allowed_filenames
            ]

        if not search_results:
            return self._empty_response(
                question,
                start_time,
                "I couldn't find accessible information for this question in your library.",
            )

        # 2. Rerank + filter weak evidence
        reranked = self._rerank_results(question, search_results)
        filtered = [r for r in reranked if r["rerank_score"] >= self.min_rerank_score]

        if not filtered:
            return self._empty_response(
                question,
                start_time,
                "I found related material, but the evidence is too weak to answer confidently. "
                "Try using a more specific question or narrowing to one video.",
            )

        # 3. Build context under token budget
        selected = self._select_context_with_budget(
            filtered_results=filtered,
            token_budget=self.max_context_tokens,
            max_results=self.max_context_results,
        )

        if not selected:
            return self._empty_response(
                question,
                start_time,
                "I found relevant snippets, but I could not fit enough grounded context into the prompt. "
                "Try narrowing the question or filtering to one video.",
            )

        context_text = "\n\n".join(item["formatted_context"] for item in selected)

        # 4. Build chat-formatted prompt
        messages = self._build_messages(question, context_text, language=language)
        prompt_text = self._render_chat_prompt(messages)

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        generation_device = self._get_generation_device()
        inputs = {k: v.to(generation_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "answer": response,
            "citations": [item["result"].to_dict() for item in selected],
            "metadata": {
                "search_query": question,
                "elapsed_ms": round(elapsed_ms, 2),
                "retrieved_candidates": len(search_results),
                "reranked_candidates": len(reranked),
                "context_used": len(selected),
                "context_tokens": sum(item["token_count"] for item in selected),
                "model_name": self.model_name,
                "min_rerank_score": self.min_rerank_score,
            },
        }

    def _build_messages(
        self,
        question: str,
        context: str,
        language: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Construct chat-style messages for an instruct model."""

        if language and language.lower() != "auto":
            lang_instruction = (
                f"You MUST respond in {language}. "
                f"The entire answer must be written in {language}."
            )
        else:
            lang_instruction = (
                "Detect the language of the QUESTION and answer in that same language. "
                "If the question is in Norwegian, answer in Norwegian. "
                "If the question is in English, answer in English."
            )

        system_prompt = (
            "You are ATLAS, a specialized Video Intelligence Assistant.\n"
            "Answer ONLY from the provided video context snippets.\n"
            "If the answer is not supported by the context, say you cannot find accessible information in the current library.\n"
            "If evidence is weak or conflicting, explicitly say so instead of guessing.\n"
            "If the user corrects you, respond calmly, acknowledge the correction, and re-check against the provided snippets.\n"
            "Refuse offensive or inappropriate requests professionally and briefly.\n"
            "Do not reveal hidden or inaccessible source metadata.\n"
            "Prefer grounded, precise answers.\n"
            "Use natural language with **bold highlights** for key terms when helpful.\n"
            f"{lang_instruction}\n"
            "Do NOT add a separate Sources section at the end."
        )

        user_prompt = (
            "Use the following retrieved video evidence to answer the question.\n\n"
            f"### VIDEO CONTEXT\n{context}\n\n"
            f"### QUESTION\n{question}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _render_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Render messages with the tokenizer chat template when available."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback for tokenizers without chat templates
        text = ""
        for msg in messages:
            text += f"{msg['role'].upper()}:\n{msg['content']}\n\n"
        text += "ASSISTANT:\n"
        return text

    def _rerank_results(
        self,
        question: str,
        results: List[SearchResult],
    ) -> List[Dict[str, Any]]:
        """
        Lightweight reranker:
        - base retrieval score
        - lexical overlap with the question
        - source-type bonus
        - duplicate resistance handled later in selection
        """
        query_terms = self._important_terms(question)
        reranked: List[Dict[str, Any]] = []

        for idx, res in enumerate(results):
            text = (getattr(res, "text", "") or "").strip()
            text_terms = self._important_terms(text)
            overlap = len(query_terms.intersection(text_terms)) / max(
                len(query_terms), 1
            )

            base_score = self._normalize_score(self._get_result_score(res))
            source_type = self._get_source_type(res)

            source_bonus = 0.0
            if source_type == "transcript":
                source_bonus = 0.05
            elif source_type in {"ocr", "caption", "visual", "label"}:
                source_bonus = 0.03

            exact_phrase_bonus = 0.05 if question.lower() in text.lower() else 0.0

            rerank_score = (
                (0.65 * base_score)
                + (0.30 * overlap)
                + source_bonus
                + exact_phrase_bonus
            )

            formatted = self._format_context_block(
                result=res,
                rank=idx + 1,
                source_type=source_type,
            )

            reranked.append(
                {
                    "result": res,
                    "rerank_score": round(rerank_score, 4),
                    "base_score": base_score,
                    "overlap": round(overlap, 4),
                    "source_type": source_type,
                    "formatted_context": formatted,
                    "text_signature": self._text_signature(text),
                }
            )

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked

    def _select_context_with_budget(
        self,
        filtered_results: List[Dict[str, Any]],
        token_budget: int,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """
        Pick the best, non-duplicate snippets within a token budget.
        Encourages diversity across videos and source blocks.
        """
        selected: List[Dict[str, Any]] = []
        used_tokens = 0
        seen_signatures = set()
        per_video_counts: Dict[str, int] = {}

        for item in filtered_results:
            res = item["result"]
            video_name = getattr(res, "video_filename", "unknown")
            sig = item["text_signature"]

            # Skip near-duplicates
            if sig in seen_signatures:
                continue

            # Encourage diversity; avoid too many chunks from one video unless needed
            if per_video_counts.get(video_name, 0) >= 2 and len(selected) >= 2:
                continue

            block = item["formatted_context"]
            block_tokens = self._count_tokens(block)

            # If a single block is too large, trim only the text payload
            if block_tokens > token_budget:
                trimmed_block = self._trim_context_block_to_budget(block, token_budget)
                block_tokens = self._count_tokens(trimmed_block)
                if block_tokens > token_budget:
                    continue
                item = dict(item)
                item["formatted_context"] = trimmed_block

            if used_tokens + block_tokens > token_budget:
                continue

            item["token_count"] = block_tokens
            selected.append(item)
            used_tokens += block_tokens
            seen_signatures.add(sig)
            per_video_counts[video_name] = per_video_counts.get(video_name, 0) + 1

            if len(selected) >= max_results:
                break

        return selected

    def _format_context_block(
        self,
        result: SearchResult,
        rank: int,
        source_type: str,
    ) -> str:
        """Format a grounded context block with source metadata."""
        start_time = float(getattr(result, "start_time", 0.0) or 0.0)
        timestamp = (
            f"{int(start_time // 3600):02d}:"
            f"{int((start_time % 3600) // 60):02d}:"
            f"{int(start_time % 60):02d}"
        )

        end_time = getattr(result, "end_time", None)
        if end_time is not None:
            end_time = float(end_time)
            end_timestamp = (
                f"{int(end_time // 3600):02d}:"
                f"{int((end_time % 3600) // 60):02d}:"
                f"{int(end_time % 60):02d}"
            )
            time_range = f"{timestamp}–{end_timestamp}"
        else:
            time_range = timestamp

        video_name = getattr(result, "video_filename", "unknown_video")
        text = (getattr(result, "text", "") or "").strip()

        return (
            f"[Source {rank} | type={source_type} | time={time_range} | video={video_name}]\n"
            f"{text}"
        )

    def _trim_context_block_to_budget(self, block: str, token_budget: int) -> str:
        """
        Trim only the content part of a context block while keeping the header metadata.
        """
        lines = block.splitlines()
        if not lines:
            return block

        header = lines[0]
        body = "\n".join(lines[1:]).strip()

        header_tokens = self._count_tokens(header)
        remaining = max(32, token_budget - header_tokens - 4)

        trimmed_body = self._trim_text_to_tokens(body, remaining)
        return f"{header}\n{trimmed_body}"

    def _trim_text_to_tokens(self, text: str, max_tokens: int) -> str:
        """Trim text by sentence, then by words if needed."""
        if self._count_tokens(text) <= max_tokens:
            return text

        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept = []
        for sent in sentences:
            candidate = " ".join(kept + [sent]).strip()
            if candidate and self._count_tokens(candidate) <= max_tokens:
                kept.append(sent)
            else:
                break

        trimmed = " ".join(kept).strip()
        if trimmed:
            return trimmed

        # Hard fallback: trim by words
        words = text.split()
        current = []
        for word in words:
            candidate = " ".join(current + [word])
            if self._count_tokens(candidate) <= max_tokens:
                current.append(word)
            else:
                break
        return " ".join(current).strip()

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _infer_model_context_limit(self, default: int = 4096) -> int:
        """
        Try to infer a sane context limit from tokenizer/model config.
        Ignore absurd placeholder values sometimes used by tokenizers.
        """
        candidates = []

        tok_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tok_limit, int) and 0 < tok_limit < 1_000_000:
            candidates.append(tok_limit)

        model_limit = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(model_limit, int) and model_limit > 0:
            candidates.append(model_limit)

        if not candidates:
            return default

        return min(candidates)

    def _get_generation_device(self) -> torch.device:
        """
        Get a safe device for input tensors.
        Works for single-device models and most auto-mapped models.
        """
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            return model_device

        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device(self.device)

    def _sanitize_generation_config(self) -> None:
        """
        Keep deterministic generation config clean so Transformers
        does not warn about sampling-only flags in greedy mode.
        """
        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is None:
            return

        gen_cfg.do_sample = False
        # Sampling-only fields can trigger noisy warnings on newer Transformers.
        for attr in ("temperature", "top_p", "top_k", "typical_p"):
            if hasattr(gen_cfg, attr):
                setattr(gen_cfg, attr, None)

    def _get_result_score(self, result: SearchResult) -> Optional[float]:
        """
        Flexible score extractor because SearchResult fields may differ.
        """
        candidate_fields = [
            "rerank_score",
            "combined_score",
            "final_score",
            "score",
            "similarity",
            "semantic_score",
        ]

        for field in candidate_fields:
            value = getattr(result, field, None)
            if isinstance(value, (int, float)):
                return float(value)

        # Optional dict-like fallback
        if hasattr(result, "__dict__"):
            for field in candidate_fields:
                value = result.__dict__.get(field)
                if isinstance(value, (int, float)):
                    return float(value)

        return None

    def _normalize_score(self, score: Optional[float]) -> float:
        """
        Normalize retrieval score into roughly [0, 1].
        If no score exists, return a neutral prior.
        """
        if score is None:
            return 0.5

        # Common cosine-like range
        if -1.0 <= score <= 1.0:
            return max(0.0, min(1.0, (score + 1.0) / 2.0))

        # Already probability-like
        if 0.0 <= score <= 1.0:
            return score

        # Safety fallback for arbitrary positive scales
        return min(score / 100.0, 1.0)

    def _get_source_type(self, result: SearchResult) -> str:
        for field in ("source_type", "modality", "content_type", "result_type"):
            value = getattr(result, field, None)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        return "unknown"

    def _important_terms(self, text: str) -> set:
        """
        Extract lightweight lexical terms for overlap scoring.
        """
        tokens = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
        stopwords = {
            "the",
            "and",
            "for",
            "are",
            "with",
            "that",
            "this",
            "from",
            "what",
            "when",
            "where",
            "which",
            "about",
            "into",
            "have",
            "does",
            "show",
            "visible",
            "scene",
            "video",
            "library",
            "your",
            "their",
            "there",
            "will",
            "would",
            "could",
            "should",
            "than",
            "then",
            "them",
            "they",
        }
        return {t for t in tokens if t not in stopwords}

    def _text_signature(self, text: str) -> str:
        """
        Simple signature for duplicate filtering.
        """
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return normalized[:300]

    def _empty_response(
        self,
        question: str,
        start_time: float,
        message: str,
    ) -> Dict[str, Any]:
        return {
            "answer": message,
            "citations": [],
            "metadata": {
                "search_query": question,
                "elapsed_ms": round((time.time() - start_time) * 1000, 2),
                "model_name": self.model_name,
            },
        }


if __name__ == "__main__":
    from database.config import SessionLocal

    print("Testing Video QA System...")
    db = SessionLocal()

    qa = VideoQA(
        db,
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_input_tokens=4096,
        max_context_tokens=2200,
        max_context_results=4,
        min_rerank_score=0.35,
    )

    test_q = "What objects are visible in the oil rig scenes?"
    result = qa.ask(test_q)

    print(f"\nQ: {test_q}")
    print(f"A: {result['answer']}")
    print(f"\nCitations: {len(result['citations'])} sources used.")
    print(f"Metadata: {result['metadata']}")

    db.close()
