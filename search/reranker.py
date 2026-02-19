"""Cross-encoder reranker for improving search precision.

Uses BAAI/bge-reranker-v2-m3 to rerank top-K candidates retrieved
from hybrid search (semantic + fuzzy). Cross-encoders jointly encode
(query, document) pairs, giving much more accurate relevance scores
than bi-encoder similarity alone.
"""

import time
import re
from typing import List, Optional

import torch
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Rerank search results using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
        max_length: int = 512,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model name
            device: "auto", "cpu", or "cuda"
            max_length: Maximum token length for (query, document) pairs
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name
        self.max_length = max_length

        print(f"Loading reranker model: {model_name}")
        print(f"Device: {device}")

        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
            trust_remote_code=True,
        )

        print(f"[OK] Reranker loaded ({model_name})")

    def rerank(
        self,
        query: str,
        results: List,
        top_k: Optional[int] = None,
        score_blend: float = 0.6,
    ) -> List:
        """
        Rerank search results using cross-encoder scoring.

        The final score blends the reranker score with the original
        retrieval score to preserve diversity while improving precision.

        Args:
            query: Original search query
            results: List of SearchResult objects from hybrid search
            top_k: Maximum results to return (None = return all)
            score_blend: Weight for reranker score vs original score
                         0.0 = keep original scores, 1.0 = use only reranker

        Returns:
            Reranked list of SearchResult objects with updated scores
        """
        if not results or len(results) <= 1:
            return results

        start_time = time.time()

        # Build (query, document_text) pairs for cross-encoder
        pairs = []
        for r in results:
            doc_text = self._clean_text(r.text)
            pairs.append((query, doc_text))

        # Score all pairs in one batch
        raw_scores = self.model.predict(pairs, show_progress_bar=False)

        # Normalize reranker scores to [0, 1] via sigmoid (already applied by
        # most BGE rerankers, but clip just in case)
        reranker_scores = []
        for s in raw_scores:
            # Sigmoid if raw logits, otherwise clamp
            if s < -10 or s > 10:
                s = 1.0 / (1.0 + _exp_safe(-s))
            reranker_scores.append(max(0.0, min(1.0, float(s))))

        # Blend reranker scores with original retrieval scores
        for i, result in enumerate(results):
            original_score = result.score
            reranker_score = reranker_scores[i]

            # Blended score: gives reranker more influence while preserving
            # retrieval diversity signals
            result.score = (
                score_blend * reranker_score
                + (1.0 - score_blend) * original_score
            )
            result.match_type = f"reranked_{result.match_type}"

        # Sort by new blended score
        results.sort(key=lambda x: x.score, reverse=True)

        elapsed_ms = (time.time() - start_time) * 1000
        print(
            f"  Reranker: {len(results)} candidates in {elapsed_ms:.0f}ms "
            f"(top: {results[0].score:.4f})"
        )

        if top_k:
            results = results[:top_k]

        return results

    def _clean_text(self, text: str) -> str:
        """Clean document text before sending to cross-encoder."""
        # Strip [OCR] prefix for cleaner comparison
        text = re.sub(r"^\[OCR\]\s*", "", text)
        # Collapse whitespace
        text = " ".join(text.split())
        return text[:1000]  # Limit length to avoid hitting max_length


def _exp_safe(x: float) -> float:
    """Safe exp to avoid overflow."""
    import math
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


# ── Global singleton (lazy loaded) ──────────────────────────────

_reranker: Optional[CrossEncoderReranker] = None
_reranker_failed: bool = False


def get_reranker(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    enabled: bool = True,
) -> Optional[CrossEncoderReranker]:
    """
    Get or create the global reranker instance.

    Returns None if disabled or if initialization fails (graceful fallback).
    """
    global _reranker, _reranker_failed

    if not enabled or _reranker_failed:
        return None

    if _reranker is None:
        try:
            _reranker = CrossEncoderReranker(model_name=model_name)
        except Exception as e:
            print(f"[WARNING] Reranker failed to load: {e}")
            print("  Search will continue without reranking.")
            _reranker_failed = True
            return None

    return _reranker
