"""
Cross-encoder-style reranker for improving search precision.

Implementation detail: this module uses an instruction LLM as a zero-shot judge
and blends that relevance score with retrieval score (hybrid rerank).
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from llm.llm_manager import get_llm_manager


_cpu_warning_emitted = False
_reranker_cache: Dict[Tuple[str, str, float], "CrossEncoderReranker"] = {}
_reranker_failed_keys: set[Tuple[str, str, float]] = set()


def _safe_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _safe_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-4B")
DEFAULT_RERANKER_MODE = os.getenv("RERANKER_MODE", "hybrid").strip().lower()
DEFAULT_RERANKER_BLEND = _safe_env_float("RERANKER_BLEND", 0.7)
DEFAULT_RERANK_LIMIT = _safe_env_int("RERANKER_TOP_N", 12)


class CrossEncoderReranker:
    """Rerank search results using an LLM-as-a-judge strategy."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = "auto",
        max_length: int = 512,
        mode: str = DEFAULT_RERANKER_MODE,
        score_blend: float = DEFAULT_RERANKER_BLEND,
        rerank_limit: int = DEFAULT_RERANK_LIMIT,
    ):
        self.device = device
        self.model_name = model_name
        self.max_length = max_length
        self.mode = (mode or "hybrid").strip().lower()
        self.score_blend = max(0.0, min(1.0, float(score_blend)))
        self.rerank_limit = max(1, int(rerank_limit))
        self.manager = get_llm_manager(model_name=model_name)

        device_tag = "[GPU]" if "cuda" in self.manager.device else "[CPU]"
        print(
            f"{device_tag} Reranker initialized ({self.manager.device}) "
            f"model={self.model_name} mode={self.mode} blend={self.score_blend:.2f}"
        )

    def rerank(
        self,
        query: str,
        results: List[Any],
        top_k: Optional[int] = None,
        score_blend: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> List[Any]:
        """Rerank results using an LLM as a judge."""
        if not results or len(results) <= 1:
            return results

        effective_mode = (mode or self.mode).strip().lower()
        effective_blend = self.score_blend if score_blend is None else float(score_blend)
        effective_blend = max(0.0, min(1.0, effective_blend))

        if effective_mode in {"disabled", "off", "none"} or effective_blend <= 0.0:
            return results[:top_k] if top_k else results

        if effective_mode == "llm_only":
            effective_blend = 1.0

        start_time = time.time()

        rerank_limit = self.rerank_limit
        if "cpu" in self.manager.device:
            global _cpu_warning_emitted
            rerank_limit = min(rerank_limit, top_k or len(results), 4)
            if not _cpu_warning_emitted:
                print(
                    "[CPU] Deep Search is running in reduced mode "
                    "(reranking the top 4 candidates only)."
                )
                _cpu_warning_emitted = True

        reranked_results = self.manager.rerank(
            query=query,
            results=results,
            top_n=rerank_limit,
            llm_weight=effective_blend,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        print(
            f"  Smart Reranker: Processed {min(len(results), rerank_limit)} candidates in "
            f"{elapsed_ms:.0f}ms (mode={effective_mode}, blend={effective_blend:.2f})."
        )

        return reranked_results[:top_k] if top_k else reranked_results


def get_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
    enabled: bool = True,
    mode: str = DEFAULT_RERANKER_MODE,
    score_blend: float = DEFAULT_RERANKER_BLEND,
) -> Optional[CrossEncoderReranker]:
    """Get or create a reranker instance (cached by model+mode+blend)."""
    if not enabled:
        return None

    mode = (mode or DEFAULT_RERANKER_MODE).strip().lower()
    score_blend = max(0.0, min(1.0, float(score_blend)))
    cache_key = (model_name, mode, round(score_blend, 4))

    if cache_key in _reranker_failed_keys:
        return None

    if cache_key not in _reranker_cache:
        try:
            _reranker_cache[cache_key] = CrossEncoderReranker(
                model_name=model_name,
                mode=mode,
                score_blend=score_blend,
            )
        except Exception as e:
            print(f"[WARNING] Reranker failed to initialize: {e}")
            _reranker_failed_keys.add(cache_key)
            return None

    return _reranker_cache[cache_key]
