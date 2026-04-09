"""
Cross-encoder reranker for improving search precision.
UPDATED: Now uses the shared 1.5B LLM Singleton as a zero-shot reranker.
"""

import time
import re
from typing import List, Optional, Any
from llm.llm_manager import get_llm_manager

class CrossEncoderReranker:
    """Rerank search results using the shared 1.5B Instruct model."""

    def __init__(
        self,
        model_name: str = "shared-1.5B-instruct",
        device: str = "auto",
        max_length: int = 512,
    ):
        self.device = device
        self.model_name = model_name
        self.max_length = max_length
        self.manager = get_llm_manager()
        print(f"[OK] Reranker initialized using shared LLM infrastructure.")

    def rerank(
        self,
        query: str,
        results: List[Any],
        top_k: Optional[int] = None,
        score_blend: float = 0.7,
    ) -> List[Any]:
        """
        Rerank results using the shared LLM as a judge.
        """
        if not results or len(results) <= 1:
            return results

        start_time = time.time()
        
        # We use the manager's rerank logic which already blends scores
        # and handles the prompt-based evaluation.
        # We rerank the top 12 to ensure speed while significantly boosting relevance.
        rerank_limit = 12
        reranked_results = self.manager.rerank(query, results, top_n=rerank_limit)
        
        elapsed_ms = (time.time() - start_time) * 1000
        print(
            f"  Smart Reranker: Processed {min(len(results), rerank_limit)} candidates in {elapsed_ms:.0f}ms."
        )

        if top_k:
            return reranked_results[:top_k]
        return reranked_results


# ── Global singleton (lazy loaded) ──────────────────────────────

_reranker: Optional[CrossEncoderReranker] = None
_reranker_failed: bool = False

def get_reranker(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    enabled: bool = True,
) -> Optional[CrossEncoderReranker]:
    """
    Get or create the global reranker instance using the shared LLM.
    """
    global _reranker, _reranker_failed

    if not enabled or _reranker_failed:
        return None

    if _reranker is None:
        try:
            _reranker = CrossEncoderReranker()
        except Exception as e:
            print(f"[WARNING] Reranker failed to initialize: {e}")
            _reranker_failed = True
            return None

    return _reranker
