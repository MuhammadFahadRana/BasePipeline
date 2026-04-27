"""SQL Server-backed semantic search engine with SemanticSearchEngine-compatible API."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from embeddings.text_embeddings import get_embedding_generator
from search.semantic_search import (
    SearchResult,
    _facet_suggestions_for_query,
    _sense_suggestions,
    extract_keywords,
)


class SqlServerSemanticSearchEngine:
    """Search engine that runs against SQL Server tables/procedures."""

    def __init__(
        self,
        db: Optional[Session] = None,
        cache_enabled: bool = False,
        cache_ttl_seconds: int = 3600,
        max_cache_size: int = 1000,
        parallel_enabled: bool = False,
        reranker_enabled: bool = False,
    ) -> None:
        # Keep ctor signature close to SemanticSearchEngine for compatibility.
        del cache_enabled, cache_ttl_seconds, max_cache_size, parallel_enabled, reranker_enabled

        self._session_factory = None
        self._owns_session = False
        if db is None or not self._session_dialect_name(db).startswith("mssql"):
            self._session_factory = self._load_session_factory()
            self.db = self._session_factory()
            self._owns_session = True
        else:
            self.db = db

        self.embedding_gen = get_embedding_generator()
        self.stats: Dict[str, Any] = {
            "queries": 0,
            "avg_latency_ms": 0.0,
            "last_error": None,
        }

    @staticmethod
    def _session_dialect_name(db: Optional[Session]) -> str:
        bind = getattr(db, "bind", None)
        dialect = getattr(bind, "dialect", None)
        return str(getattr(dialect, "name", "") or "").lower()

    @staticmethod
    def _load_session_factory():
        # Lazy import so postgres-only mode never trips MSSQL dependency checks.
        from database.SQL.mssql_connection import SessionLocal

        return SessionLocal

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_score(value: Any) -> float:
        score = SqlServerSemanticSearchEngine._safe_float(value, default=0.0)
        if score < 0.0:
            return 0.0
        return score

    def _update_latency_stats(self, latency_ms: float) -> None:
        n = int(self.stats.get("queries", 0))
        current_avg = float(self.stats.get("avg_latency_ms", 0.0) or 0.0)
        self.stats["avg_latency_ms"] = (
            ((current_avg * (n - 1)) + latency_ms) / n if n > 0 else latency_ms
        )

    def _lookup_video_metadata(self, filename: Optional[str]) -> tuple[int, str]:
        if not filename:
            return 0, ""
        row = self.db.execute(
            text(
                """
                SELECT TOP (1) id, file_path
                FROM dbo.videos
                WHERE filename = :filename
                """
            ),
            {"filename": filename},
        ).fetchone()
        if not row:
            return 0, ""
        return self._safe_int(getattr(row, "id", 0), 0), str(
            getattr(row, "file_path", "") or ""
        )

    def update_db(self, db: Optional[Session] = None) -> None:
        """
        Keep the engine pinned to an MSSQL session.
        API request handlers pass the primary postgres session into generic
        sync hooks, so ignore any non-MSSQL session here.
        """
        if db is None:
            return
        if self._session_dialect_name(db).startswith("mssql"):
            self.db = db

    def _rows_to_results(
        self,
        rows: List[Any],
        default_match_type: str = "hybrid",
        default_score: float = 0.0,
    ) -> List[SearchResult]:
        results: List[SearchResult] = []
        for row in rows:
            segment_id = self._safe_int(getattr(row, "segment_id", 0), 0)
            video_id = self._safe_int(getattr(row, "video_id", 0), 0)
            video_filename = str(getattr(row, "video_filename", "") or "")
            video_path = str(
                getattr(row, "video_path", getattr(row, "file_path", "")) or ""
            )
            if (not video_id or not video_path) and video_filename:
                looked_up_id, looked_up_path = self._lookup_video_metadata(
                    video_filename
                )
                if not video_id:
                    video_id = looked_up_id
                if not video_path:
                    video_path = looked_up_path

            start_time = self._safe_float(getattr(row, "start_time", 0.0), 0.0)
            end_time = self._safe_float(getattr(row, "end_time", start_time), start_time)
            result_text = str(
                getattr(row, "text", getattr(row, "result_text", "")) or ""
            )
            combined_score = self._normalize_score(
                getattr(row, "combined_score", default_score)
            )
            match_type = str(getattr(row, "match_type", default_match_type) or default_match_type)
            result = SearchResult(
                segment_id=segment_id,
                video_id=video_id,
                video_filename=video_filename,
                video_path=video_path,
                start_time=start_time,
                end_time=end_time,
                text=result_text,
                score=combined_score,
                match_type=match_type,
                db_source="sqlserver",
            )
            results.append(result)
        return results

    def search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.65,
        text_weight: float = 0.35,
        min_score: float = 0.15,
        video_filter: Optional[str] = None,
        log_query: bool = True,
        use_cache: bool = True,
        deep_search: bool = False,
    ) -> List[SearchResult]:
        """Hybrid semantic/text search via SQL Server stored procedure + fallback SQL."""
        del log_query, use_cache, deep_search
        start_time = time.time()
        self.stats["queries"] = int(self.stats.get("queries", 0)) + 1

        clean_query = (query or "").strip()
        if not clean_query:
            return []

        query_instruction = (
            "Given a query, retrieve relevant passages that answer the query\nQuery: "
        )
        query_embedding = self.embedding_gen.encode_single(
            clean_query, instruction=query_instruction
        )
        query_embedding_json = json.dumps(query_embedding.tolist())

        rows: List[Any] = []
        try:
            rows = self.db.execute(
                text(
                    """
                    EXEC dbo.hybrid_search
                        @query_text=:query_text,
                        @query_embedding=:query_embedding,
                        @text_weight=:text_weight,
                        @semantic_weight=:semantic_weight,
                        @limit_results=:limit_results
                    """
                ),
                {
                    "query_text": clean_query,
                    "query_embedding": query_embedding_json,
                    "text_weight": float(text_weight),
                    "semantic_weight": float(semantic_weight),
                    "limit_results": int(max(1, top_k * 4)),
                },
            ).fetchall()
        except Exception as exc:
            self.stats["last_error"] = f"hybrid_search failed: {exc}"
            try:
                self.db.rollback()
            except Exception:
                pass

        if not rows:
            rows = self.db.execute(
                text(
                    """
                    SELECT
                        ts.id AS segment_id,
                        v.id AS video_id,
                        v.filename AS video_filename,
                        v.file_path AS video_path,
                        ts.start_time,
                        ts.end_time,
                        ts.[text] AS result_text,
                        CAST(
                            CASE WHEN ts.[text] LIKE '%' + :query_text + '%'
                                 THEN 1.0
                                 ELSE 0.0
                            END
                        AS FLOAT) AS combined_score
                    FROM dbo.transcript_segments ts
                    JOIN dbo.videos v ON v.id = ts.video_id
                    WHERE ts.[text] LIKE '%' + :query_text + '%'
                      AND (:video_filter IS NULL OR v.filename = :video_filter)
                    ORDER BY combined_score DESC, ts.start_time ASC
                    OFFSET 0 ROWS
                    FETCH NEXT :limit_results ROWS ONLY
                    """
                ),
                {
                    "query_text": clean_query,
                    "video_filter": video_filter,
                    "limit_results": int(max(1, top_k * 4)),
                },
            ).fetchall()

        results = self._rows_to_results(rows, default_match_type="hybrid")
        if video_filter:
            results = [
                r
                for r in results
                if (r.video_filename or "").lower() == video_filter.lower()
            ]

        filtered = [r for r in results if r.score >= float(min_score)]
        filtered.sort(key=lambda x: x.score, reverse=True)
        final_results = filtered[:top_k]

        self._update_latency_stats((time.time() - start_time) * 1000.0)
        return final_results

    def search_with_fallback(
        self,
        query: str,
        top_k: int = 10,
        video_filter: Optional[str] = None,
        log_query: bool = True,
        facet: str = "auto",
        deep_search: bool = False,
    ) -> Dict[str, Any]:
        """Tiered fallback search compatible with existing API responses."""
        del deep_search
        clean_query = (query or "").strip()
        metadata: Dict[str, Any] = {
            "original_query": clean_query,
            "corrected_query": None,
            "corrections": [],
            "did_you_mean": None,
            "search_strategy": "direct",
            "search_message": None,
            "tiers_tried": [],
            "keywords_used": extract_keywords(clean_query),
            "facet_applied": (facet or "auto"),
            "facets": _facet_suggestions_for_query(clean_query),
            "sense_suggestions": _sense_suggestions(clean_query),
        }

        if not clean_query:
            metadata["search_strategy"] = "no_results"
            metadata["search_message"] = "Empty query."
            return {"results": [], "search_metadata": metadata}

        results = self.search(
            query=clean_query,
            top_k=top_k,
            semantic_weight=0.65,
            text_weight=0.35,
            min_score=0.20,
            video_filter=video_filter,
            log_query=log_query,
        )
        metadata["tiers_tried"].append("direct")
        if results:
            return {"results": results[:top_k], "search_metadata": metadata}

        relaxed = self.search(
            query=clean_query,
            top_k=top_k,
            semantic_weight=0.5,
            text_weight=0.5,
            min_score=0.05,
            video_filter=video_filter,
            log_query=False,
        )
        metadata["tiers_tried"].append("relaxed")
        metadata["search_strategy"] = "relaxed"
        if relaxed:
            metadata["search_message"] = f'Showing best available matches for "{clean_query}"'
            return {"results": relaxed[:top_k], "search_metadata": metadata}

        metadata["search_strategy"] = "no_results"
        metadata["search_message"] = (
            f'No results found for "{clean_query}". Try simpler or different keywords.'
        )
        return {"results": [], "search_metadata": metadata}

    def search_exact_phrase(
        self, phrase: str, video_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """Exact phrase match (case-insensitive in typical SQL Server collations)."""
        clean_phrase = (phrase or "").strip()
        if not clean_phrase:
            return []

        rows = self.db.execute(
            text(
                """
                SELECT
                    ts.id AS segment_id,
                    v.id AS video_id,
                    v.filename AS video_filename,
                    v.file_path AS video_path,
                    ts.start_time,
                    ts.end_time,
                    ts.[text] AS result_text
                FROM dbo.transcript_segments ts
                JOIN dbo.videos v ON v.id = ts.video_id
                WHERE ts.[text] LIKE '%' + :phrase + '%'
                  AND (:video_filter IS NULL OR v.filename = :video_filter)
                ORDER BY ts.start_time ASC
                OFFSET 0 ROWS
                FETCH NEXT :top_k ROWS ONLY
                """
            ),
            {
                "phrase": clean_phrase,
                "video_filter": video_filter,
                "top_k": 200,
            },
        ).fetchall()

        results = self._rows_to_results(
            rows, default_match_type="exact", default_score=1.0
        )
        for result in results:
            result.score = 1.0
            result.match_type = "exact"
        return results

    def clear_cache(self, memory_only: bool = False):
        del memory_only
        return None

    def cleanup_expired_cache(self):
        try:
            row = self.db.execute(
                text("EXEC dbo.clean_query_cache")
            ).fetchone()
            if not row:
                return 0
            return self._safe_int(row[0], 0)
        except Exception:
            try:
                self.db.rollback()
            except Exception:
                pass
            return 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "backend": "sqlserver",
            "total_queries": int(self.stats.get("queries", 0)),
            "avg_latency_ms": round(float(self.stats.get("avg_latency_ms", 0.0) or 0.0), 2),
            "last_error": self.stats.get("last_error"),
        }

    def close(self) -> None:
        if self._owns_session and self.db is not None:
            try:
                self.db.close()
            except Exception:
                pass

    def __del__(self):
        self.close()
