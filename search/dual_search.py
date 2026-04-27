"""Dual-database search wrapper (PostgreSQL + SQL Server)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from search.semantic_search import SearchResult, SemanticSearchEngine
from search.sqlserver_search import SqlServerSemanticSearchEngine


class DualSemanticSearchEngine:
    """Federates search across postgres/sqlserver and tags provenance."""

    def __init__(
        self,
        postgres_engine: Optional[SemanticSearchEngine] = None,
        sqlserver_engine: Optional[SqlServerSemanticSearchEngine] = None,
        mode: str = "both",
        rrf_k: int = 60,
    ) -> None:
        self.postgres_engine = postgres_engine
        self.sqlserver_engine = sqlserver_engine
        self.mode = self._normalize_mode(mode)
        self.rrf_k = max(1, int(rrf_k))

    @staticmethod
    def _normalize_mode(mode: Optional[str]) -> str:
        raw = (mode or "both").strip().lower()
        aliases = {
            "pg": "postgres",
            "postgresql": "postgres",
            "mssql": "sqlserver",
            "sql": "sqlserver",
            "dual": "both",
            "all": "both",
        }
        normalized = aliases.get(raw, raw)
        if normalized not in {"postgres", "sqlserver", "both"}:
            return "both"
        return normalized

    @property
    def db(self):
        if self.postgres_engine is not None:
            return getattr(self.postgres_engine, "db", None)
        return None

    @db.setter
    def db(self, value):
        if self.postgres_engine is not None:
            self.postgres_engine.db = value

    def update_db(self, db) -> None:
        if self.postgres_engine is not None:
            self.postgres_engine.db = db
        if self.sqlserver_engine is not None and hasattr(self.sqlserver_engine, "update_db"):
            self.sqlserver_engine.update_db(db)

    def _active_sources(self) -> List[Tuple[str, Any]]:
        sources: List[Tuple[str, Any]] = []
        if self.mode in {"postgres", "both"} and self.postgres_engine is not None:
            sources.append(("postgres", self.postgres_engine))
        if self.mode in {"sqlserver", "both"} and self.sqlserver_engine is not None:
            sources.append(("sqlserver", self.sqlserver_engine))

        if sources:
            return sources

        if self.postgres_engine is not None:
            return [("postgres", self.postgres_engine)]
        if self.sqlserver_engine is not None:
            return [("sqlserver", self.sqlserver_engine)]
        return []

    @staticmethod
    def _tag_results(results: List[SearchResult], source_name: str) -> List[SearchResult]:
        tagged: List[SearchResult] = []
        for item in results or []:
            item.db_source = source_name
            tagged.append(item)
        return tagged

    def _rrf_merge(
        self, source_results: Dict[str, List[SearchResult]], top_k: int
    ) -> List[SearchResult]:
        scored: List[Tuple[float, float, SearchResult]] = []
        for source_name, results in source_results.items():
            ranked = sorted(results or [], key=lambda x: float(x.score), reverse=True)
            for rank, result in enumerate(ranked, start=1):
                rrf_score = 1.0 / float(self.rrf_k + rank)
                raw_score = max(0.0, float(getattr(result, "score", 0.0) or 0.0))
                combined_rank_score = rrf_score + (0.02 * min(raw_score, 1.0))
                result.db_source = source_name
                scored.append((combined_rank_score, raw_score, result))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [entry[2] for entry in scored[:top_k]]

    def _merge_by_mode(
        self, source_results: Dict[str, List[SearchResult]], top_k: int
    ) -> List[SearchResult]:
        if not source_results:
            return []
        if len(source_results) == 1:
            only_source = next(iter(source_results.keys()))
            single = self._tag_results(source_results[only_source], only_source)
            single.sort(key=lambda x: float(x.score), reverse=True)
            return single[:top_k]
        return self._rrf_merge(source_results, top_k=top_k)

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
        source_results: Dict[str, List[SearchResult]] = {}
        per_source_k = max(top_k, min(100, top_k * 2))
        for source_name, engine in self._active_sources():
            try:
                results = engine.search(
                    query=query,
                    top_k=per_source_k,
                    semantic_weight=semantic_weight,
                    text_weight=text_weight,
                    min_score=min_score,
                    video_filter=video_filter,
                    log_query=log_query,
                    use_cache=use_cache,
                    deep_search=deep_search,
                )
                source_results[source_name] = self._tag_results(results, source_name)
            except Exception as exc:
                print(f"[dual-search] {source_name} search failed: {exc}")

        return self._merge_by_mode(source_results, top_k=top_k)

    def search_with_fallback(
        self,
        query: str,
        top_k: int = 10,
        video_filter: Optional[str] = None,
        log_query: bool = True,
        facet: str = "auto",
        deep_search: bool = False,
    ) -> Dict[str, Any]:
        source_results: Dict[str, List[SearchResult]] = {}
        source_metadata: Dict[str, Dict[str, Any]] = {}
        per_source_k = max(top_k, min(100, top_k * 2))

        for source_name, engine in self._active_sources():
            try:
                payload = engine.search_with_fallback(
                    query=query,
                    top_k=per_source_k,
                    video_filter=video_filter,
                    log_query=log_query,
                    facet=facet,
                    deep_search=deep_search,
                )
                rows = self._tag_results(payload.get("results", []), source_name)
                source_results[source_name] = rows
                source_metadata[source_name] = payload.get("search_metadata", {}) or {}
            except Exception as exc:
                print(f"[dual-search] {source_name} fallback failed: {exc}")

        merged_results = self._merge_by_mode(source_results, top_k=top_k)
        primary_meta = source_metadata.get("postgres") or source_metadata.get("sqlserver") or {}
        tiers_tried: List[str] = []
        for source_name, metadata in source_metadata.items():
            for tier in metadata.get("tiers_tried") or []:
                tiers_tried.append(f"{source_name}:{tier}")

        metadata: Dict[str, Any] = {
            "original_query": primary_meta.get("original_query", query),
            "corrected_query": primary_meta.get("corrected_query"),
            "corrections": primary_meta.get("corrections", []),
            "did_you_mean": primary_meta.get("did_you_mean"),
            "search_strategy": "dual",
            "search_message": primary_meta.get("search_message"),
            "tiers_tried": tiers_tried,
            "keywords_used": primary_meta.get("keywords_used"),
            "facet_applied": primary_meta.get("facet_applied", facet or "auto"),
            "facets": primary_meta.get("facets") or [],
            "sense_suggestions": primary_meta.get("sense_suggestions") or [],
            "source_modes": list(source_results.keys()),
            "source_counts": {k: len(v) for k, v in source_results.items()},
            "source_metadata": source_metadata,
        }
        return {"results": merged_results, "search_metadata": metadata}

    def search_exact_phrase(
        self, phrase: str, video_filter: Optional[str] = None
    ) -> List[SearchResult]:
        source_results: Dict[str, List[SearchResult]] = {}
        for source_name, engine in self._active_sources():
            try:
                rows = engine.search_exact_phrase(
                    phrase=phrase,
                    video_filter=video_filter,
                )
                source_results[source_name] = self._tag_results(rows, source_name)
            except Exception as exc:
                print(f"[dual-search] {source_name} exact search failed: {exc}")
        return self._merge_by_mode(source_results, top_k=200)

    def clear_cache(self, memory_only: bool = False):
        if self.postgres_engine is not None:
            try:
                self.postgres_engine.clear_cache(memory_only=memory_only)
            except Exception:
                pass
        if self.sqlserver_engine is not None:
            try:
                self.sqlserver_engine.clear_cache(memory_only=memory_only)
            except Exception:
                pass

    def cleanup_expired_cache(self):
        cleaned = 0
        if self.postgres_engine is not None:
            try:
                cleaned += int(self.postgres_engine.cleanup_expired_cache() or 0)
            except Exception:
                pass
        if self.sqlserver_engine is not None:
            try:
                cleaned += int(self.sqlserver_engine.cleanup_expired_cache() or 0)
            except Exception:
                pass
        return cleaned

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "backend": "dual",
            "mode": self.mode,
            "sources": {},
        }
        if self.postgres_engine is not None:
            try:
                stats["sources"]["postgres"] = self.postgres_engine.get_stats()
            except Exception as exc:
                stats["sources"]["postgres"] = {"error": str(exc)}
        if self.sqlserver_engine is not None:
            try:
                stats["sources"]["sqlserver"] = self.sqlserver_engine.get_stats()
            except Exception as exc:
                stats["sources"]["sqlserver"] = {"error": str(exc)}
        return stats
