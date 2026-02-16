"""Semantic search engine with typo/fuzzy matching support - OPTIMIZED."""

import re
import hashlib
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.models import Video, TranscriptSegment, Embedding, SearchQuery
from embeddings.text_embeddings import get_embedding_generator

# Global NeuSpell checker cache (lazy loaded once)
_neuspell_checker = None
_neuspell_loaded = False


def _get_neuspell_checker():
    """Get or create global NeuSpell BertChecker instance (loaded once)."""
    global _neuspell_checker, _neuspell_loaded
    if _neuspell_loaded:
        return _neuspell_checker
    
    _neuspell_loaded = True
    try:
        import torch
        import torch.nn as nn
        from neuspell import BertChecker
        
        # Monkeypatches for legacy model loading
        original_load = torch.load
        original_load_state_dict = nn.Module.load_state_dict
        
        try:
            def new_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            torch.load = new_load
            
            def new_load_state_dict(self, state_dict, strict=True, **kwargs):
                if hasattr(self, 'bert_model'):
                    strict = False
                return original_load_state_dict(self, state_dict, strict=strict, **kwargs)
            nn.Module.load_state_dict = new_load_state_dict
            
            checker = BertChecker()
            checker.from_pretrained()
            _neuspell_checker = checker
            print("NeuSpell BertChecker loaded successfully (cached globally)")
        finally:
            torch.load = original_load
            nn.Module.load_state_dict = original_load_state_dict
            
    except Exception as e:
        print(f"Failed to load NeuSpell: {e}")
        _neuspell_checker = None
    
    return _neuspell_checker


@dataclass
class SearchResult:
    """Search result with video timestamp and text."""

    segment_id: int
    video_id: int
    video_filename: str
    video_path: str  # Full path to video file
    start_time: float
    end_time: float
    text: str
    score: float
    match_type: str  # "exact", "fuzzy", "semantic"
    keyframe_path: str = ""  # Path to keyframe image for thumbnails

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "segment_id": self.segment_id,
            "video_id": self.video_id,
            "video_filename": self.video_filename,
            "video_path": self.video_path,
            "timestamp": f"{int(self.start_time // 3600):02d}:{int((self.start_time % 3600) // 60):02d}:{int(self.start_time % 60):02d}",
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "text": self.text,
            "score": round(self.score, 4),
            "match_type": self.match_type,
            "keyframe_path": self.keyframe_path,
        }


class SemanticSearchEngine:
    """Search engine with typo tolerance and semantic understanding."""

    def __init__(
        self, 
        db: Session,
        # Caching options
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 3600,
        max_cache_size: int = 1000,
        # Parallel execution
        parallel_enabled: bool = True
    ):
        """
        Initialize search engine.

        Args:
            db: Database session
            cache_enabled: Enable query result caching (OPTIMIZATION #5)
            cache_ttl_seconds: Cache time-to-live in seconds (default: 1 hour)
            max_cache_size: Maximum entries in memory cache
            parallel_enabled: Run semantic + fuzzy searches in parallel (OPTIMIZATION #4)
        """
        self.db = db
        self.embedding_gen = get_embedding_generator()

        # Use globally cached NeuSpell checker (loaded once, reused across requests)
        self.checker = _get_neuspell_checker()
        
        # Caching configuration
        self.cache_enabled = cache_enabled
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.max_cache_size = max_cache_size
        self._memory_cache: Dict[str, Tuple[List[SearchResult], datetime]] = {}
        
        # Parallel execution
        self.parallel_enabled = parallel_enabled
        self._executor = ThreadPoolExecutor(max_workers=2) if parallel_enabled else None
        
        # Performance statistics
        self.stats = {
            'queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_hits': 0,
            'db_hits': 0,
            'avg_latency_ms': 0.0,
        }

    def _load_neuspell(self):
        """Load NeuSpell BertChecker with compatibility patches for legacy models."""
        import torch
        import torch.nn as nn
        from neuspell import BertChecker
        
        # Monkeypatches for legacy model loading
        original_load = torch.load
        original_load_state_dict = nn.Module.load_state_dict
        
        try:
            # Patch torch.load to disable weights_only=True
            def new_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            torch.load = new_load
            
            # Patch load_state_dict to be non-strict and ignore assign
            def new_load_state_dict(self, state_dict, strict=True, **kwargs):
                if hasattr(self, 'bert_model'):
                    strict = False
                return original_load_state_dict(self, state_dict, strict=strict, **kwargs)
            nn.Module.load_state_dict = new_load_state_dict
            
            checker = BertChecker()
            checker.from_pretrained()
            return checker
            
        finally:
            # Restore original functions safely
            torch.load = original_load
            nn.Module.load_state_dict = original_load_state_dict

    def _correct_typos(self, query: str) -> Tuple[str, List[str]]:
        """
        Correct common typos in query.

        Args:
            query: Original query

        Returns:
            Tuple of (corrected_query, list_of_corrections)
        """
        if not self.checker:
            return query, []
            
        try:
            # Clean query first? NeuSpell handles punctuation usually, but let's be safe
            # Actually, passing raw query is often better for context
            corrected_query = self.checker.correct(query)
            
            # Remove any trailing periods if original didn't have them (NeuSpell sometimes adds punctuation)
            if not query.endswith('.') and corrected_query.endswith('.'):
                corrected_query = corrected_query[:-1]
                
            corrections = []
            if corrected_query.lower() != query.lower():
                corrections.append(f"{query} → {corrected_query}")
                
            return corrected_query, corrections
        except Exception as e:
            print(f"Autocorrection failed: {e}")
            return query, []

    def _fuzzy_match_score(self, query_term: str, text: str) -> float:
        """
        Calculate fuzzy matching score for a term in text.

        Args:
            query_term: Search term
            text: Text to search in

        Returns:
            Fuzzy match score (0-1)
        """
        query_term = query_term.lower()
        text = text.lower()

        # Exact match
        if query_term in text:
            return 1.0

        # Fuzzy match using sequence matcher
        best_score = 0.0
        words = text.split()

        for word in words:
            # Skip if length difference is too big (e.g. "oh" vs "omega")
            if abs(len(word) - len(query_term)) > 3:
                continue

            # Skip short words for fuzzy matching (less than 4 chars) unless exact
            if len(query_term) < 4 and word != query_term:
                continue
                
            score = SequenceMatcher(None, query_term, word).ratio()
            
            # Boost score if one is substring of another
            if query_term in word or word in query_term:
                score = max(score, 0.9)
                
            best_score = max(best_score, score)

        return best_score

    def search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        text_weight: float = 0.3,
        min_score: float = 0.15,
        video_filter: Optional[str] = None,
        log_query: bool = True,
        use_cache: bool = True,
    ) -> List[SearchResult]:
        """
        Hybrid search combining semantic similarity and fuzzy text matching.
        NOW OPTIMIZED with parallel execution (#4) and caching (#5)!

        Args:
            query: Search query (e.g., "where Omega Alpha well is discussed")
            top_k: Number of results to return
            semantic_weight: Weight for semantic similarity (0-1)
            text_weight: Weight for fuzzy text matching (0-1)
            min_score: Minimum combined score threshold
            video_filter: Optional video filename to filter results
            log_query: Log query to database for analytics
            use_cache: Use cached results if available (OPTIMIZATION #5)

        Returns:
            List of SearchResult objects sorted by score
        """
        start_time = time.time()
        self.stats['queries'] += 1
        
        # OPTIMIZATION #5: Check cache if enabled
        cache_key = self._cache_key(query, top_k, semantic_weight=semantic_weight,
                                    text_weight=text_weight, min_score=min_score,
                                    video_filter=video_filter)
        
        if self.cache_enabled and use_cache:
            # Memory cache first (fastest ~1ms)
            cached = self._check_memory_cache(cache_key)
            if cached:
                self._update_latency_stats((time.time() - start_time) * 1000)
                return cached[:top_k]
            
            # Database cache second (~5ms)
            cached = self._check_db_cache(cache_key)
            if cached:
                self._update_latency_stats((time.time() - start_time) * 1000)
                return cached[:top_k]
        
        self.stats['cache_misses'] += 1
        
        # Typo correction
        corrected_query, corrections = self._correct_typos(query)

        if corrections:
            print(f"Typo corrections applied: {', '.join(corrections)}")
            print(f"Corrected query: {corrected_query}")

        # Generate query embedding (optimized for Qwen3-Embedding)
        query_instruction = "Given a query, retrieve relevant passages that answer the query\nQuery: "
        query_embedding = self.embedding_gen.encode_single(corrected_query, instruction=query_instruction)

        # OPTIMIZATION #4: Parallel execution if enabled
        if self.parallel_enabled and self._executor:
            semantic_future = self._executor.submit(
                self._semantic_search, query_embedding, top_k=top_k * 3, video_filter=video_filter
            )
            fuzzy_future = self._executor.submit(
                self._fuzzy_text_search, corrected_query, top_k=top_k * 3, video_filter=video_filter
            )
            semantic_results = semantic_future.result()
            fuzzy_results = fuzzy_future.result()
        else:
            semantic_results = self._semantic_search(
                query_embedding, top_k=top_k * 3, video_filter=video_filter
            )
            fuzzy_results = self._fuzzy_text_search(
                corrected_query, top_k=top_k * 3, video_filter=video_filter
            )

        # Combine and re-rank results
        combined_results = self._combine_results(
            semantic_results,
            fuzzy_results,
            semantic_weight=semantic_weight,
            text_weight=text_weight,
        )
        
        # Filter out extremely short noisy segments (e.g. "Talking", "Except")
        combined_results = [r for r in combined_results if len(r.text.strip()) > 7]

        # Sort by score first
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        # Filter by minimum score
        combined_results = [r for r in combined_results if r.score >= min_score]

        # Dynamic/Relative Filtering: Drop results that are much worse than top result
        if combined_results:
            top_score = combined_results[0].score
            # Keep results within 60% of top score (was 75%, relaxed for better recall)
            relative_threshold = top_score * 0.60
            combined_results = [r for r in combined_results if r.score >= relative_threshold]

        final_results = combined_results[:top_k]
        
        # OPTIMIZATION #5: Save to cache
        if self.cache_enabled and final_results:
            cache_params = {'top_k': top_k, 'semantic_weight': semantic_weight,
                          'text_weight': text_weight, 'min_score': min_score,
                          'video_filter': video_filter}
            self._save_to_cache(cache_key, query, final_results, cache_params)

        # Log query
        if log_query and final_results:
            self._log_query(query, query_embedding.tolist(), len(final_results), final_results[0].segment_id)
        
        self._update_latency_stats((time.time() - start_time) * 1000)
        return final_results

    def search_with_fallback(
        self,
        query: str,
        top_k: int = 10,
        video_filter: Optional[str] = None,
        log_query: bool = True,
    ) -> Dict:
        """
        Tiered search with fallback strategy (Google-like behavior).
        
        Tier 1: Full query search (semantic + fuzzy)
        Tier 2: Relaxed thresholds if Tier 1 returns too few results
        Tier 3: Word decomposition — search individual words, merge results
        
        Returns:
            Dict with 'results', 'search_metadata' containing strategy info
        """
        # Typo correction applied once
        corrected_query, corrections = self._correct_typos(query)
        search_query = corrected_query if corrections else query
        
        metadata = {
            "original_query": query,
            "corrected_query": search_query if corrections else None,
            "corrections": corrections,
            "search_strategy": "exact",
            "search_message": None,
            "tiers_tried": [],
        }
        
        # ── Tier 1: Full query search with standard thresholds ──
        results = self.search(
            query=search_query,
            top_k=top_k,
            min_score=0.20,
            video_filter=video_filter,
            log_query=log_query,
        )
        metadata["tiers_tried"].append("full_query")
        
        # Check quality: enough results with decent scores?
        good_results = [r for r in results if r.score >= 0.30]
        if len(good_results) >= min(3, top_k):
            metadata["search_strategy"] = "direct"
            if corrections:
                metadata["search_message"] = f"Showing results for \"{search_query}\" (corrected from \"{query}\")"
            return {"results": results, "search_metadata": metadata}
        
        # ── Tier 2: Relaxed thresholds ──
        relaxed_results = self.search(
            query=search_query,
            top_k=top_k,
            min_score=0.10,
            video_filter=video_filter,
            log_query=False,  # Don't double-log
        )
        metadata["tiers_tried"].append("relaxed")
        
        if len(relaxed_results) > len(results):
            results = relaxed_results
        
        good_results = [r for r in results if r.score >= 0.20]
        if len(good_results) >= min(2, top_k):
            metadata["search_strategy"] = "relaxed"
            metadata["search_message"] = f"Showing best available matches for \"{search_query}\""
            return {"results": results, "search_metadata": metadata}
        
        # ── Tier 3: Word decomposition ──
        # Filter out stop words / filler words that add no search value
        STOP_WORDS = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "it", "as", "are",
            "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "shall", "should",
            "may", "might", "must", "can", "could", "not", "no", "nor",
            "so", "if", "then", "than", "that", "this", "these", "those",
            "what", "which", "who", "whom", "how", "when", "where", "why",
            "all", "each", "every", "both", "few", "more", "most", "some",
            "any", "other", "into", "about", "between", "through", "during",
            "before", "after", "above", "below", "up", "down", "out", "off",
            "over", "under", "again", "further", "once", "here", "there",
            "very", "just", "also", "too", "only", "own", "same", "such",
        }
        words = [w for w in search_query.lower().split() if len(w) >= 3 and w not in STOP_WORDS]
        
        if len(words) > 1:
            metadata["tiers_tried"].append("decomposed")
            all_word_results = {}
            
            for word in words:
                word_results = self.search(
                    query=word,
                    top_k=top_k,
                    min_score=0.15,
                    video_filter=video_filter,
                    log_query=False,
                )
                for r in word_results:
                    key = r.segment_id
                    if key in all_word_results:
                        # Boost score for segments matching multiple words
                        existing = all_word_results[key]
                        existing.score = max(existing.score, r.score) * 1.15
                    else:
                        all_word_results[key] = r
            
            decomposed_results = sorted(
                all_word_results.values(),
                key=lambda x: x.score, 
                reverse=True
            )[:top_k]
            
            if decomposed_results and (
                not results or decomposed_results[0].score > (results[0].score if results else 0)
            ):
                results = decomposed_results
                metadata["search_strategy"] = "expanded"
                matched_words = ", ".join(f'"{w}"' for w in words)
                metadata["search_message"] = (
                    f"No exact matches for \"{search_query}\". "
                    f"Showing results for individual terms: {matched_words}"
                )
            elif results:
                metadata["search_strategy"] = "relaxed"
                metadata["search_message"] = f"Showing best available matches for \"{search_query}\""
        
        # Final fallback message if still no results
        if not results:
            metadata["search_strategy"] = "no_results"
            metadata["search_message"] = (
                f"No results found for \"{search_query}\". "
                f"Try simpler or different keywords."
            )
        
        return {"results": results, "search_metadata": metadata}

    def _semantic_search(
        self, query_embedding, top_k: int = 20, video_filter: Optional[str] = None
    ) -> Dict[int, Tuple[float, TranscriptSegment]]:
        """
        Semantic search using vector similarity.

        Returns:
            Dict mapping segment_id -> (score, segment)
        """
        # Use pgvector cosine similarity (1 - cosine_distance)
        query_filter = ""
        if video_filter:
            query_filter = "AND v.filename = :video_filter"

        sql_query = text(f"""
            SELECT 
                ts.id as segment_id,
                v.id as video_id,
                v.filename,
                v.file_path,
                COALESCE(ts.start_time, s.start_time) as start_time,
                COALESCE(ts.end_time, s.end_time) as end_time,
                COALESCE(ts.text, '[Scene match: ocr/caption]') as text,
                1 - (e.embedding <=> CAST(:query_embedding AS vector)) AS similarity,
                s.scene_id,
                ve.keyframe_path
            FROM embeddings e
            LEFT JOIN transcript_segments ts ON e.segment_id = ts.id
            LEFT JOIN scenes s ON (e.scene_id = s.id OR ts.scene_id = s.id)
            LEFT JOIN visual_embeddings ve ON s.id = ve.scene_id
            JOIN videos v ON (ts.video_id = v.id OR s.video_id = v.id)
            WHERE 1=1 {query_filter}
            ORDER BY e.embedding <=> CAST(:query_embedding AS vector)
            LIMIT :top_k
        """)

        params = {"query_embedding": query_embedding.tolist(), "top_k": top_k}
        if video_filter:
            params["video_filter"] = video_filter

        results = self.db.execute(sql_query, params).fetchall()

        semantic_scores = {}
        for row in results:
            # Check column count and names from row
            segment_id = row.segment_id if row.segment_id else 0
            video_id = row.video_id
            filename = row.filename
            file_path = row.file_path
            start = row.start_time
            end = row.end_time
            segment_text = row.text
            similarity = row.similarity
            scene_val = row.scene_id
            keyframe_path = row.keyframe_path
            
            # Create a pseudo-segment or real segment
            segment = TranscriptSegment(
                id=segment_id,
                video_id=video_id,
                start_time=start,
                end_time=end,
                text=segment_text if segment_id else f"{segment_text} (Scene {scene_val})",
            )
            # Store video filename and path in custom attributes
            segment.video_filename = filename
            segment.video_path = file_path
            
            # Use segment_id if available, otherwise a unique key based on scene
            key = segment_id if segment_id else f"scene_{scene_val}_{video_id}"
            semantic_scores[key] = (similarity, segment, keyframe_path)

        return semantic_scores

    def _fuzzy_text_search(
        self, query: str, top_k: int = 20, video_filter: Optional[str] = None
    ) -> Dict[int, Tuple[float, TranscriptSegment]]:
        """
        Fuzzy text search using PostgreSQL full-text search.
        
        NOW SEARCHES BOTH:
        - Transcript text (spoken words)
        - OCR text from keyframes (visible text like "Deepsea Stavanger")

        Returns:
            Dict mapping segment_id -> (score, segment)
        """
        # PostgreSQL full-text search
        query_filter = ""
        if video_filter:
            query_filter = "AND v.filename = :video_filter"

        # Enhanced query: searches transcript AND scene OCR text
        sql_query = text(f"""
            SELECT 
                ts.id,
                ts.video_id,
                v.filename,
                v.file_path,
                ts.start_time,
                ts.end_time,
                ts.text,
                -- Combine scores from transcript and OCR (OCR weighted slightly lower)
                GREATEST(
                    ts_rank(to_tsvector('simple', ts.text), plainto_tsquery('simple', :query)),
                    COALESCE(
                        ts_rank(to_tsvector('simple', s.ocr_text), plainto_tsquery('simple', :query)) * 0.9,
                        0
                    )
                ) AS rank,
                s.ocr_text as scene_ocr,
                ve.keyframe_path
            FROM transcript_segments ts
            JOIN videos v ON ts.video_id = v.id
            LEFT JOIN scenes s ON ts.scene_id = s.id
            LEFT JOIN visual_embeddings ve ON s.id = ve.scene_id
            WHERE (
                to_tsvector('simple', ts.text) @@ plainto_tsquery('simple', :query)
                OR to_tsvector('simple', COALESCE(s.ocr_text, '')) @@ plainto_tsquery('simple', :query)
            )
            {query_filter}
            ORDER BY rank DESC
            LIMIT :top_k
        """)

        params = {"query": query, "top_k": top_k}
        if video_filter:
            params["video_filter"] = video_filter

        results = self.db.execute(sql_query, params).fetchall()

        fuzzy_scores = {}
        for row in results:
            segment_id, video_id, filename, file_path, start, end, segment_text, rank, ocr_text, keyframe_path = row
            segment = TranscriptSegment(
                id=segment_id,
                video_id=video_id,
                start_time=start,
                end_time=end,
                text=segment_text,
            )
            segment.video_filename = filename
            segment.video_path = file_path
            
            # If match came from OCR, append hint to text
            if ocr_text and query.lower() in ocr_text.lower():
                segment.text = f"{segment_text} [OCR: {ocr_text[:100]}]"
            
            fuzzy_scores[segment_id] = (rank, segment, keyframe_path)

        return fuzzy_scores

    def _combine_results(
        self,
        semantic_results: Dict,
        fuzzy_results: Dict,
        semantic_weight: float = 0.7,
        text_weight: float = 0.3,
    ) -> List[SearchResult]:
        """Combine semantic and fuzzy search results with weighted scoring."""
        # Get all unique segment IDs
        all_segment_ids = set(semantic_results.keys()) | set(fuzzy_results.keys())

        combined = []
        for key in all_segment_ids:
            # Get scores and keyframe path
            semantic_entry = semantic_results.get(key, (0, None, None))
            fuzzy_entry = fuzzy_results.get(key, (0, None, None))
            
            semantic_score = semantic_entry[0]
            fuzzy_score = fuzzy_entry[0]

            # Get segment
            segment = semantic_entry[1]
            if segment is None:
                segment = fuzzy_entry[1]
                
            # Get keyframe (prefer visual match if available)
            keyframe_path = semantic_entry[2]
            if not keyframe_path:
                keyframe_path = fuzzy_entry[2]
            
            segment_id = segment.id if segment.id else 0

            # Normalize fuzzy score (ts_rank is usually small, so boost it)
            fuzzy_score_norm = min(fuzzy_score * 10.0, 1.0)

            # Combined score
            combined_score = (
                semantic_weight * semantic_score + text_weight * fuzzy_score_norm
            )

            # Determine match type
            if semantic_score > 0.7:
                match_type = "semantic"
            elif fuzzy_score > 0.5:
                match_type = "fuzzy"
            else:
                match_type = "hybrid"

            result = SearchResult(
                segment_id=segment_id,
                video_id=segment.video_id,
                video_filename=segment.video_filename,
                video_path=segment.video_path,
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=segment.text,
                score=combined_score,
                match_type=match_type,
                keyframe_path=keyframe_path or ""
            )

            combined.append(result)

        return combined

    def _log_query(self, query_text: str, query_embedding: List, results_count: int, top_result_id: int):
        """Log search query for analytics."""
        try:
            query_log = SearchQuery(
                query_text=query_text,
                query_embedding=query_embedding,
                results_count=results_count,
                top_result_id=top_result_id,
            )
            self.db.add(query_log)
            self.db.commit()
        except Exception as e:
            print(f"Warning: Failed to log query: {e}")
            self.db.rollback()

    def search_exact_phrase(
        self, phrase: str, video_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for exact phrase match.

        Args:
            phrase: Exact phrase to search for
            video_filter: Optional video filename filter

        Returns:
            List of SearchResult objects
        """
        query = self.db.query(TranscriptSegment, Video).join(
            Video, TranscriptSegment.video_id == Video.id
        )

        # Case-insensitive exact match
        query = query.filter(TranscriptSegment.text.ilike(f"%{phrase}%"))

        if video_filter:
            query = query.filter(Video.filename == video_filter)

        results = query.all()

        search_results = []
        for segment, video in results:
            result = SearchResult(
                segment_id=segment.id,
                video_id=segment.video_id,
                video_filename=video.filename,
                video_path=video.file_path,
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=segment.text,
                score=1.0,  # Exact match
                match_type="exact",
            )
            search_results.append(result)

        return search_results
    
    # OPTIMIZATION #5: Caching methods
    def _cache_key(self, query: str, top_k: int, **kwargs) -> str:
        """Generate cache key from query parameters."""
        cache_data = {
            'query': query.lower().strip(),
            'top_k': top_k,
            **kwargs
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _check_memory_cache(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Check in-memory cache (fastest)."""
        if cache_key in self._memory_cache:
            results, timestamp = self._memory_cache[cache_key]
            
            if datetime.now() - timestamp < self.cache_ttl:
                self.stats['cache_hits'] += 1
                self.stats['memory_hits'] += 1
                return results
            else:
                # Expired - remove
                del self._memory_cache[cache_key]
        
        return None
    
    def _check_db_cache(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Check database cache (persistent)."""
        try:
            result = self.db.execute(text("""
                SELECT cached_results 
                FROM query_cache 
                WHERE query_hash = :cache_key 
                AND expires_at > NOW()
                LIMIT 1
            """), {'cache_key': cache_key})
            
            row = result.fetchone()
            if row:
                # Update hit count
                self.db.execute(text("SELECT update_cache_stats(:cache_key)"), 
                              {'cache_key': cache_key})
                self.db.commit()
                
                # Deserialize results
                cached_data = row[0]
                results = [SearchResult(**item) for item in cached_data]
                
                # Add to memory cache
                self._memory_cache[cache_key] = (results, datetime.now())
                
                self.stats['cache_hits'] += 1
                self.stats['db_hits'] += 1
                return results
                
        except Exception as e:
            # If cache table doesn't exist yet, silently skip
            pass
        
        return None
    
    def _save_to_cache(self, cache_key: str, query: str, results: List[SearchResult], params: Dict):
        """Save results to both memory and database cache."""
        # Memory cache
        self._memory_cache[cache_key] = (results, datetime.now())
        
        # Enforce max size (LRU eviction)
        if len(self._memory_cache) > self.max_cache_size:
            sorted_cache = sorted(self._memory_cache.items(), key=lambda x: x[1][1])
            self._memory_cache = dict(sorted_cache[-self.max_cache_size:])
        
        # Database cache (if table exists)
        try:
            serialized = [r.to_dict() for r in results]
            
            self.db.execute(text("""
                INSERT INTO query_cache (
                    query_text, query_hash, query_params, cached_results, expires_at
                )
                VALUES (
                    :query_text, :query_hash, :query_params::jsonb, 
                    :cached_results::jsonb, NOW() + :ttl_interval::interval
                )
                ON CONFLICT (query_hash) DO UPDATE
                SET cached_results = EXCLUDED.cached_results,
                    hit_count = query_cache.hit_count + 1,
                    last_used = NOW(),
                    expires_at = NOW() + :ttl_interval::interval
            """), {
                'query_text': query,
                'query_hash': cache_key,
                'query_params': json.dumps(params),
                'cached_results': json.dumps(serialized),
                'ttl_interval': f'{self.cache_ttl.total_seconds()} seconds'
            })
            self.db.commit()
            
        except Exception as e:
            # If cache table doesn't exist, skip DB caching
            self.db.rollback()
    
    def _update_latency_stats(self, latency_ms: float):
        """Update average latency statistics."""
        n = self.stats['queries']
        current_avg = self.stats['avg_latency_ms']
        self.stats['avg_latency_ms'] = (current_avg * (n - 1) + latency_ms) / n if n > 0 else latency_ms
    
    def get_stats(self) -> Dict:
        """Get performance statistics."""
        total = self.stats['queries']
        hit_rate = self.stats['cache_hits'] / total if total > 0 else 0
        
        return {
            'total_queries': total,
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': f"{hit_rate:.1%}",
            'memory_hits': self.stats['memory_hits'],
            'db_hits': self.stats['db_hits'],
            'avg_latency_ms': round(self.stats['avg_latency_ms'], 2),
            'memory_cache_size': len(self._memory_cache),
            'parallel_enabled': self.parallel_enabled,
            'cache_enabled': self.cache_enabled,
        }
    
    def clear_cache(self, memory_only: bool = False):
        """Clear cache (useful for testing)."""
        self._memory_cache.clear()
        
        if not memory_only:
            try:
                self.db.execute(text("TRUNCATE TABLE query_cache"))
                self.db.commit()
            except:
                pass
    
    def cleanup_expired_cache(self):
        """Remove expired entries from database cache."""
        try:
            result = self.db.execute(text("SELECT clean_query_cache()"))
            deleted = result.fetchone()[0]
            self.db.commit()
            return deleted
        except:
            return 0
    
    def __del__(self):
        """Cleanup thread pool."""
        if self._executor:
            self._executor.shutdown(wait=False)

