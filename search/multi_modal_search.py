"""Multi-modal search combining text (transcript) and vision (keyframes) similarity."""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import text

from search.semantic_search import SemanticSearchEngine, SearchResult
from embeddings.vision_embeddings import get_vision_embedding_generator


@dataclass
class MultiModalSearchResult(SearchResult):
    """Search result with both text and vision scores."""
    vision_score: float = 0.0
    combined_score: float = 0.0
    # keyframe_path is inherited from SearchResult
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            "vision_score": round(self.vision_score, 4),
            "combined_score": round(self.combined_score, 4)
        })
        return base_dict


class MultiModalSearchEngine:
    """Search engine combining text and vision embeddings."""
    
    def __init__(
        self,
        db: Session,
        text_weight: float = 0.5,
        vision_weight: float = 0.5,
        vision_model: str = "google/siglip-base-patch16-224"
    ):
        """
        Initialize multi-modal search engine.
        
        Args:
            db: Database session
            text_weight: Weight for text similarity (0-1)
            vision_weight: Weight for vision similarity (0-1)
            vision_model: Vision model name (SigLIP)
        """
        self.db = db
        self.text_weight = text_weight
        self.vision_weight = vision_weight
        
        # Initialize text search engine
        self.text_search = SemanticSearchEngine(db)
        
        # Lazy load vision model (only when needed)
        self._vision_gen = None
        self.vision_model_name = vision_model
        
        # Validate weights
        if not np.isclose(text_weight + vision_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {text_weight + vision_weight}")
    
    @property
    def vision_gen(self):
        """Lazy load vision embedding generator."""
        if self._vision_gen is None:
            self._vision_gen = get_vision_embedding_generator(self.vision_model_name)
        return self._vision_gen
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        video_filter: Optional[str] = None,
        use_vision: bool = True
    ) -> List[MultiModalSearchResult]:
        """
        Perform multi-modal search.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            video_filter: Filter by specific video filename
            use_vision: Whether to include vision similarity
        
        Returns:
            List of multi-modal search results ranked by combined score
        """
        # 1. Get Text Candidates
        text_results = self.text_search.search(
            query=query,
            top_k=top_k * 5 if use_vision else top_k,
            video_filter=video_filter
        )
        
        if not use_vision or self.vision_weight == 0:
            return [
                MultiModalSearchResult(
                    **result.__dict__,
                    vision_score=0.0,
                    combined_score=result.score
                )
                for result in text_results[:top_k]
            ]
            
        # 2. Get Visual Candidates (Query Text -> Vision Embeddings)
        visual_candidates = self.search_visual_only(
            query=query,
            top_k=top_k * 2,
            video_filter=video_filter
        )
        
        # 3. Merge Candidates
        candidate_map = {} # key -> (text_score, vision_score, SearchResult)
        
        # Add text candidates
        for r in text_results:
            key = f"seg_{r.segment_id}" if r.segment_id else f"scene_{r.video_id}_{r.start_time}"
            candidate_map[key] = [r.score, 0.0, r]
            
        # Add visual candidates (or update existing)
        for r in visual_candidates:
            key = f"seg_{r.segment_id}" if r.segment_id else f"scene_{r.video_id}_{r.start_time}"
            if key in candidate_map:
                candidate_map[key][1] = r.vision_score
            else:
                candidate_map[key] = [0.0, r.vision_score, r]
                
        # 4. Fill missing vision scores for text candidates
        query_vision_embedding = self.vision_gen.encode_text(query, normalize=True)
        
        final_results = []
        for key, (t_score, v_score, base_result) in candidate_map.items():
            # If we don't have a vision score yet, fetch it from DB
            current_v_score = v_score
            current_keyframe = base_result.keyframe_path
            
            if current_v_score == 0.0:
                vision_data = self._get_vision_embedding_for_segment(base_result.segment_id)
                if vision_data:
                    emb, path = vision_data
                    current_v_score = float(np.dot(query_vision_embedding, emb))
                    if not current_keyframe:
                        current_keyframe = path
            
            # Combine scores
            combined_score = (self.text_weight * t_score) + (self.vision_weight * current_v_score)
            
            # Create MultiModal result (safely handling existing dict keys)
            res_data = base_result.__dict__.copy()
            res_data.update({
                "vision_score": current_v_score,
                "combined_score": combined_score,
                "score": combined_score, # Update base score for sorting
                "keyframe_path": current_keyframe
            })
            
            mm_res = MultiModalSearchResult(**res_data)
            final_results.append(mm_res)
            
    # 5. Sort and return
        final_results.sort(key=lambda x: x.combined_score, reverse=True)
        return final_results[:top_k]

    def search_with_fallback(
        self,
        query: str,
        top_k: int = 10,
        video_filter: Optional[str] = None,
    ) -> dict:
        """
        Multi-modal search with tiered fallback strategy.
        
        Uses the text search engine's fallback tiers, then enriches
        with vision scores. Auto-adjusts text/vision weights when
        text results are low confidence to avoid score dilution.
        
        Returns:
            Dict with 'results', 'search_metadata'
        """
        # 1. Get text results with fallback
        fallback_data = self.text_search.search_with_fallback(
            query=query,
            top_k=top_k * 3 if self.vision_weight > 0 else top_k,
            video_filter=video_filter,
        )
        
        text_results = fallback_data["results"]
        metadata = fallback_data["search_metadata"]
        
        # If no vision needed or no text results, return as-is
        if self.vision_weight == 0 or not text_results:
            mm_results = [
                MultiModalSearchResult(
                    **r.__dict__,
                    vision_score=0.0,
                    combined_score=r.score
                )
                for r in text_results[:top_k]
            ]
            return {"results": mm_results, "search_metadata": metadata}
        
        # 2. Auto-adjust weights: if text results are weak, favor text more
        #    to avoid vision scores diluting already-weak text matches
        top_text_score = text_results[0].score if text_results else 0
        effective_text_weight = self.text_weight
        effective_vision_weight = self.vision_weight
        
        if top_text_score < 0.3:
            # Low confidence text — go text-heavy to preserve what we found
            effective_text_weight = 0.7
            effective_vision_weight = 0.3
        
        # 3. Enrich with vision scores
        try:
            query_vision_embedding = self.vision_gen.encode_text(query, normalize=True)
        except Exception:
            # Vision unavailable — return text-only results
            mm_results = [
                MultiModalSearchResult(
                    **r.__dict__,
                    vision_score=0.0,
                    combined_score=r.score
                )
                for r in text_results[:top_k]
            ]
            return {"results": mm_results, "search_metadata": metadata}
        
        # Also get visual-only candidates
        try:
            visual_candidates = self.search_visual_only(
                query=query,
                top_k=top_k * 2,
                video_filter=video_filter
            )
        except Exception:
            visual_candidates = []
        
        # Build candidate map
        candidate_map = {}
        for r in text_results:
            key = f"seg_{r.segment_id}" if r.segment_id else f"scene_{r.video_id}_{r.start_time}"
            candidate_map[key] = [r.score, 0.0, r]
        
        for r in visual_candidates:
            key = f"seg_{r.segment_id}" if r.segment_id else f"scene_{r.video_id}_{r.start_time}"
            if key in candidate_map:
                candidate_map[key][1] = r.vision_score
            else:
                candidate_map[key] = [0.0, r.vision_score, r]
        
        # Fill missing vision scores
        final_results = []
        for key, (t_score, v_score, base_result) in candidate_map.items():
            current_v_score = v_score
            current_keyframe = base_result.keyframe_path
            
            if current_v_score == 0.0 and hasattr(base_result, 'segment_id') and base_result.segment_id:
                vision_data = self._get_vision_embedding_for_segment(base_result.segment_id)
                if vision_data:
                    emb, path = vision_data
                    current_v_score = float(np.dot(query_vision_embedding, emb))
                    if not current_keyframe:
                        current_keyframe = path
            
            combined_score = (effective_text_weight * t_score) + (effective_vision_weight * current_v_score)
            
            res_data = base_result.__dict__.copy()
            res_data.update({
                "vision_score": current_v_score,
                "combined_score": combined_score,
                "score": combined_score,
                "keyframe_path": current_keyframe
            })
            
            # Remove any keys that don't belong to MultiModalSearchResult
            for extra_key in list(res_data.keys()):
                if extra_key not in MultiModalSearchResult.__dataclass_fields__:
                    del res_data[extra_key]
            
            mm_res = MultiModalSearchResult(**res_data)
            final_results.append(mm_res)
        
        final_results.sort(key=lambda x: x.combined_score, reverse=True)
        return {"results": final_results[:top_k], "search_metadata": metadata}
    
    def _get_vision_embedding_for_segment(self, segment_id: int) -> Optional[tuple]:
        """
        Get vision embedding for a segment via its scene.
        
        Returns:
            Tuple of (embedding_array, keyframe_path) or None
        """
        result = self.db.execute(text("""
            SELECT ve.embedding, ve.keyframe_path
            FROM transcript_segments ts
            JOIN scenes s ON ts.scene_id = s.id
            JOIN visual_embeddings ve ON s.id = ve.scene_id
            WHERE ts.id = :segment_id
            AND ve.embedding_model = :model_name
            LIMIT 1
        """), {"segment_id": segment_id, "model_name": self.vision_model_name})
        
        row = result.fetchone()
        if row:
            raw_embedding = row[0]
            if isinstance(raw_embedding, str):
                # Handle cases where DB returns vector as string like "[1,2,3]"
                import json
                try:
                    embedding = np.array(json.loads(raw_embedding), dtype=np.float32)
                except Exception:
                    # Alternative parsing if json.loads fails (e.g. pgvector string format)
                    cleaned = raw_embedding.replace('[', '').replace(']', '').split(',')
                    embedding = np.array([float(x.strip()) for x in cleaned if x.strip()], dtype=np.float32)
            else:
                embedding = np.array(raw_embedding, dtype=np.float32)
                
            # Normalize if not already (safeguard)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding, row[1]
        
        return None
    
    def search_visual_only(
        self,
        query: str,
        top_k: int = 10,
        video_filter: Optional[str] = None
    ) -> List[MultiModalSearchResult]:
        """
        Perform vision-only search (query text → find similar keyframes).
        
        Args:
            query: Search query text
            top_k: Number of results to return
            video_filter: Filter by specific video filename
        
        Returns:
            List of results ranked by visual similarity
        """
        # Generate vision embedding for query
        query_vision_embedding = self.vision_gen.encode_text(query, normalize=True)
        
        # Search visual embeddings database
        query_filter = ""
        params = {
            "query_embedding": query_vision_embedding.tolist(),
            "top_k": top_k,
            "model_name": self.vision_model_name
        }
        
        if video_filter:
            query_filter = "AND v.filename = :video_filter"
            params["video_filter"] = video_filter
        
        sql_query = f"""
            SELECT 
                ts.id as segment_id,
                v.id as video_id,
                v.filename as video_filename,
                v.file_path as video_path,
                COALESCE(ts.start_time, s.start_time) as start_time,
                COALESCE(ts.end_time, s.end_time) as end_time,
                COALESCE(ts.text, '[Visual match]') as text,
                ve.keyframe_path,
                1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS similarity
            FROM visual_embeddings ve
            JOIN scenes s ON ve.scene_id = s.id
            JOIN videos v ON s.video_id = v.id
            LEFT JOIN transcript_segments ts ON ts.scene_id = s.id
            WHERE ve.embedding_model = :model_name
            {query_filter}
            ORDER BY ve.embedding <=> CAST(:query_embedding AS vector)
            LIMIT :top_k
        """
        
        result = self.db.execute(text(sql_query), params)
        
        results = []
        for row in result.fetchall():
            res = MultiModalSearchResult(
                segment_id=row.segment_id,
                video_id=row.video_id,
                video_filename=row.video_filename,
                video_path=row.video_path,
                start_time=row.start_time,
                end_time=row.end_time,
                text=row.text,
                score=0.0,
                vision_score=float(row.similarity),
                combined_score=float(row.similarity),
                match_type="visual"
            )
            res.keyframe_path = row.keyframe_path
            results.append(res)
        
        return results


def set_optimal_weights(search_type: str = "balanced") -> tuple:
    """
    Get optimal text/vision weights for different search scenarios.
    
    Args:
        search_type: One of "balanced", "text_heavy", "vision_heavy", "visual_only"
    
    Returns:
        Tuple of (text_weight, vision_weight)
    """
    weights = {
        "balanced": (0.5, 0.5),
        "text_heavy": (0.7, 0.3),
        "vision_heavy": (0.3, 0.7),
        "visual_only": (0.0, 1.0),
        "text_only": (1.0, 0.0)
    }
    
    return weights.get(search_type, (0.5, 0.5))
