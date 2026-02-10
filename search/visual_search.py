"""Pure visual search and enhanced multi-modal search."""
import numpy as np
from typing import List, Optional, Union
from pathlib import Path
from PIL import Image
from sqlalchemy import text
from sqlalchemy.orm import Session

from search.semantic_search import SearchResult
from embeddings.vision_embeddings import get_vision_embedding_generator


class VisualSearchEngine:
    """Pure visual search - find images matching a text query."""
    
    def __init__(self, db: Session):
        """
        Initialize visual search engine.
        
        Args:
            db: Database session
        """
        self.db = db
        self.vision_model = get_vision_embedding_generator()
    
    def search_visual(
        self,
        query: str,
        top_k: int = 10,
        video_filter: Optional[str] = None,
        min_score: float = 0.01
    ) -> List[SearchResult]:
        """
        Search for visual content matching the query.
        
        This IGNORES the transcript and searches purely by visual similarity!
        Perfect for queries like:
        - "picture of an oil rig"
        - "image of safety equipment"
        - "show me drilling operations"
        
        Args:
            query: Text description of what to find visually
            top_k: Number of results
            video_filter: Optional video filename filter
            min_score: Minimum similarity score
            
        Returns:
            List of SearchResult objects with visual matches
        """
        print(f"🔍 Visual Search (SigLIP): '{query}'")
        
        # 1. Encode the text query as a vision embedding
        query_embedding = self.vision_model.encode_text(query, normalize=True)
        
        # 2. Search visual_embeddings table
        results = self._execute_visual_query(query_embedding, top_k, video_filter, min_score)
        print(f"Found {len(results)} visual matches")
        return results

    def search_by_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        top_k: int = 10,
        video_filter: Optional[str] = None,
        min_score: float = 0.01
    ) -> List[SearchResult]:
        """
        Reverse image search: find similar moments in videos based on an input image.
        
        Args:
            image_input: Query image (path, bytes, or PIL Image)
            top_k: Number of results
            video_filter: Optional video filename filter
            min_score: Minimum similarity score
            
        Returns:
            List of SearchResult objects
        """
        print("🔍 Reverse Image Search (SigLIP)")
        
        # 1. Encode the image query
        query_embedding = self.vision_model.encode_image(image_input, normalize=True)
        
        # 2. Re-use the same search logic as text-to-visual
        return self._execute_visual_query(query_embedding, top_k, video_filter, min_score)

    def _execute_visual_query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        video_filter: Optional[str] = None,
        min_score: float = 0.01
    ) -> List[SearchResult]:
        """Internal helper to execute vector search on visual_embeddings."""
        query_filter = ""
        if video_filter:
            query_filter = "AND v.filename = :video_filter"
        
        sql_query = text(f"""
            SELECT 
                ts.id as segment_id,
                ts.video_id,
                v.filename,
                v.file_path,
                ts.start_time,
                ts.end_time,
                ts.text,
                s.scene_id,
                ve.keyframe_path,
                1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS visual_similarity
            FROM visual_embeddings ve
            JOIN scenes s ON ve.scene_id = s.id
            JOIN videos v ON s.video_id = v.id
            -- Get corresponding transcript segment
            LEFT JOIN transcript_segments ts ON (
                ts.video_id = v.id
                AND ts.start_time <= s.start_time + 1
                AND ts.end_time >= s.start_time - 1
            )
            WHERE 1=1 {query_filter}
            ORDER BY ve.embedding <=> CAST(:query_embedding AS vector)
            LIMIT :top_k
        """)
        
        params = {
            'query_embedding': query_embedding.tolist(),
            'top_k': top_k * 2
        }
        
        if video_filter:
            params['video_filter'] = video_filter
            
        try:
            result = self.db.execute(sql_query, params)
            rows = result.fetchall()
            
            results = []
            for row in rows:
                if row.visual_similarity < min_score:
                    continue
                
                result = SearchResult(
                    segment_id=row.segment_id if row.segment_id else 0,
                    video_id=row.video_id,
                    video_filename=row.filename,
                    video_path=row.file_path,
                    start_time=row.start_time if row.start_time else 0,
                    end_time=row.end_time if row.end_time else 0,
                    text=row.text if row.text else f"[Visual match: scene {row.scene_id}]",
                    score=float(row.visual_similarity),
                    match_type="visual",
                    keyframe_path=row.keyframe_path or ""
                )
                results.append(result)
            
            return results[:top_k]
        except Exception as e:
            print(f"  ✗ Visual search execution error: {e}")
            return []
    
    def search_visual_only_scenes(
        self,
        query: str,
        top_k: int = 10,
        video_filter: Optional[str] = None
    ) -> List[dict]:
        """
        Search scenes directly without needing transcript segments.
        Returns raw scene data.
        """
        try:
            query_embedding = self.vision_model.encode_text(query, normalize=True)
            
            query_filter = ""
            if video_filter:
                query_filter = "AND v.filename = :video_filter"
            
            sql_query = text(f"""
                SELECT 
                    v.id as video_id,
                    v.filename,
                    v.file_path,
                    s.scene_id,
                    s.start_time,
                    s.end_time,
                    ve.keyframe_path,
                    1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS similarity
                FROM visual_embeddings ve
                JOIN scenes s ON ve.scene_id = s.id  
                JOIN videos v ON s.video_id = v.id
                WHERE 1=1 {query_filter}
                ORDER BY ve.embedding <=> CAST(:query_embedding AS vector)
                LIMIT :top_k
            """)
            
            params = {
                'query_embedding': query_embedding.tolist(),
                'top_k': top_k
            }
            
            if video_filter:
                params['video_filter'] = video_filter
            
            result = self.db.execute(sql_query, params)
            rows = result.fetchall()
            
            return [
                {
                    'video_id': row.video_id,
                    'video_filename': row.filename,
                    'video_path': row.file_path,
                    'scene_id': row.scene_id,
                    'start_time': row.start_time,
                    'end_time': row.end_time,
                    'keyframe_path': row.keyframe_path,
                    'similarity': float(row.similarity)
                }
                for row in rows
            ]
        except Exception as e:
            print(f"  ✗ Visual scene search error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def search_by_image_and_text(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        text_query: str,
        top_k: int = 10,
        video_filter: Optional[str] = None,
        min_score: float = 0.15,
        image_weight: float = 0.7,
        text_weight: float = 0.3
    ) -> List[SearchResult]:
        """
        Combined image + text search: fuses image and text embeddings
        for more precise visual search.
        
        Args:
            image_input: Query image (path, bytes, or PIL Image)
            text_query: Additional text to refine the search
            top_k: Number of results
            video_filter: Optional video filename filter
            min_score: Minimum similarity score
            image_weight: Weight for image embedding (0-1)
            text_weight: Weight for text embedding (0-1)
            
        Returns:
            List of SearchResult objects
        """
        print(f"🔍 Combined Image+Text Search (SigLIP) (image={image_weight}, text={text_weight})")
        
        # Encode both inputs
        image_embedding = self.vision_model.encode_image(image_input, normalize=True)
        text_embedding = self.vision_model.encode_text(text_query, normalize=True)
        
        # Fuse embeddings via weighted average
        combined = image_weight * image_embedding + text_weight * text_embedding
        # Re-normalize
        combined = combined / np.linalg.norm(combined)
        
        return self._execute_visual_query(combined, top_k, video_filter, min_score)


class HybridSearchEngine:
    """Truly hybrid search: combines text, semantic, AND visual search."""
    
    def __init__(
        self,
        db: Session,
        text_weight: float = 0.33,
        semantic_weight: float = 0.33,
        visual_weight: float = 0.34
    ):
        """
        Initialize hybrid search engine.
        
        Args:
            db: Database session
            text_weight: Weight for fuzzy text matching
            semantic_weight: Weight for semantic similarity
            visual_weight: Weight for visual similarity
        """
        self.db = db
        self.text_weight = text_weight
        self.semantic_weight = semantic_weight
        self.visual_weight = visual_weight
        
        # Import here to avoid circular imports
        from search.semantic_search import SemanticSearchEngine
        self.semantic_engine = SemanticSearchEngine(db)
        self.visual_engine = VisualSearchEngine(db)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        video_filter: Optional[str] = None,
        auto_mode: bool = True
    ) -> List[SearchResult]:
        """
        Unified search combining ALL modalities.
        
        Automatically detects query type:
        - Visual queries: "picture of", "image of", "show me" → Higher visual weight
        - Text queries: "discussed", "mentioned", "talked about" → Higher text weight
        - Balanced queries: "oil rig", "drilling" → Equal weights
        
        Args:
            query: Search query
            top_k: Number of results
            video_filter: Optional video filter
            auto_mode: Automatically adjust weights based on query
            
        Returns:
            Unified ranked results from all search modalities
        """
        # Auto-detect query type
        if auto_mode:
            query_lower = query.lower()
            if any(word in query_lower for word in ['picture', 'image', 'photo', 'show', 'visual', 'scene']):
                # Visual-heavy query
                text_w, semantic_w, visual_w = 0.1, 0.2, 0.7
                print(f"📷 Detected visual query → weights: text={text_w}, semantic={semantic_w}, visual={visual_w}")
            elif any(word in query_lower for word in ['discussed', 'mentioned', 'talked', 'said', 'speaking']):
                # Text-heavy query
                text_w, semantic_w, visual_w = 0.4, 0.5, 0.1
                print(f"💬 Detected text query → weights: text={text_w}, semantic={semantic_w}, visual={visual_w}")
            else:
                # Balanced
                text_w, semantic_w, visual_w = self.text_weight, self.semantic_weight, self.visual_weight
                print(f"⚖️ Balanced query → weights: text={text_w}, semantic={semantic_w}, visual={visual_w}")
        else:
            text_w, semantic_w, visual_w = self.text_weight, self.semantic_weight, self.visual_weight
        
        # Normalize weights so text+semantic and visual sum to 1.0
        text_semantic_w = text_w + semantic_w
        total_w = text_semantic_w + visual_w
        text_semantic_w /= total_w
        visual_w /= total_w
        
        # 1. Get semantic + text results (existing hybrid search)
        text_results = self.semantic_engine.search(
            query=query,
            top_k=top_k * 3,
            video_filter=video_filter,
            log_query=False  # Don't log intermediate searches
        )
        
        # 2. Get pure visual results
        visual_results = self.visual_engine.search_visual(
            query=query,
            top_k=top_k * 3,
            video_filter=video_filter
        )
        
        # 3. Merge and re-rank by combining scores
        #    text_results already have blended text+semantic scores, so we
        #    just scale them by the text_semantic_w share.
        merged = {}
        
        for result in text_results:
            key = (result.video_id, result.segment_id, result.start_time)
            if key not in merged:
                merged[key] = {
                    'result': result,
                    'text_score': result.score * text_semantic_w,
                    'visual_score': 0.0
                }
        
        for result in visual_results:
            key = (result.video_id, result.segment_id, result.start_time)
            if key in merged:
                merged[key]['visual_score'] = result.score * visual_w
            else:
                merged[key] = {
                    'result': result,
                    'text_score': 0.0,
                    'visual_score': result.score * visual_w
                }
        
        # Calculate combined scores and sort
        final_results = []
        for key, data in merged.items():
            result = data['result']
            combined_score = data['text_score'] + data['visual_score']
            result.score = combined_score
            final_results.append(result)
        
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        print(f"  ✓ Combined {len(text_results)} text + {len(visual_results)} visual → {len(final_results)} total")
        
        return final_results[:top_k]


def create_visual_search_engine(db: Session) -> VisualSearchEngine:
    """Convenience function to create visual search engine."""
    return VisualSearchEngine(db)


def create_hybrid_search_engine(db: Session) -> HybridSearchEngine:
    """Convenience function to create hybrid search engine."""
    return HybridSearchEngine(db)
