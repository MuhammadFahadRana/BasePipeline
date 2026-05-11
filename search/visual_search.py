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
        print(f"Visual Search ({self.vision_model.model_name}): '{query}'")
        
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
        print(f"Reverse Image Search ({self.vision_model.model_name})")
        
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
            WITH ranked AS (
                SELECT
                    ts.id as segment_id,
                    v.id as video_id,
                    v.filename,
                    v.file_path,
                    COALESCE(ts.start_time, s.start_time) as start_time,
                    COALESCE(ts.end_time, s.end_time) as end_time,
                    COALESCE(ts.text, '[Visual match]') as text,
                    s.scene_id,
                    ve.keyframe_path,
                    ve.sample_time,
                    ve.frame_role,
                    1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS visual_similarity,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.id
                        ORDER BY ve.embedding <=> CAST(:query_embedding AS vector)
                    ) as rn
                FROM visual_embeddings ve
                JOIN scenes s ON ve.scene_id = s.id
                JOIN videos v ON s.video_id = v.id
                LEFT JOIN transcript_segments ts ON (
                    ts.video_id = v.id
                    AND ts.start_time <= s.start_time + 1
                    AND ts.end_time >= s.start_time - 1
                )
                WHERE ve.embedding_model = :model_name {query_filter}
            )
            SELECT * FROM ranked
            WHERE rn = 1
            ORDER BY visual_similarity DESC
            LIMIT :top_k
        """)
        
        params = {
            'query_embedding': query_embedding.tolist(),
            'top_k': top_k * 2,
            'model_name': self.vision_model.model_name,
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
                    keyframe_path=row.keyframe_path or "",
                    evidence_time=row.sample_time,
                    evidence_frame_role=row.frame_role,
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
                    ve.sample_time,
                    ve.frame_role,
                    1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS similarity
                FROM visual_embeddings ve
                JOIN scenes s ON ve.scene_id = s.id  
                JOIN videos v ON s.video_id = v.id
                WHERE ve.embedding_model = :model_name {query_filter}
                ORDER BY ve.embedding <=> CAST(:query_embedding AS vector)
                LIMIT :top_k
            """)
            
            params = {
                'query_embedding': query_embedding.tolist(),
                'top_k': top_k,
                'model_name': self.vision_model.model_name,
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
                    'sample_time': row.sample_time,
                    'frame_role': row.frame_role,
                    'similarity': float(row.similarity)
                }
                for row in rows
            ]
        except Exception as e:
            print(f"  ✗ Visual scene search error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def search_video_level(
        self,
        query: str,
        top_k: int = 10,
        video_filter: Optional[str] = None,
    ) -> List[dict]:
        """Search video-level visual embeddings for whole-video discovery."""
        try:
            query_embedding = self.vision_model.encode_text(query, normalize=True)
            video_embedding_model = f"video-temporal-mean:{self.vision_model.model_name}"

            query_filter = ""
            if video_filter:
                query_filter = "AND v.filename = :video_filter"

            sql_query = text(f"""
                SELECT
                    v.id AS video_id,
                    v.filename,
                    v.file_path,
                    v.duration_seconds,
                    ve.frame_count,
                    1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS similarity
                FROM video_embeddings ve
                JOIN videos v ON ve.video_id = v.id
                WHERE ve.embedding_model = :model_name
                {query_filter}
                ORDER BY ve.embedding <=> CAST(:query_embedding AS vector)
                LIMIT :top_k
            """)

            params = {
                "query_embedding": query_embedding.tolist(),
                "model_name": video_embedding_model,
                "top_k": top_k,
            }
            if video_filter:
                params["video_filter"] = video_filter

            rows = self.db.execute(sql_query, params).fetchall()
            return [
                {
                    "video_id": row.video_id,
                    "video_filename": row.filename,
                    "video_path": row.file_path,
                    "duration_seconds": row.duration_seconds,
                    "frame_count": row.frame_count,
                    "similarity": float(row.similarity),
                    "embedding_model": video_embedding_model,
                }
                for row in rows
            ]
        except Exception as e:
            print(f"  Video-level visual search error: {e}")
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
        print(
            f"Combined Image+Text Search ({self.vision_model.model_name}) "
            f"(image={image_weight}, text={text_weight})"
        )
        
        # Encode both inputs
        image_embedding = self.vision_model.encode_image(image_input, normalize=True)
        text_embedding = self.vision_model.encode_text(text_query, normalize=True)
        
        # Fuse embeddings via weighted average
        combined = image_weight * image_embedding + text_weight * text_embedding
        # Re-normalize
        combined = combined / np.linalg.norm(combined)
        
        return self._execute_visual_query(combined, top_k, video_filter, min_score)


def create_visual_search_engine(db: Session) -> VisualSearchEngine:
    """Convenience function to create visual search engine."""
    return VisualSearchEngine(db)
