"""Debug visual similarity scores."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from database.config import SessionLocal
from embeddings.vision_embeddings import get_vision_embedding_generator

def debug_scores():
    db = SessionLocal()
    try:
        vision_gen = get_vision_embedding_generator()
        query = "oil rig"
        print(f"Encoding query: '{query}'")
        query_embedding = vision_gen.encode_text(query, normalize=True)
        
        sql = text("""
            SELECT 
                ve.id,
                v.filename,
                1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS visual_similarity
            FROM visual_embeddings ve
            JOIN scenes s ON ve.scene_id = s.id
            JOIN videos v ON s.video_id = v.id
            ORDER BY ve.embedding <=> CAST(:query_embedding AS vector)
            LIMIT 5
        """)
        
        result = db.execute(sql, {"query_embedding": query_embedding.tolist()})
        rows = result.fetchall()
        
        print("\nTop 5 Raw Results:")
        for row in rows:
            print(f"- {row.filename} (ID: {row.id}): Similarity = {row.visual_similarity}")
            
    finally:
        db.close()

if __name__ == "__main__":
    debug_scores()
