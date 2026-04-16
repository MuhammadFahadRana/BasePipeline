"""Test exact image match."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.config import SessionLocal
from database.models import VisualEmbedding
from search.visual_search import VisualSearchEngine

def test_exact_match():
    db = SessionLocal()
    try:
        # Get one frame from DB
        ve = db.query(VisualEmbedding).first()
        if not ve:
            print("No visual embeddings found!")
            return
        
        frame_path = ve.keyframe_path
        print(f"Searching for exact frame: {frame_path}")
        
        visual_engine = VisualSearchEngine(db)
        # Force min_score to 0 to see what we get
        results = visual_engine.search_by_image(image_input=frame_path, top_k=3, min_score=0.0)
        
        print("\nResults for exact match:")
        for r in results:
            print(f"- {r.video_filename} [{r.start_time:.2f}s]: Score = {r.score:.4f} (Path: {r.keyframe_path})")
            if r.keyframe_path == frame_path:
                print("  ✓ Found the original frame!")
            
    finally:
        db.close()

if __name__ == "__main__":
    test_exact_match()
