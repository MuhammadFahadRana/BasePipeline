"""Verify search functionality."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from database.config import SessionLocal
from search.semantic_search import SemanticSearchEngine
from search.visual_search import VisualSearchEngine

def verify_search():
    db = SessionLocal()
    try:
        print("\n--- Testing Semantic Search (Qwen3) ---")
        semantic_engine = SemanticSearchEngine(db)
        # Search for "technique" as requested by user
        results = semantic_engine.search("technique", top_k=5)
        print(f"Results for 'techniques': {len(results)}")
        for r in results:
            print(f"- {r.video_filename} [{r.start_time:.2f}s]: {r.text[:100]}... (Score: {r.score:.4f})")

        print("\n--- Testing Visual Search (SigLIP) ---")
        visual_engine = VisualSearchEngine(db)
        # Search for "technique" as requested by user
        results = visual_engine.search_visual("technique", top_k=5)
        print(f"Results for 'techniques': {len(results)}")
        for r in results:
            print(f"- {r.video_filename} [{r.start_time:.2f}s]: Score = {r.score:.4f}")

        print("\n--- Testing Combined Search ---")
        results = visual_engine.search_visual("drilling operation", top_k=3)
        print(f"Results for 'drilling operation': {len(results)}")
        for r in results:
            print(f"- {r.video_filename} [{r.start_time:.2f}s]: {r.text[:50]}... (Score: {r.score:.4f})")

    finally:
        db.close()

if __name__ == "__main__":
    verify_search()
