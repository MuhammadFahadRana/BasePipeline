import sys
import os
from pathlib import Path
sys.path.insert(0, '.')

from api.app import app
from search.multi_modal_search import MultiModalSearchEngine
from database.config import SessionLocal

def test_visual_scores_for_fuzzy():
    print("Testing Visual Scores for Fuzzy/OCR Matches...")
    db = SessionLocal()
    try:
        # Balanced weights to trigger both scores
        mm_search = MultiModalSearchEngine(db, text_weight=0.5, vision_weight=0.5)
        
        # 'Oil rig' usually triggers fuzzy/OCR matches which have result_id < 0
        results_data = mm_search.search_with_fallback("oil rig", top_k=5)
        results = results_data["results"]
        
        print(f"Total results: {len(results)}")
        for i, r in enumerate(results):
            # Check for result_id if it exists
            rid = getattr(r, 'result_id', 'N/A')
            print(f"[{i}] Text: {r.text[:40]}...")
            print(f"    Result ID: {rid}")
            print(f"    Text Score: {r.text_score:.4f}")
            print(f"    Vision Score: {r.vision_score:.4f}")
            print(f"    Combined Score: {r.combined_score:.4f}")
            
            # Verify scores are non-zero if embeddings exist
            if r.vision_score == 0:
                 print("    ⚠ Warning: Vision score is ZERO")
            if r.text_score == 0:
                 print("    ⚠ Warning: Text score is ZERO")
                 
    finally:
        db.close()

if __name__ == "__main__":
    test_visual_scores_for_fuzzy()
