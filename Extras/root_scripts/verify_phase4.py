"""
Verification script for Phase 4: Semantic Enrichment & QA System
"""

import os
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from database.config import SessionLocal, init_db
from database.models import Video, Scene
from extract_visual_features import VisualFeatureExtractor
from llm.video_qa import VideoQA
from search.semantic_search import SemanticSearchEngine

def verify_pipeline():
    print("\n--- Phase 4 Verification ---\n")
    
    # 1. Test Qwen2-VL Extraction
    test_image = "processed/scenes/video1/video1_scene_0.jpg"
    if not os.path.exists(test_image):
        print(f"Error: Test image {test_image} not found.")
        return
        
    print(f"Testing VisualFeatureExtractor on {test_image}...")
    try:
        extractor = VisualFeatureExtractor(load_in_4bit=True)
        result = extractor.analyze_image(test_image)
        print(f"✓ Extraction Result:\n  Caption: {result['caption']}\n  Labels: {result['object_labels']}")
    except Exception as e:
        print(f"✗ Extraction Failed: {e}")
        return

    # 2. Test Database Ingestion (Dry Run / Mock)
    print("\nVerifying Database Schema and Ingestion Logic...")
    db = SessionLocal()
    try:
        # Check if we can create a mock scene with new fields
        mock_scene = Scene(
            video_id=1, # Assume video 1 exists
            scene_id=999,
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            caption="Test caption of an oil rig",
            object_labels=["oil rig", "ocean", "sky"]
        )
        print("✓ Scene model successfully supports new fields.")
    except Exception as e:
        print(f"✗ Database Model Error: {e}")
    finally:
        db.close()

    # 3. Test Search Indexing
    print("\nTesting Enhanced Search (Indexing logic check)...")
    db = SessionLocal()
    try:
        search_engine = SemanticSearchEngine(db)
        # We search for something that might be in our mock/future data
        results = search_engine.search("oil rig", top_k=5)
        print(f"✓ Search successfully executed (Found {len(results)} results).")
    except Exception as e:
        print(f"✗ Search Error: {e}")
    finally:
        db.close()

    # 4. Test Video QA
    print("\nTesting Video QA System...")
    db = SessionLocal()
    try:
        qa = VideoQA(db)
        qa_result = qa.ask("What objects are visible in the video?")
        print(f"✓ QA Result:\n  Answer: {qa_result['answer']}")
        print(f"  Citations: {len(qa_result['citations'])}")
    except Exception as e:
        print(f"✗ QA Error: {e}")
    finally:
        db.close()

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    verify_pipeline()
