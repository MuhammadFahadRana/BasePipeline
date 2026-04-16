#!/usr/bin/env python3
"""
Quick start script for video processing and ingestion.

This script:
1. Processes all videos in the videos folder
2. Ingests them into the database
3. Starts the API server

Usage:
    python quick_start.py
"""

import sys
from pathlib import Path

print("="*60)
print("VIDEO SEMANTIC SEARCH - QUICK START")
print("="*60)

# Step 1: Process videos
print("\n[Step 1/3] Processing videos...")
print("-" * 60)

from basic_pipeline import BasicVideoPipeline

# Check if videos exist
video_dir = Path("videos")
video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))

if not video_files:
    print("⚠ No videos found in 'videos/' folder")
    print("Please add videos to process")
    sys.exit(1)

print(f"Found {len(video_files)} videos:")
for vf in video_files:
    print(f"  - {vf.name}")

# Initialize pipeline with Whisper base (good balance of speed/accuracy)
pipeline = BasicVideoPipeline(
    backend="whisper",
    model_variant={"name": "base", "description": "Fast, good for simple audio"},
    scene_threshold=20.0,
    device="auto"
)

# Process all videos (skip already processed)
try:
    results = pipeline.batch_process(
        video_folder="videos",
        output_base="processed",
        use_hash=False,
        force=False  # Skip already processed videos
    )
    print(f"\n✓ Video processing complete")
except Exception as e:
    print(f"✗ Processing failed: {e}")
    sys.exit(1)

# Step 2: Setup database and ingest
print("\n[Step 2/3] Setting up database and ingesting data...")
print("-" * 60)

try:
    from database.config import test_connection, init_db
    from database.ingest import DataIngester
    
    # Test connection
    if not test_connection():
        print("✗ Database connection failed")
        print("Please:")
        print("  1. Install PostgreSQL")
        print("  2. Create database: createdb video_semantic_search")
        print("  3. Configure .env file")
        sys.exit(1)
    
    # Initialize database
    init_db()
    
    # Ingest videos
    with DataIngester() as ingester:
        stats = ingester.ingest_batch(
            processed_dir="processed",
            generate_embeddings=True,
            skip_existing=True
        )
    
    print(f"\n✓ Database setup complete")
    print(f"  Videos ingested: {stats['success']}")
    
except Exception as e:
    print(f"✗ Database setup failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Start API (optional)
print("\n[Step 3/3] API Server")
print("-" * 60)
print("To start the API server, run:")
print("  python api/app.py")
print("\nOr start manually:")
print("  uvicorn api.app:app --reload --host 0.0.0.0 --port 8000")

print("\n" + "="*60)
print("✅ QUICK START COMPLETE")
print("="*60)

# Test a quick search
print("\n🔍 Testing search...")
try:
    from database.config import SessionLocal
    from search.semantic_search import SemanticSearchEngine
    
    db = SessionLocal()
    search_engine = SemanticSearchEngine(db)
    
    results = search_engine.search("Omega Alpha well", top_k=1)
    
    if results:
        r = results[0]
        print(f"✓ Search working! Found:")
        print(f"  Video: {r.video_filename}")
        print(f"  Time: {r.to_dict()['timestamp']}")
        print(f"  Text: {r.text[:80]}...")
    
    db.close()
except Exception as e:
    print(f"⚠ Search test failed: {e}")

print("\n📚 Next steps:")
print("  1. Start API: python api/app.py")
print("  2. Try queries: curl 'http://localhost:8000/search/quick?q=your+query'")
print("  3. View docs: http://localhost:8000/docs")
