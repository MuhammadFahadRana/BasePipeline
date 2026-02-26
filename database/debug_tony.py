
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import traceback
from database.mssql_connection import SessionLocal
from database.ingest_sql import SQLDataIngester
from database.models_sqlserver import Video

def debug_ingestion():
    db = SessionLocal()
    ingester = SQLDataIngester(db=db)
    
    results_path = Path("processed/results/Tony Robbins- 토니 라빈스 '행동의 이유' - TED Talk/results.json")
    
    if not results_path.exists():
        print(f"Error: {results_path} does not exist.")
        return

    print(f"--- Debugging ingestion for: {results_path} ---")
    
    # Check if video already exists
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
        video_name = results["video"]["filename"]
    
    existing = db.query(Video).filter(Video.filename == video_name).first()
    if existing:
        print(f"Video {video_name} already in DB (ID: {existing.id}). Deleting to retry...")
        db.delete(existing)
        db.commit()
        print("Deleted existing record.")

    try:
        print("Starting ingest_video...")
        result = ingester.ingest_video(results_path, skip_existing=False)
        print("Ingestion result:", result)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    debug_ingestion()
