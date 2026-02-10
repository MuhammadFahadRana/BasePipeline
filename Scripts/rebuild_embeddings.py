"""Script to rebuild embedding tables for model upgrades (512 -> 768 dims)."""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from database.config import SessionLocal, init_db

def rebuild_embedding_tables():
    """Drop and recreate tables affected by embedding model changes."""
    print("⚠️  Warning: This will delete all existing embeddings.")
    print("You will need to re-run ingestion to populate them with new models.")
    
    confirm = input("Are you sure? (y/N): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    db = SessionLocal()
    try:
        print("Dropping embedding-related tables...")
        # Drop tables in correct order
        db.execute(text("DROP TABLE IF EXISTS search_image_cache CASCADE"))
        db.execute(text("DROP TABLE IF EXISTS visual_embeddings CASCADE"))
        db.execute(text("DROP TABLE IF EXISTS embeddings CASCADE"))
        db.commit()
        
        print("Recreating tables with new schema...")
        # init_db() will recreate missing tables based on schema.sql
        init_db()
        
        print("✓ Database tables rebuilt successfully.")
        print("Next steps: Run 'python basic_pipeline.py' to re-ingest your videos.")
        
    except Exception as e:
        db.rollback()
        print(f"✗ Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    rebuild_embedding_tables()
