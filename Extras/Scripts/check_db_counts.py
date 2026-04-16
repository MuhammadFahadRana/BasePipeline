"""Check database counts."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from database.config import SessionLocal

def check_counts():
    db = SessionLocal()
    try:
        tables = ['videos', 'scenes', 'transcript_segments', 'embeddings', 'visual_embeddings']
        print(f"{'Table':<25} | {'Count':<10}")
        print("-" * 40)
        for table in tables:
            try:
                result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                print(f"{table:<25} | {count:<10}")
            except Exception as e:
                print(f"{table:<25} | Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    check_counts()
