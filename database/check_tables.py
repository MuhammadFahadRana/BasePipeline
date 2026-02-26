"""Check which tables exist in the SQL Server database."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.mssql_connection import engine
from sqlalchemy import text

with engine.connect() as conn:
    # List all user tables
    tables = conn.execute(text(
        "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE' ORDER BY TABLE_NAME"
    )).fetchall()
    print("Existing tables:")
    for t in tables:
        print(f"  - {t[0]}")
    
    # Expected tables
    expected = ['videos', 'scenes', 'transcript_segments', 'embeddings', 'visual_embeddings', 'query_cache', 'search_queries', 'search_image_cache']
    existing = [t[0] for t in tables]
    missing = [t for t in expected if t not in existing]
    
    if missing:
        print(f"\nMissing tables: {missing}")
    else:
        print("\nAll expected tables exist!")
