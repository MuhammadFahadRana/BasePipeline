"""
migrate_add_caption_columns.py

Idempotent migration: adds `caption` (TEXT) and `object_labels` (JSONB)
columns to the `scenes` table and creates a GIN full-text index on caption.

Run once:
    python database/migrate_add_caption_columns.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.config import engine, test_connection
from sqlalchemy import text


def run_migration():
    if not test_connection():
        print("✗ Cannot connect to database. Check your .env config.")
        sys.exit(1)

    with engine.begin() as conn:
        # 1. Add caption column (idempotent)
        conn.execute(text("""
            ALTER TABLE scenes
            ADD COLUMN IF NOT EXISTS caption TEXT;
        """))
        print("✓ Column 'caption' ready")

        # 2. Add object_labels column (idempotent)
        conn.execute(text("""
            ALTER TABLE scenes
            ADD COLUMN IF NOT EXISTS object_labels JSONB DEFAULT '[]'::jsonb;
        """))
        print("✓ Column 'object_labels' ready")

        # 3. Add ocr_processed_at if somehow also missing
        conn.execute(text("""
            ALTER TABLE scenes
            ADD COLUMN IF NOT EXISTS ocr_processed_at TIMESTAMP;
        """))
        print("✓ Column 'ocr_processed_at' ready")

        # 4. GIN index for fast full-text search on captions
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_scenes_caption_fts
            ON scenes USING GIN(to_tsvector('english', COALESCE(caption, '')));
        """))
        print("✓ GIN index on caption ready")

        # 5. GIN index on object_labels for containment queries
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_scenes_object_labels
            ON scenes USING GIN(object_labels);
        """))
        print("✓ GIN index on object_labels ready")

    # Report current state
    with engine.connect() as conn:
        cols = conn.execute(text("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'scenes'
            ORDER BY ordinal_position;
        """)).fetchall()

    print("\nCurrent scenes table columns:")
    for col in cols:
        print(f"  {col[0]}: {col[1]}")

    print("\n✓ Migration complete. You can now run: python enrich_scenes_with_captions.py")


if __name__ == "__main__":
    run_migration()
