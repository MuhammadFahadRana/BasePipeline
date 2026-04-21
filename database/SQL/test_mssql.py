"""Verify SQL Server schema objects for VideoSemanticDB."""

import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from database.SQL.mssql_connection import engine


def main() -> None:
    with engine.connect() as conn:
        db_name = conn.execute(text("SELECT DB_NAME()")).scalar_one()
        has_vector = conn.execute(
            text("SELECT CASE WHEN EXISTS (SELECT 1 FROM sys.types WHERE name = 'vector') THEN 1 ELSE 0 END")
        ).scalar_one()

        tables = conn.execute(
            text(
                """
                SELECT name
                FROM sys.tables
                WHERE schema_id = SCHEMA_ID('dbo')
                ORDER BY name
                """
            )
        ).fetchall()

        print(f"Connected to: {db_name}")
        print(f"Vector type available: {'yes' if has_vector else 'no (JSON fallback)'}")
        print("Tables:")
        for row in tables:
            print(f"  - {row[0]}")


if __name__ == "__main__":
    main()
