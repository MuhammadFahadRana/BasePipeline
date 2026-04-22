#!/usr/bin/env python3
"""
Import exported embeddings JSONL into local SQL Server staging table.

Default target:
- SQL Server on 127.0.0.1,14333
- Database: VideoSemanticDB
- Login: slurm_ingest
- Table: dbo.stg_embeddings_import
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import pyodbc


CREATE_TABLE_SQL = """
IF OBJECT_ID('dbo.stg_embeddings_import', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.stg_embeddings_import
    (
        import_id BIGINT IDENTITY(1,1) PRIMARY KEY,
        source_hash CHAR(40) NOT NULL,
        source_type NVARCHAR(32) NOT NULL,
        source_name NVARCHAR(512) NULL,
        source_file NVARCHAR(1024) NOT NULL,
        item_kind NVARCHAR(64) NULL,
        item_path NVARCHAR(512) NULL,
        item_index INT NULL,
        logical_id NVARCHAR(256) NULL,
        scene_id NVARCHAR(128) NULL,
        segment_id NVARCHAR(128) NULL,
        start_time FLOAT NULL,
        end_time FLOAT NULL,
        text_for_embedding NVARCHAR(MAX) NOT NULL,
        model_name NVARCHAR(200) NOT NULL,
        embedding_dim INT NOT NULL,
        embedding_json NVARCHAR(MAX) NOT NULL,
        created_at DATETIME2(7) NOT NULL,
        imported_at DATETIME2(7) NOT NULL CONSTRAINT DF_stg_embeddings_import_imported_at DEFAULT SYSUTCDATETIME()
    );

    CREATE UNIQUE INDEX UX_stg_embeddings_import_source_hash
        ON dbo.stg_embeddings_import(source_hash);
END
"""

INSERT_SQL = """
IF NOT EXISTS (
    SELECT 1
    FROM dbo.stg_embeddings_import
    WHERE source_hash = ?
)
BEGIN
    INSERT INTO dbo.stg_embeddings_import
    (
        source_hash,
        source_type,
        source_name,
        source_file,
        item_kind,
        item_path,
        item_index,
        logical_id,
        scene_id,
        segment_id,
        start_time,
        end_time,
        text_for_embedding,
        model_name,
        embedding_dim,
        embedding_json,
        created_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
END
"""


def get_connection(server: str, database: str, username: str, password: str) -> pyodbc.Connection:
    conn_str = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        "TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to embeddings.jsonl")
    parser.add_argument("--server", default="127.0.0.1,14333")
    parser.add_argument("--database", default="VideoSemanticDB")
    parser.add_argument("--username", default="slurm_ingest")
    parser.add_argument("--password", required=True)
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl).resolve()

    conn = get_connection(args.server, args.database, args.username, args.password)
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()

    inserted = 0
    skipped = 0

    for line_no, row in load_jsonl(jsonl_path):
        try:
            embedding_json = json.dumps(row["embedding"], ensure_ascii=False)

            params = (
                row["source_hash"],             # IF NOT EXISTS check
                row["source_hash"],
                row.get("source_type"),
                row.get("source_name"),
                row.get("source_file"),
                row.get("item_kind"),
                row.get("item_path"),
                row.get("item_index"),
                row.get("logical_id"),
                row.get("scene_id"),
                row.get("segment_id"),
                row.get("start_time"),
                row.get("end_time"),
                row.get("text_for_embedding"),
                row.get("model_name"),
                row.get("embedding_dim"),
                embedding_json,
                row.get("created_at"),
            )

            before = conn.total_changes if hasattr(conn, "total_changes") else None
            cur.execute(INSERT_SQL, params)
            conn.commit()

            # pyodbc does not expose inserted/skipped cleanly here, so count optimistically
            inserted += 1

        except pyodbc.IntegrityError:
            skipped += 1
            conn.rollback()
        except Exception as e:
            conn.rollback()
            print(f"[ERROR] line {line_no}: {e}")

    cur.close()
    conn.close()

    print(f"[OK] import finished")
    print(f"    file     : {jsonl_path}")
    print(f"    inserted : ~{inserted}")
    print(f"    skipped  : ~{skipped}")


if __name__ == "__main__":
    main()