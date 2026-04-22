#!/usr/bin/env python3
"""
Import embeddings JSONL into local SQL Server.

Default target:
- 127.0.0.1,14333
- VideoSemanticDB
- dbo.stg_embeddings_import
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Dict, Any, List, Tuple

import pyodbc


CREATE_TABLE_SQL = """
IF OBJECT_ID('dbo.stg_embeddings_import', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.stg_embeddings_import
    (
        import_id BIGINT IDENTITY(1,1) PRIMARY KEY,
        content_hash CHAR(40) NOT NULL,
        source_type NVARCHAR(32) NOT NULL,
        source_name NVARCHAR(512) NULL,
        source_filename NVARCHAR(512) NULL,
        source_file NVARCHAR(1024) NOT NULL,
        record_type NVARCHAR(64) NOT NULL,
        segment_id INT NULL,
        scene_id INT NULL,
        page_number INT NULL,
        chunk_index INT NULL,
        start_time FLOAT NULL,
        end_time FLOAT NULL,
        keyframe_path NVARCHAR(1024) NULL,
        caption NVARCHAR(MAX) NULL,
        ocr_text NVARCHAR(MAX) NULL,
        text_for_embedding NVARCHAR(MAX) NOT NULL,
        embedding_model NVARCHAR(200) NOT NULL,
        embedding_dim INT NOT NULL,
        embedding_json NVARCHAR(MAX) NOT NULL,
        created_at DATETIME2(7) NOT NULL,
        imported_at DATETIME2(7) NOT NULL CONSTRAINT DF_stg_embeddings_import_imported_at DEFAULT SYSUTCDATETIME()
    );

    CREATE UNIQUE INDEX UX_stg_embeddings_import_content_hash
    ON dbo.stg_embeddings_import(content_hash)
    WITH (IGNORE_DUP_KEY = ON);
END
"""

INSERT_SQL = """
INSERT INTO dbo.stg_embeddings_import
(
    content_hash,
    source_type,
    source_name,
    source_filename,
    source_file,
    record_type,
    segment_id,
    scene_id,
    page_number,
    chunk_index,
    start_time,
    end_time,
    keyframe_path,
    caption,
    ocr_text,
    text_for_embedding,
    embedding_model,
    embedding_dim,
    embedding_json,
    created_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    conn = pyodbc.connect(conn_str)
    return conn


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def chunked(items: List[Tuple], size: int) -> Iterator[List[Tuple]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to embeddings.jsonl")
    parser.add_argument("--server", default="127.0.0.1,14333")
    parser.add_argument("--database", default="VideoSemanticDB")
    parser.add_argument("--username", default="slurm_ingest")
    parser.add_argument("--password", required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl).resolve()

    conn = get_connection(args.server, args.database, args.username, args.password)
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()

    rows: List[Tuple] = []
    total = 0

    for row in read_jsonl(jsonl_path):
        rows.append(
            (
                row.get("content_hash"),
                row.get("source_type"),
                row.get("source_name"),
                row.get("source_filename"),
                row.get("source_file"),
                row.get("record_type"),
                row.get("segment_id"),
                row.get("scene_id"),
                row.get("page_number"),
                row.get("chunk_index"),
                row.get("start_time"),
                row.get("end_time"),
                row.get("keyframe_path"),
                row.get("caption"),
                row.get("ocr_text"),
                row.get("text_for_embedding"),
                row.get("embedding_model"),
                row.get("embedding_dim"),
                json.dumps(row.get("embedding", []), ensure_ascii=False),
                row.get("created_at"),
            )
        )

    cur.fast_executemany = True
    for batch in chunked(rows, args.batch_size):
        cur.executemany(INSERT_SQL, batch)
        conn.commit()
        total += len(batch)

    cur.close()
    conn.close()

    print(f"[OK] Imported approximately {total} rows from {jsonl_path}")


if __name__ == "__main__":
    main()
