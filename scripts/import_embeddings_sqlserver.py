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
from datetime import datetime, timezone

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


def _parse_created_at(value: Any) -> datetime:
    """Parse ISO timestamp to timezone-naive UTC datetime for DATETIME2."""
    if isinstance(value, datetime):
        dt = value
    else:
        raw = str(value or "").strip()
        if not raw:
            return datetime.utcnow()
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return datetime.utcnow()

    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _to_row(payload: Dict[str, Any]) -> Tuple:
    """Map exporter JSONL payload to staging table row."""
    source_file = payload.get("source_file")
    source_file_str = str(source_file or "")
    source_filename = Path(source_file_str).name if source_file_str else None

    return (
        payload.get("content_hash") or payload.get("source_hash"),
        payload.get("source_type"),
        payload.get("source_name"),
        payload.get("source_filename") or source_filename,
        source_file_str,
        payload.get("record_type") or payload.get("item_kind"),
        payload.get("segment_id"),
        payload.get("scene_id"),
        payload.get("page_number"),
        payload.get("chunk_index"),
        payload.get("start_time"),
        payload.get("end_time"),
        payload.get("keyframe_path"),
        payload.get("caption"),
        payload.get("ocr_text"),
        payload.get("text_for_embedding"),
        payload.get("embedding_model") or payload.get("model_name"),
        payload.get("embedding_dim"),
        json.dumps(payload.get("embedding", []), ensure_ascii=False),
        _parse_created_at(payload.get("created_at")),
    )


def _validate_row(row: Tuple) -> bool:
    # Required columns by table schema:
    # content_hash, source_type, source_file, record_type, text_for_embedding, embedding_model, embedding_dim
    required_indexes = [0, 1, 4, 5, 15, 16, 17]
    for idx in required_indexes:
        value = row[idx]
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
    return True


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
    attempted = 0
    valid = 0
    skipped_invalid = 0

    cur.fast_executemany = True
    for row in read_jsonl(jsonl_path):
        attempted += 1
        mapped = _to_row(row)
        if not _validate_row(mapped):
            skipped_invalid += 1
            continue

        rows.append(mapped)
        valid += 1

        if len(rows) >= args.batch_size:
            cur.executemany(INSERT_SQL, rows)
            conn.commit()
            rows.clear()

    if rows:
        cur.executemany(INSERT_SQL, rows)
        conn.commit()
        rows.clear()

    cur.close()
    conn.close()

    print(
        f"[OK] Import finished from {jsonl_path} | "
        f"attempted={attempted} valid={valid} skipped_invalid={skipped_invalid}"
    )


if __name__ == "__main__":
    main()
