#!/usr/bin/env python3
"""
Promote staged embeddings into canonical SQL Server tables.

Source:
  dbo.stg_embeddings_import

Targets:
  - dbo.embeddings (video segment/scene embeddings)
  - dbo.document_embeddings (document chunk embeddings)
  - dbo.embedding_projections (optional)
  - dbo.document_embedding_projections (optional)

Behavior:
  - Idempotent upsert by canonical uniqueness keys
  - Best-effort mapping of staged rows to canonical IDs
  - Optional projection upsert (if projection tables exist)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database.SQL.mssql_connection import engine, test_connection


@dataclass
class VectorColumnInfo:
    is_vector: bool
    dimensions: int | None
    type_name: str


def _to_json_string(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _cast_vector_expr(param_name: str, dim: int) -> str:
    # pyodbc binds long strings as ntext, which cannot cast directly to VECTOR.
    return f"CAST(CAST(:{param_name} AS VARCHAR(MAX)) AS VECTOR({dim}))"


def _insert_embedding_value_sql(col_info: VectorColumnInfo, param_name: str) -> str:
    if col_info.is_vector:
        if col_info.dimensions is None:
            raise RuntimeError("Vector column missing dimensions metadata")
        return _cast_vector_expr(param_name, col_info.dimensions)
    return f":{param_name}"


def _get_vector_column_info_optional(table_name: str, column_name: str) -> VectorColumnInfo | None:
    sql = text(
        """
        SELECT TOP (1)
            type_name = t.name,
            vector_dimensions = c.vector_dimensions
        FROM sys.columns c
        JOIN sys.types t ON c.user_type_id = t.user_type_id
        WHERE c.object_id = OBJECT_ID(:object_name)
          AND c.name = :column_name
        """
    )
    with engine.connect() as conn:
        row = conn.execute(
            sql,
            {"object_name": f"dbo.{table_name}", "column_name": column_name},
        ).mappings().first()
    if not row:
        return None
    return VectorColumnInfo(
        is_vector=(str(row["type_name"]).lower() == "vector"),
        dimensions=int(row["vector_dimensions"]) if row.get("vector_dimensions") is not None else None,
        type_name=str(row["type_name"]).lower(),
    )


def _normalize_text(t: str | None) -> str:
    return " ".join((t or "").strip().split())


def _file_stem(filename: str) -> str:
    p = Path(filename)
    stem = p.stem if p.suffix else p.name
    return stem.strip().lower()


def _project_embedding(values: list[float], dim: int) -> list[float]:
    out = list(values[:dim])
    if len(out) < dim:
        out.extend([0.0] * (dim - len(out)))
    norm = math.sqrt(sum(v * v for v in out))
    if norm > 0:
        out = [v / norm for v in out]
    return out


def build_indexes() -> dict[str, Any]:
    idx: dict[str, Any] = {}
    with engine.connect() as conn:
        videos = conn.execute(text("SELECT id, filename FROM dbo.videos")).mappings().all()
        video_by_exact: dict[str, int] = {}
        video_by_stem: dict[str, int] = {}
        for row in videos:
            vid = int(row["id"])
            name = str(row["filename"])
            video_by_exact[name.lower()] = vid
            video_by_stem[_file_stem(name)] = vid
        idx["video_by_exact"] = video_by_exact
        idx["video_by_stem"] = video_by_stem

        scenes = conn.execute(
            text("SELECT id, video_id, scene_id, start_time, end_time FROM dbo.scenes")
        ).mappings().all()
        scene_by_vid_scene: dict[tuple[int, int], int] = {}
        scenes_by_video: dict[int, list[dict[str, Any]]] = {}
        for row in scenes:
            sid = int(row["id"])
            vid = int(row["video_id"])
            scene_id = int(row["scene_id"])
            scene_by_vid_scene[(vid, scene_id)] = sid
            scenes_by_video.setdefault(vid, []).append(
                {
                    "id": sid,
                    "start": float(row["start_time"]),
                    "end": float(row["end_time"]),
                }
            )
        idx["scene_by_vid_scene"] = scene_by_vid_scene
        idx["scenes_by_video"] = scenes_by_video

        segments = conn.execute(
            text(
                "SELECT id, video_id, start_time, end_time, [text] FROM dbo.transcript_segments"
            )
        ).mappings().all()
        segments_by_video: dict[int, list[dict[str, Any]]] = {}
        for row in segments:
            vid = int(row["video_id"])
            segments_by_video.setdefault(vid, []).append(
                {
                    "id": int(row["id"]),
                    "start": float(row["start_time"]),
                    "end": float(row["end_time"]),
                    "text_norm": _normalize_text(str(row["text"])),
                }
            )
        idx["segments_by_video"] = segments_by_video

        doc_chunks = conn.execute(
            text(
                """
                SELECT dc.id, d.filename, dc.chunk_index, dc.page_number, dc.[text]
                FROM dbo.document_chunks dc
                JOIN dbo.documents d ON d.id = dc.document_id
                """
            )
        ).mappings().all()
        doc_chunks_by_filename: dict[str, list[dict[str, Any]]] = {}
        doc_chunks_by_text: dict[str, list[int]] = {}
        for row in doc_chunks:
            cid = int(row["id"])
            fname = str(row["filename"])
            record = {
                "id": cid,
                "chunk_index": row["chunk_index"],
                "page_number": row["page_number"],
                "text_norm": _normalize_text(str(row["text"])),
            }
            key = _file_stem(fname)
            doc_chunks_by_filename.setdefault(key, []).append(record)
            doc_chunks_by_text.setdefault(record["text_norm"], []).append(cid)
        idx["doc_chunks_by_filename"] = doc_chunks_by_filename
        idx["doc_chunks_by_text"] = doc_chunks_by_text

    return idx


def map_video_id(source_name: str, idx: dict[str, Any]) -> int | None:
    s = (source_name or "").strip().lower()
    if not s:
        return None
    return idx["video_by_exact"].get(s) or idx["video_by_stem"].get(_file_stem(s))


def map_scene_id(stg: dict[str, Any], idx: dict[str, Any], tolerance: float) -> int | None:
    video_id = map_video_id(str(stg.get("source_name") or ""), idx)
    if video_id is None:
        return None
    if stg.get("scene_id") is not None:
        key = (video_id, int(stg["scene_id"]))
        if key in idx["scene_by_vid_scene"]:
            return idx["scene_by_vid_scene"][key]

    start = stg.get("start_time")
    end = stg.get("end_time")
    if start is None or end is None:
        return None
    start = float(start)
    end = float(end)
    best = None
    best_score = 10e9
    for row in idx["scenes_by_video"].get(video_id, []):
        d = abs(row["start"] - start) + abs(row["end"] - end)
        if d < best_score:
            best_score = d
            best = row["id"]
    if best is not None and best_score <= (2 * tolerance):
        return int(best)
    return None


def map_segment_id(stg: dict[str, Any], idx: dict[str, Any], tolerance: float) -> int | None:
    video_id = map_video_id(str(stg.get("source_name") or ""), idx)
    if video_id is None:
        return None
    start = stg.get("start_time")
    end = stg.get("end_time")
    text_norm = _normalize_text(str(stg.get("text_for_embedding") or ""))
    if start is None or end is None or not text_norm:
        return None
    start = float(start)
    end = float(end)

    candidates: list[dict[str, Any]] = []
    for row in idx["segments_by_video"].get(video_id, []):
        if abs(row["start"] - start) <= tolerance and abs(row["end"] - end) <= tolerance:
            candidates.append(row)
    if not candidates:
        return None
    for row in candidates:
        if row["text_norm"] == text_norm:
            return int(row["id"])
    return int(candidates[0]["id"])


def map_document_chunk_id(stg: dict[str, Any], idx: dict[str, Any]) -> int | None:
    source_name = str(stg.get("source_name") or "")
    chunks = idx["doc_chunks_by_filename"].get(_file_stem(source_name), [])
    if chunks:
        ci = stg.get("chunk_index")
        pn = stg.get("page_number")
        if ci is not None:
            for c in chunks:
                if c["chunk_index"] == ci:
                    return int(c["id"])
        if pn is not None:
            for c in chunks:
                if c["page_number"] == pn:
                    return int(c["id"])
        text_norm = _normalize_text(str(stg.get("text_for_embedding") or ""))
        for c in chunks:
            if c["text_norm"] == text_norm:
                return int(c["id"])

    text_norm = _normalize_text(str(stg.get("text_for_embedding") or ""))
    matches = idx["doc_chunks_by_text"].get(text_norm, [])
    if len(matches) == 1:
        return int(matches[0])
    return None


def upsert_video_embedding(
    conn,
    stg: dict[str, Any],
    col_info: VectorColumnInfo,
    segment_id: int | None,
    scene_id: int | None,
) -> tuple[int, bool]:
    model = str(stg["embedding_model"])
    emb_json = str(stg["embedding_json"])
    emb_expr = _insert_embedding_value_sql(col_info, "embedding_json")

    existing = conn.execute(
        text(
            """
            SELECT id FROM dbo.embeddings
            WHERE
              ((segment_id = :segment_id) OR (segment_id IS NULL AND :segment_id IS NULL))
              AND ((scene_id = :scene_id) OR (scene_id IS NULL AND :scene_id IS NULL))
              AND embedding_model = :embedding_model
            """
        ),
        {"segment_id": segment_id, "scene_id": scene_id, "embedding_model": model},
    ).scalar()

    if existing:
        conn.execute(
            text(
                f"""
                UPDATE dbo.embeddings
                SET embedding = {emb_expr}
                WHERE id = :id
                """
            ),
            {"id": int(existing), "embedding_json": emb_json},
        )
        return int(existing), False

    new_id = conn.execute(
        text(
            f"""
            INSERT INTO dbo.embeddings (segment_id, scene_id, embedding, embedding_model)
            OUTPUT INSERTED.id
            VALUES (:segment_id, :scene_id, {emb_expr}, :embedding_model)
            """
        ),
        {
            "segment_id": segment_id,
            "scene_id": scene_id,
            "embedding_json": emb_json,
            "embedding_model": model,
        },
    ).scalar_one()
    return int(new_id), True


def upsert_document_embedding(
    conn, stg: dict[str, Any], col_info: VectorColumnInfo, chunk_id: int
) -> tuple[int, bool]:
    model = str(stg["embedding_model"])
    emb_json = str(stg["embedding_json"])
    emb_expr = _insert_embedding_value_sql(col_info, "embedding_json")

    existing = conn.execute(
        text(
            """
            SELECT id FROM dbo.document_embeddings
            WHERE chunk_id = :chunk_id AND embedding_model = :embedding_model
            """
        ),
        {"chunk_id": chunk_id, "embedding_model": model},
    ).scalar()

    if existing:
        conn.execute(
            text(
                f"""
                UPDATE dbo.document_embeddings
                SET embedding = {emb_expr}
                WHERE id = :id
                """
            ),
            {"id": int(existing), "embedding_json": emb_json},
        )
        return int(existing), False

    new_id = conn.execute(
        text(
            f"""
            INSERT INTO dbo.document_embeddings (chunk_id, embedding, embedding_model)
            OUTPUT INSERTED.id
            VALUES (:chunk_id, {emb_expr}, :embedding_model)
            """
        ),
        {"chunk_id": chunk_id, "embedding_json": emb_json, "embedding_model": model},
    ).scalar_one()
    return int(new_id), True


def upsert_projection(
    conn,
    table: str,
    fk_name: str,
    fk_id: int,
    extra_fk_name: str,
    extra_fk_id: int | None,
    model: str,
    projection: list[float],
    col_info: VectorColumnInfo,
    projection_dim: int,
    projection_method: str,
) -> bool:
    proj_json = _to_json_string(projection)
    proj_expr = _insert_embedding_value_sql(col_info, "projection_json")

    existing = conn.execute(
        text(
            f"""
            SELECT id FROM dbo.{table}
            WHERE {fk_name} = :fk_id
              AND projection_dim = :projection_dim
              AND projection_method = :projection_method
            """
        ),
        {
            "fk_id": fk_id,
            "projection_dim": projection_dim,
            "projection_method": projection_method,
        },
    ).scalar()

    if existing:
        conn.execute(
            text(
                f"""
                UPDATE dbo.{table}
                SET projection = {proj_expr},
                    embedding_model = :embedding_model
                WHERE id = :id
                """
            ),
            {"id": int(existing), "projection_json": proj_json, "embedding_model": model},
        )
        return False

    conn.execute(
        text(
            f"""
            INSERT INTO dbo.{table}
                ({fk_name}, {extra_fk_name}, projection, projection_dim, embedding_model, projection_method)
            VALUES
                (:fk_id, :extra_fk_id, {proj_expr}, :projection_dim, :embedding_model, :projection_method)
            """
        ),
        {
            "fk_id": fk_id,
            "extra_fk_id": extra_fk_id,
            "projection_json": proj_json,
            "projection_dim": projection_dim,
            "embedding_model": model,
            "projection_method": projection_method,
        },
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote staged embeddings into canonical tables.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of staged rows.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows to process per transaction.")
    parser.add_argument("--tolerance", type=float, default=0.35, help="Time tolerance for segment/scene matching.")
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=int(os.getenv("MSSQL_TEXT_PROJECTION_DIM", "1024")),
        help="Projection dimension written to projection tables.",
    )
    parser.add_argument(
        "--projection-method",
        default="head_l2_norm",
        help="Projection method label stored in projection tables.",
    )
    parser.add_argument("--no-projections", action="store_true", help="Disable projection upsert.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not test_connection():
        return 1

    emb_col = _get_vector_column_info_optional("embeddings", "embedding")
    doc_emb_col = _get_vector_column_info_optional("document_embeddings", "embedding")
    if emb_col is None or doc_emb_col is None:
        raise RuntimeError("Canonical embedding tables are missing required embedding columns.")

    video_proj_col = _get_vector_column_info_optional("embedding_projections", "projection")
    doc_proj_col = _get_vector_column_info_optional("document_embedding_projections", "projection")
    projections_enabled = (not args.no_projections) and (video_proj_col is not None) and (doc_proj_col is not None)
    if not args.no_projections and not projections_enabled:
        print(
            "[WARN] Projection tables not found (or incomplete). "
            "Continuing without projection upserts."
        )

    indexes = build_indexes()

    query = "SELECT * FROM dbo.stg_embeddings_import ORDER BY import_id"
    if args.limit is not None:
        query = f"SELECT TOP ({int(args.limit)}) * FROM dbo.stg_embeddings_import ORDER BY import_id"

    stats = {
        "total_read": 0,
        "video_rows": 0,
        "doc_rows": 0,
        "video_inserted": 0,
        "video_updated": 0,
        "doc_inserted": 0,
        "doc_updated": 0,
        "video_proj_inserted": 0,
        "video_proj_updated": 0,
        "doc_proj_inserted": 0,
        "doc_proj_updated": 0,
        "unresolved_video": 0,
        "unresolved_doc": 0,
        "invalid_embedding": 0,
        "skipped_kind": 0,
        "dryrun_resolvable_video": 0,
        "dryrun_resolvable_doc": 0,
    }

    with engine.connect() as read_conn:
        result = read_conn.execution_options(stream_results=True).execute(text(query))
        while True:
            batch = result.mappings().fetchmany(args.batch_size)
            if not batch:
                break
            tx_ctx = nullcontext(None) if args.dry_run else engine.begin()
            with tx_ctx as write_conn:
                for stg in batch:
                    stats["total_read"] += 1
                    source_type = str(stg.get("source_type") or "").strip().lower()
                    record_type = str(stg.get("record_type") or "").strip().lower()
                    model = str(stg.get("embedding_model") or "").strip()
                    emb_json = stg.get("embedding_json")
                    if not model or not emb_json:
                        stats["invalid_embedding"] += 1
                        continue
                    try:
                        emb_values = json.loads(str(emb_json))
                        if not isinstance(emb_values, list) or not emb_values:
                            raise ValueError("embedding not list")
                        emb_values = [float(v) for v in emb_values]
                    except Exception:
                        stats["invalid_embedding"] += 1
                        continue

                    if source_type == "video":
                        if record_type in {"segments", "transcript_segments"}:
                            segment_id = map_segment_id(stg, indexes, args.tolerance)
                            if segment_id is None:
                                stats["unresolved_video"] += 1
                                continue
                            stats["video_rows"] += 1
                            if args.dry_run:
                                stats["dryrun_resolvable_video"] += 1
                                continue
                            emb_id, inserted = upsert_video_embedding(
                                write_conn, stg, emb_col, segment_id=segment_id, scene_id=None
                            )
                            if inserted:
                                stats["video_inserted"] += 1
                            else:
                                stats["video_updated"] += 1
                            if projections_enabled:
                                inserted_proj = upsert_projection(
                                    write_conn,
                                    table="embedding_projections",
                                    fk_name="embedding_id",
                                    fk_id=emb_id,
                                    extra_fk_name="segment_id",
                                    extra_fk_id=segment_id,
                                    model=model,
                                    projection=_project_embedding(emb_values, args.projection_dim),
                                    col_info=video_proj_col,
                                    projection_dim=args.projection_dim,
                                    projection_method=args.projection_method,
                                )
                                if inserted_proj:
                                    stats["video_proj_inserted"] += 1
                                else:
                                    stats["video_proj_updated"] += 1
                        elif record_type == "scenes":
                            scene_id = map_scene_id(stg, indexes, args.tolerance)
                            if scene_id is None:
                                stats["unresolved_video"] += 1
                                continue
                            stats["video_rows"] += 1
                            if args.dry_run:
                                stats["dryrun_resolvable_video"] += 1
                                continue
                            emb_id, inserted = upsert_video_embedding(
                                write_conn, stg, emb_col, segment_id=None, scene_id=scene_id
                            )
                            if inserted:
                                stats["video_inserted"] += 1
                            else:
                                stats["video_updated"] += 1
                            if projections_enabled:
                                inserted_proj = upsert_projection(
                                    write_conn,
                                    table="embedding_projections",
                                    fk_name="embedding_id",
                                    fk_id=emb_id,
                                    extra_fk_name="scene_id",
                                    extra_fk_id=scene_id,
                                    model=model,
                                    projection=_project_embedding(emb_values, args.projection_dim),
                                    col_info=video_proj_col,
                                    projection_dim=args.projection_dim,
                                    projection_method=args.projection_method,
                                )
                                if inserted_proj:
                                    stats["video_proj_inserted"] += 1
                                else:
                                    stats["video_proj_updated"] += 1
                        else:
                            stats["skipped_kind"] += 1
                    elif source_type == "document":
                        chunk_id = map_document_chunk_id(stg, indexes)
                        if chunk_id is None:
                            stats["unresolved_doc"] += 1
                            continue
                        stats["doc_rows"] += 1
                        if args.dry_run:
                            stats["dryrun_resolvable_doc"] += 1
                            continue
                        doc_emb_id, inserted = upsert_document_embedding(
                            write_conn, stg, doc_emb_col, chunk_id=chunk_id
                        )
                        if inserted:
                            stats["doc_inserted"] += 1
                        else:
                            stats["doc_updated"] += 1
                        if projections_enabled:
                            inserted_proj = upsert_projection(
                                write_conn,
                                table="document_embedding_projections",
                                fk_name="document_embedding_id",
                                fk_id=doc_emb_id,
                                extra_fk_name="chunk_id",
                                extra_fk_id=chunk_id,
                                model=model,
                                projection=_project_embedding(emb_values, args.projection_dim),
                                col_info=doc_proj_col,
                                projection_dim=args.projection_dim,
                                projection_method=args.projection_method,
                            )
                            if inserted_proj:
                                stats["doc_proj_inserted"] += 1
                            else:
                                stats["doc_proj_updated"] += 1
                    else:
                        stats["skipped_kind"] += 1

    print("=" * 72)
    print("STAGING PROMOTION SUMMARY")
    print("=" * 72)
    for key, value in stats.items():
        print(f"{key}: {value}")
    if args.dry_run:
        print("[INFO] Dry run mode: no writes were performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
