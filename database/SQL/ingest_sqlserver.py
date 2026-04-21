"""Ingest processed video/document data directly into SQL Server.

This script is separate from PostgreSQL-oriented ingesters and targets:
  - dbo.videos / dbo.scenes / dbo.transcript_segments / dbo.embeddings / dbo.visual_embeddings
  - dbo.documents / dbo.document_chunks / dbo.document_embeddings

Usage examples:
  python database/SQL/ingest_sqlserver.py
  python database/SQL/ingest_sqlserver.py --skip-videos --limit-documents 10
  python database/SQL/ingest_sqlserver.py --no-text-embeddings --no-visual-embeddings
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import text

# Ensure project-root imports resolve when executed as a file.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from database.SQL.mssql_connection import engine, test_connection
from embeddings.text_embeddings import get_embedding_generator
from embeddings.vision_embeddings import get_vision_embedding_generator

DEFAULT_TEXT_MODEL = os.getenv(
    "TEXT_EMBEDDING_MODEL",
    os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
)
DEFAULT_VISION_MODEL = os.getenv(
    "VISION_EMBEDDING_MODEL", "google/siglip-base-patch16-224"
)
DEFAULT_TEXT_DEVICE = os.getenv("TEXT_EMBEDDING_DEVICE", "auto")
DEFAULT_VISION_DEVICE = os.getenv("VISION_EMBEDDING_DEVICE", "auto")


@dataclass
class VectorColumnInfo:
    is_vector: bool
    dimensions: int | None
    type_name: str


class SqlServerIngester:
    def __init__(
        self,
        text_batch_size: int = 32,
        visual_batch_size: int = 16,
        enable_text_embeddings: bool = True,
        enable_visual_embeddings: bool = True,
        text_model_name: str = DEFAULT_TEXT_MODEL,
        vision_model_name: str = DEFAULT_VISION_MODEL,
        text_device: str = DEFAULT_TEXT_DEVICE,
        vision_device: str = DEFAULT_VISION_DEVICE,
    ) -> None:
        self.text_batch_size = text_batch_size
        self.visual_batch_size = visual_batch_size
        self.enable_text_embeddings = enable_text_embeddings
        self.enable_visual_embeddings = enable_visual_embeddings
        self.text_model_name = text_model_name
        self.vision_model_name = vision_model_name
        self.text_device = text_device
        self.vision_device = vision_device

        self._text_gen = None
        self._vision_gen = None

        self.embedding_col_info = self._get_vector_column_info("embeddings", "embedding")
        self.doc_embedding_col_info = self._get_vector_column_info(
            "document_embeddings", "embedding"
        )
        self.visual_embedding_col_info = self._get_vector_column_info(
            "visual_embeddings", "embedding"
        )

        if self.enable_text_embeddings:
            text_gen = self.text_gen
            self._assert_dimension_compat(
                model_dim=text_gen.embedding_dim,
                col_info=self.embedding_col_info,
                label="dbo.embeddings.embedding",
            )
            self._assert_dimension_compat(
                model_dim=text_gen.embedding_dim,
                col_info=self.doc_embedding_col_info,
                label="dbo.document_embeddings.embedding",
            )

        if self.enable_visual_embeddings:
            vision_gen = self.vision_gen
            self._assert_dimension_compat(
                model_dim=vision_gen.embedding_dim,
                col_info=self.visual_embedding_col_info,
                label="dbo.visual_embeddings.embedding",
            )

    @property
    def text_gen(self):
        if self._text_gen is None:
            self._text_gen = get_embedding_generator(
                model_name=self.text_model_name,
                device=self.text_device,
            )
        return self._text_gen

    @property
    def vision_gen(self):
        if self._vision_gen is None:
            self._vision_gen = get_vision_embedding_generator(
                model_name=self.vision_model_name,
                device=self.vision_device,
            )
        return self._vision_gen

    @staticmethod
    def _assert_dimension_compat(
        model_dim: int, col_info: VectorColumnInfo, label: str
    ) -> None:
        if col_info.is_vector and col_info.dimensions is not None and model_dim != col_info.dimensions:
            raise ValueError(
                f"Dimension mismatch for {label}: model={model_dim}, column={col_info.dimensions}. "
                "Update SQL schema VECTOR dimensions (or choose matching model) and recreate DB."
            )

    @staticmethod
    def _normalize_optional_text(value: Any) -> str | None:
        if value is None:
            return None
        txt = str(value).strip()
        if not txt:
            return None
        lowered = txt.lower()
        if lowered in {"none", "null", "n/a", "na", "no text", "no visible text"}:
            return None
        return txt

    @staticmethod
    def _normalize_object_labels(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            out = [str(v).strip() for v in value if str(v).strip()]
        elif isinstance(value, str):
            out = [v.strip() for v in value.split(",") if v.strip()]
        else:
            out = [str(value).strip()] if str(value).strip() else []
        invalid = {"none", "null", "n/a", "na", "no objects"}
        out = [v for v in out if v.lower() not in invalid]
        return list(dict.fromkeys(out))

    @classmethod
    def _compose_scene_text(
        cls, caption: Any, object_labels: Any, ocr_text: Any
    ) -> str | None:
        parts: list[str] = []
        c = cls._normalize_optional_text(caption)
        if c:
            parts.append(c)
        labels = cls._normalize_object_labels(object_labels)
        if labels:
            parts.append(" ".join(labels))
        o = cls._normalize_optional_text(ocr_text)
        if o:
            parts.append(o)
        joined = " ".join(parts).strip()
        return joined or None

    @staticmethod
    def _to_json_string(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def _get_vector_column_info(self, table_name: str, column_name: str) -> VectorColumnInfo:
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
            raise RuntimeError(f"Missing column dbo.{table_name}.{column_name}")
        type_name = str(row["type_name"]).lower()
        dimensions = row.get("vector_dimensions")
        return VectorColumnInfo(
            is_vector=(type_name == "vector"),
            dimensions=int(dimensions) if dimensions is not None else None,
            type_name=type_name,
        )

    @staticmethod
    def _resolve_path(path_value: str | None) -> Path | None:
        if not path_value:
            return None
        p = Path(path_value)
        if p.exists():
            return p
        if not p.is_absolute():
            candidate = Path.cwd() / p
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _iter_video_result_files(base_dir: Path) -> list[Path]:
        if not base_dir.exists():
            return []
        return sorted(p for p in base_dir.glob("*/results.json") if p.is_file())

    @staticmethod
    def _iter_document_result_files(base_dir: Path) -> list[Path]:
        if not base_dir.exists():
            return []
        direct = sorted(p for p in base_dir.glob("*/results/results.json") if p.is_file())
        deep = sorted(p for p in base_dir.glob("**/results/results.json") if p.is_file())
        # Keep deterministic + dedup.
        seen: set[Path] = set()
        ordered: list[Path] = []
        for p in [*direct, *deep]:
            if p not in seen:
                ordered.append(p)
                seen.add(p)
        return ordered

    @staticmethod
    def _cast_vector_expr(param_name: str, dim: int) -> str:
        # pyodbc binds long strings as ntext, which cannot cast directly to VECTOR.
        # Cast to VARCHAR(MAX) first, then VECTOR(dim).
        return f"CAST(CAST(:{param_name} AS VARCHAR(MAX)) AS VECTOR({dim}))"

    def _insert_embedding_value_sql(
        self, col_info: VectorColumnInfo, param_name: str
    ) -> str:
        if col_info.is_vector:
            if col_info.dimensions is None:
                raise RuntimeError("Vector column missing dimensions metadata")
            return self._cast_vector_expr(param_name, col_info.dimensions)
        return f":{param_name}"

    def _insert_text_embedding(
        self,
        conn,
        segment_id: int | None,
        scene_id: int | None,
        embedding: list[float],
        model_name: str,
    ) -> None:
        emb_json = self._to_json_string(embedding)
        emb_expr = self._insert_embedding_value_sql(self.embedding_col_info, "embedding_json")
        sql = text(
            f"""
            INSERT INTO dbo.embeddings (segment_id, scene_id, embedding, embedding_model)
            VALUES (:segment_id, :scene_id, {emb_expr}, :embedding_model)
            """
        )
        conn.execute(
            sql,
            {
                "segment_id": segment_id,
                "scene_id": scene_id,
                "embedding_json": emb_json,
                "embedding_model": model_name,
            },
        )

    def _insert_document_embedding(
        self, conn, chunk_id: int, embedding: list[float], model_name: str
    ) -> None:
        emb_json = self._to_json_string(embedding)
        emb_expr = self._insert_embedding_value_sql(self.doc_embedding_col_info, "embedding_json")
        sql = text(
            f"""
            INSERT INTO dbo.document_embeddings (chunk_id, embedding, embedding_model)
            VALUES (:chunk_id, {emb_expr}, :embedding_model)
            """
        )
        conn.execute(
            sql,
            {
                "chunk_id": chunk_id,
                "embedding_json": emb_json,
                "embedding_model": model_name,
            },
        )

    def _insert_visual_embedding(
        self,
        conn,
        scene_id: int,
        keyframe_path: str,
        sample_time: float | None,
        frame_role: str,
        frame_index: int | None,
        embedding: list[float],
        model_name: str,
    ) -> None:
        emb_json = self._to_json_string(embedding)
        emb_expr = self._insert_embedding_value_sql(
            self.visual_embedding_col_info, "embedding_json"
        )
        sql = text(
            f"""
            INSERT INTO dbo.visual_embeddings
                (scene_id, keyframe_path, sample_time, frame_role, frame_index, embedding, embedding_model)
            VALUES
                (:scene_id, :keyframe_path, :sample_time, :frame_role, :frame_index, {emb_expr}, :embedding_model)
            """
        )
        conn.execute(
            sql,
            {
                "scene_id": scene_id,
                "keyframe_path": keyframe_path,
                "sample_time": sample_time,
                "frame_role": frame_role,
                "frame_index": frame_index,
                "embedding_json": emb_json,
                "embedding_model": model_name,
            },
        )

    @staticmethod
    def _find_scene_id_for_segment(
        scenes: list[dict[str, Any]], segment_start: float
    ) -> int | None:
        for s in scenes:
            if s["start_time"] <= segment_start <= s["end_time"]:
                return s["db_scene_id"]
        return None

    @staticmethod
    def _delete_existing_video(conn, video_id: int) -> None:
        # Delete scene-level embeddings first (segment-level rows cascade via transcript FK).
        conn.execute(
            text(
                """
                DELETE e
                FROM dbo.embeddings e
                JOIN dbo.scenes s ON s.id = e.scene_id
                WHERE s.video_id = :video_id
                """
            ),
            {"video_id": video_id},
        )
        conn.execute(
            text(
                """
                DELETE ve
                FROM dbo.visual_embeddings ve
                JOIN dbo.scenes s ON s.id = ve.scene_id
                WHERE s.video_id = :video_id
                """
            ),
            {"video_id": video_id},
        )
        conn.execute(text("DELETE FROM dbo.videos WHERE id = :video_id"), {"video_id": video_id})

    def ingest_video_result_file(self, result_file: Path) -> dict[str, Any]:
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        video_info = payload.get("video", {})
        transcription = payload.get("transcription", {})
        scene_analysis = payload.get("scene_analysis", {})

        filename = video_info.get("filename")
        if not filename:
            raise ValueError(f"Missing video.filename in {result_file}")

        scenes_raw = scene_analysis.get("scenes", []) or []
        segments_raw = transcription.get("segments", []) or []
        language = transcription.get("language", "en")

        with engine.begin() as conn:
            existing_video_id = conn.execute(
                text("SELECT id FROM dbo.videos WHERE filename = :filename"),
                {"filename": filename},
            ).scalar()
            if existing_video_id:
                self._delete_existing_video(conn, int(existing_video_id))

            video_id = conn.execute(
                text(
                    """
                    INSERT INTO dbo.videos
                        (filename, file_path, file_size_mb, duration_seconds, whisper_model, scene_threshold, video_fingerprint)
                    OUTPUT INSERTED.id
                    VALUES
                        (:filename, :file_path, :file_size_mb, :duration_seconds, :whisper_model, :scene_threshold, :video_fingerprint)
                    """
                ),
                {
                    "filename": filename,
                    "file_path": video_info.get("path", ""),
                    "file_size_mb": video_info.get("size_mb"),
                    "duration_seconds": scene_analysis.get("total_duration"),
                    "whisper_model": payload.get("processing_info", {}).get("transcription_backend"),
                    "scene_threshold": payload.get("processing_info", {}).get("scene_threshold"),
                    "video_fingerprint": self._to_json_string(
                        {
                            "mtime": video_info.get("mtime"),
                            "mtime_iso": video_info.get("mtime_iso"),
                            "size_mb": video_info.get("size_mb"),
                        }
                    ),
                },
            ).scalar_one()

            scenes_inserted: list[dict[str, Any]] = []
            for scene in scenes_raw:
                scene_id = conn.execute(
                    text(
                        """
                        INSERT INTO dbo.scenes
                            (video_id, scene_id, start_time, end_time, duration, start_frame, end_frame,
                             keyframe_path, ocr_text, ocr_text_norm, ocr_confidence, ocr_processed_at,
                             object_labels, caption)
                        OUTPUT INSERTED.id
                        VALUES
                            (:video_id, :scene_id, :start_time, :end_time, :duration, :start_frame, :end_frame,
                             :keyframe_path, :ocr_text, :ocr_text_norm, :ocr_confidence,
                             CASE WHEN :ocr_text IS NULL THEN NULL ELSE SYSUTCDATETIME() END,
                             :object_labels, :caption)
                        """
                    ),
                    {
                        "video_id": int(video_id),
                        "scene_id": int(scene.get("scene_id")),
                        "start_time": float(scene.get("start_time", 0.0)),
                        "end_time": float(scene.get("end_time", 0.0)),
                        "duration": float(scene.get("duration", 0.0)),
                        "start_frame": scene.get("start_frame"),
                        "end_frame": scene.get("end_frame"),
                        "keyframe_path": scene.get("keyframe_path"),
                        "ocr_text": self._normalize_optional_text(scene.get("ocr_text")),
                        "ocr_text_norm": self._normalize_optional_text(scene.get("ocr_text")),
                        "ocr_confidence": scene.get("ocr_confidence"),
                        "object_labels": self._to_json_string(
                            self._normalize_object_labels(scene.get("object_labels"))
                        ),
                        "caption": self._normalize_optional_text(scene.get("caption")),
                    },
                ).scalar_one()

                scenes_inserted.append(
                    {
                        "db_scene_id": int(scene_id),
                        "scene_id": int(scene.get("scene_id", 0)),
                        "start_time": float(scene.get("start_time", 0.0)),
                        "end_time": float(scene.get("end_time", 0.0)),
                        "duration": float(scene.get("duration", 0.0)),
                        "keyframe_path": scene.get("keyframe_path"),
                        "caption": scene.get("caption"),
                        "object_labels": scene.get("object_labels"),
                        "ocr_text": scene.get("ocr_text"),
                    }
                )

            segment_rows: list[tuple[int, str]] = []
            for idx, seg in enumerate(segments_raw):
                seg_start = float(seg.get("start", 0.0))
                seg_end = float(seg.get("end", seg_start))
                seg_text = str(seg.get("text", "")).strip()
                db_scene_id = self._find_scene_id_for_segment(scenes_inserted, seg_start)

                segment_id = conn.execute(
                    text(
                        """
                        INSERT INTO dbo.transcript_segments
                            (video_id, scene_id, segment_index, start_time, end_time, [text], [language])
                        OUTPUT INSERTED.id
                        VALUES
                            (:video_id, :scene_id, :segment_index, :start_time, :end_time, :text, :language)
                        """
                    ),
                    {
                        "video_id": int(video_id),
                        "scene_id": db_scene_id,
                        "segment_index": idx,
                        "start_time": seg_start,
                        "end_time": seg_end,
                        "text": seg_text,
                        "language": language,
                    },
                ).scalar_one()
                segment_rows.append((int(segment_id), seg_text))

            text_embedding_count = 0
            if self.enable_text_embeddings and segment_rows:
                texts = [row[1] for row in segment_rows]
                vectors = self.text_gen.encode(
                    texts, batch_size=self.text_batch_size, show_progress=True
                )
                for (segment_id, _txt), vec in zip(segment_rows, vectors):
                    self._insert_text_embedding(
                        conn=conn,
                        segment_id=segment_id,
                        scene_id=None,
                        embedding=vec.tolist(),
                        model_name=self.text_gen.model_name,
                    )
                    text_embedding_count += 1

                scene_texts: list[tuple[int, str]] = []
                for s in scenes_inserted:
                    scene_text = self._compose_scene_text(
                        s.get("caption"), s.get("object_labels"), s.get("ocr_text")
                    )
                    if scene_text:
                        scene_texts.append((int(s["db_scene_id"]), scene_text))

                if scene_texts:
                    scene_vectors = self.text_gen.encode(
                        [t for _, t in scene_texts],
                        batch_size=min(self.text_batch_size, 16),
                        show_progress=False,
                    )
                    for (db_scene_id, _txt), vec in zip(scene_texts, scene_vectors):
                        self._insert_text_embedding(
                            conn=conn,
                            segment_id=None,
                            scene_id=db_scene_id,
                            embedding=vec.tolist(),
                            model_name=self.text_gen.model_name,
                        )
                        text_embedding_count += 1

            visual_embedding_count = 0
            if self.enable_visual_embeddings and scenes_inserted:
                scene_with_frame: list[tuple[int, str, float | None]] = []
                for s in scenes_inserted:
                    keyframe = self._resolve_path(s.get("keyframe_path"))
                    if keyframe and keyframe.exists():
                        midpoint = (float(s["start_time"]) + float(s["end_time"])) / 2.0
                        scene_with_frame.append((int(s["db_scene_id"]), str(keyframe), midpoint))

                for i in range(0, len(scene_with_frame), self.visual_batch_size):
                    batch = scene_with_frame[i : i + self.visual_batch_size]
                    paths = [x[1] for x in batch]
                    vectors = self.vision_gen.encode_images(
                        paths, batch_size=len(paths), show_progress=False, normalize=True
                    )
                    for (db_scene_id, keyframe_path, midpoint), vec in zip(batch, vectors):
                        self._insert_visual_embedding(
                            conn=conn,
                            scene_id=db_scene_id,
                            keyframe_path=keyframe_path,
                            sample_time=midpoint,
                            frame_role="mid",
                            frame_index=None,
                            embedding=vec.tolist(),
                            model_name=self.vision_gen.model_name,
                        )
                        visual_embedding_count += 1

        return {
            "video": filename,
            "video_id": int(video_id),
            "scenes": len(scenes_raw),
            "segments": len(segment_rows),
            "text_embeddings": text_embedding_count,
            "visual_embeddings": visual_embedding_count,
        }

    @staticmethod
    def _delete_existing_document(conn, document_id: int) -> None:
        # Chunk/embedding rows cascade from documents -> document_chunks.
        conn.execute(text("DELETE FROM dbo.documents WHERE id = :document_id"), {"document_id": document_id})

    def ingest_document_result_file(self, result_file: Path) -> dict[str, Any]:
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        meta = payload.get("metadata", {})
        chunks = payload.get("chunks", []) or []

        filename = meta.get("filename")
        file_path = meta.get("file_path")
        if not filename or not file_path:
            raise ValueError(f"Missing document metadata filename/file_path in {result_file}")

        with engine.begin() as conn:
            existing_document_id = conn.execute(
                text(
                    """
                    SELECT TOP (1) id
                    FROM dbo.documents
                    WHERE filename = :filename
                      AND file_path = :file_path
                    """
                ),
                {"filename": filename, "file_path": file_path},
            ).scalar()
            if existing_document_id:
                self._delete_existing_document(conn, int(existing_document_id))

            document_id = conn.execute(
                text(
                    """
                    INSERT INTO dbo.documents
                        (filename, file_path, file_type, file_size_mb, total_pages,
                         extraction_method, ocr_model, [language])
                    OUTPUT INSERTED.id
                    VALUES
                        (:filename, :file_path, :file_type, :file_size_mb, :total_pages,
                         :extraction_method, :ocr_model, :language)
                    """
                ),
                {
                    "filename": filename,
                    "file_path": file_path,
                    "file_type": meta.get("file_type"),
                    "file_size_mb": meta.get("file_size_mb"),
                    "total_pages": meta.get("total_pages"),
                    "extraction_method": meta.get("extraction_method"),
                    "ocr_model": meta.get("ocr_model"),
                    "language": meta.get("language", "en"),
                },
            ).scalar_one()

            chunk_rows: list[tuple[int, str]] = []
            for idx, chunk in enumerate(chunks):
                chunk_index = int(chunk.get("chunk_index", idx))
                chunk_text = str(chunk.get("text", "")).strip()
                summary = self._normalize_optional_text(chunk.get("summary"))

                chunk_id = conn.execute(
                    text(
                        """
                        INSERT INTO dbo.document_chunks
                            (document_id, chunk_index, page_number, section_heading, [text], summary, ocr_confidence)
                        OUTPUT INSERTED.id
                        VALUES
                            (:document_id, :chunk_index, :page_number, :section_heading, :text, :summary, :ocr_confidence)
                        """
                    ),
                    {
                        "document_id": int(document_id),
                        "chunk_index": chunk_index,
                        "page_number": chunk.get("page_number"),
                        "section_heading": self._normalize_optional_text(
                            chunk.get("section_heading")
                        ),
                        "text": chunk_text,
                        "summary": summary,
                        "ocr_confidence": chunk.get("ocr_confidence"),
                    },
                ).scalar_one()

                embed_text = chunk_text
                if summary:
                    embed_text = f"[{summary}] {chunk_text}"
                chunk_rows.append((int(chunk_id), embed_text))

            doc_embedding_count = 0
            if self.enable_text_embeddings and chunk_rows:
                vectors = self.text_gen.encode(
                    [x[1] for x in chunk_rows],
                    batch_size=min(self.text_batch_size, 16),
                    show_progress=True,
                )
                for (chunk_id, _txt), vec in zip(chunk_rows, vectors):
                    self._insert_document_embedding(
                        conn=conn,
                        chunk_id=chunk_id,
                        embedding=vec.tolist(),
                        model_name=self.text_gen.model_name,
                    )
                    doc_embedding_count += 1

        return {
            "document": filename,
            "document_id": int(document_id),
            "chunks": len(chunk_rows),
            "document_embeddings": doc_embedding_count,
        }

    def ingest_videos(self, base_dir: Path, limit: int | None = None) -> dict[str, Any]:
        files = self._iter_video_result_files(base_dir)
        if limit is not None:
            files = files[:limit]

        stats = {"total": len(files), "ok": 0, "failed": 0, "errors": []}
        if not files:
            print(f"[WARN] No video results found in: {base_dir}")
            return stats

        print(f"[INFO] Video result files: {len(files)}")
        for idx, f in enumerate(files, 1):
            print(f"[VIDEO {idx}/{len(files)}] {f}")
            try:
                out = self.ingest_video_result_file(f)
                stats["ok"] += 1
                print(
                    f"  [OK] video_id={out['video_id']}, scenes={out['scenes']}, "
                    f"segments={out['segments']}, text_emb={out['text_embeddings']}, "
                    f"visual_emb={out['visual_embeddings']}"
                )
            except Exception as exc:
                stats["failed"] += 1
                stats["errors"].append({"file": str(f), "error": str(exc)})
                print(f"  [ERROR] {exc}")
        return stats

    def ingest_documents(self, base_dir: Path, limit: int | None = None) -> dict[str, Any]:
        files = self._iter_document_result_files(base_dir)
        if limit is not None:
            files = files[:limit]

        stats = {"total": len(files), "ok": 0, "failed": 0, "errors": []}
        if not files:
            print(f"[WARN] No document results found in: {base_dir}")
            return stats

        print(f"[INFO] Document result files: {len(files)}")
        for idx, f in enumerate(files, 1):
            print(f"[DOC {idx}/{len(files)}] {f}")
            try:
                out = self.ingest_document_result_file(f)
                stats["ok"] += 1
                print(
                    f"  [OK] document_id={out['document_id']}, chunks={out['chunks']}, "
                    f"doc_emb={out['document_embeddings']}"
                )
            except Exception as exc:
                stats["failed"] += 1
                stats["errors"].append({"file": str(f), "error": str(exc)})
                print(f"  [ERROR] {exc}")
        return stats


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest processed results into SQL Server VideoSemanticDB.")
    p.add_argument("--videos-base", type=str, default="processed/results")
    p.add_argument("--documents-base", type=str, default="processed/documents")
    p.add_argument("--skip-videos", action="store_true")
    p.add_argument("--skip-documents", action="store_true")
    p.add_argument("--limit-videos", type=int, default=None)
    p.add_argument("--limit-documents", type=int, default=None)
    p.add_argument("--no-text-embeddings", action="store_true")
    p.add_argument("--no-visual-embeddings", action="store_true")
    p.add_argument("--text-batch-size", type=int, default=32)
    p.add_argument("--visual-batch-size", type=int, default=16)
    p.add_argument("--text-model", type=str, default=DEFAULT_TEXT_MODEL)
    p.add_argument("--vision-model", type=str, default=DEFAULT_VISION_MODEL)
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    if not test_connection():
        return 1

    ingester = SqlServerIngester(
        text_batch_size=args.text_batch_size,
        visual_batch_size=args.visual_batch_size,
        enable_text_embeddings=not args.no_text_embeddings,
        enable_visual_embeddings=not args.no_visual_embeddings,
        text_model_name=args.text_model,
        vision_model_name=args.vision_model,
    )

    print("=" * 72)
    print("SQL SERVER INGESTION")
    print("=" * 72)

    video_stats = {"total": 0, "ok": 0, "failed": 0, "errors": []}
    doc_stats = {"total": 0, "ok": 0, "failed": 0, "errors": []}

    if not args.skip_videos:
        video_stats = ingester.ingest_videos(Path(args.videos_base), limit=args.limit_videos)
    if not args.skip_documents:
        doc_stats = ingester.ingest_documents(
            Path(args.documents_base), limit=args.limit_documents
        )

    print("=" * 72)
    print("INGESTION SUMMARY")
    print("=" * 72)
    print(
        f"Videos:    total={video_stats['total']} ok={video_stats['ok']} failed={video_stats['failed']}"
    )
    print(
        f"Documents: total={doc_stats['total']} ok={doc_stats['ok']} failed={doc_stats['failed']}"
    )

    if video_stats["errors"] or doc_stats["errors"]:
        print("\nErrors:")
        for err in [*video_stats["errors"], *doc_stats["errors"]]:
            print(f"  - {err['file']}: {err['error']}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
