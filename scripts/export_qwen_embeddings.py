#!/usr/bin/env python3
"""
Export embeddings from existing processed results.

Input patterns:
- processed/results/<video_name>/results.json
- processed/documents/<document_name>/results/results.json

Output:
- <output_dir>/embeddings.jsonl
- <output_dir>/manifest.json
- <output_dir>/errors.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def unique_nonempty(parts: Iterable[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for part in parts:
        part = clean_text(part)
        if not part or part in seen:
            continue
        seen.add(part)
        out.append(part)
    return out


@dataclass
class ExportRecord:
    content_hash: str
    source_type: str                  # video | document
    source_name: str
    source_filename: str
    source_file: str
    record_type: str                  # transcript_segment | scene | document_chunk
    segment_id: Optional[int]
    scene_id: Optional[int]
    page_number: Optional[int]
    chunk_index: Optional[int]
    start_time: Optional[float]
    end_time: Optional[float]
    keyframe_path: Optional[str]
    caption: Optional[str]
    ocr_text: Optional[str]
    text_for_embedding: str
    embedding_model: str
    embedding_dim: int
    embedding: List[float]
    created_at: str


def discover_result_files(processed_root: Path) -> List[Path]:
    files: List[Path] = []
    files.extend(sorted((processed_root / "results").glob("*/results.json")))
    files.extend(sorted((processed_root / "documents").glob("*/results/results.json")))
    return files


def is_video_result(path: Path) -> bool:
    # processed/results/<video_name>/results.json
    return len(path.parts) >= 3 and "results" in path.parts and "documents" not in path.parts


def build_video_records(data: Dict[str, Any], result_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    video_meta = data.get("video", {}) or {}
    source_name = result_path.parent.name
    source_filename = video_meta.get("filename") or source_name
    source_file = str(result_path)

    transcription = data.get("transcription", {}) or {}
    for seg in transcription.get("segments", []) or []:
        seg_text = clean_text(seg.get("text"))
        if not seg_text:
            continue

        records.append(
            {
                "source_type": "video",
                "source_name": source_name,
                "source_filename": source_filename,
                "source_file": source_file,
                "record_type": "transcript_segment",
                "segment_id": seg.get("id"),
                "scene_id": None,
                "page_number": None,
                "chunk_index": None,
                "start_time": seg.get("start"),
                "end_time": seg.get("end"),
                "keyframe_path": None,
                "caption": None,
                "ocr_text": None,
                "text_for_embedding": seg_text,
            }
        )

    scene_analysis = data.get("scene_analysis", {}) or {}
    for scene in scene_analysis.get("scenes", []) or []:
        transcript_bits = [
            clean_text(ts.get("text"))
            for ts in (scene.get("transcript_segments") or [])
        ]
        text_parts = unique_nonempty(
            [
                scene.get("caption"),
                scene.get("ocr_text"),
                *transcript_bits,
            ]
        )
        scene_text = "\n".join(text_parts).strip()
        if not scene_text:
            continue

        records.append(
            {
                "source_type": "video",
                "source_name": source_name,
                "source_filename": source_filename,
                "source_file": source_file,
                "record_type": "scene",
                "segment_id": None,
                "scene_id": scene.get("scene_id"),
                "page_number": None,
                "chunk_index": None,
                "start_time": scene.get("start_time"),
                "end_time": scene.get("end_time"),
                "keyframe_path": scene.get("keyframe_path"),
                "caption": clean_text(scene.get("caption")) or None,
                "ocr_text": clean_text(scene.get("ocr_text")) or None,
                "text_for_embedding": scene_text,
            }
        )

    return records


def build_document_records(data: Dict[str, Any], result_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    metadata = data.get("metadata", {}) or {}
    source_name = result_path.parent.parent.name  # processed/documents/<name>/results/results.json
    source_filename = metadata.get("filename") or source_name
    source_file = str(result_path)

    for chunk in data.get("chunks", []) or []:
        chunk_text = clean_text(chunk.get("text"))
        if not chunk_text:
            continue

        records.append(
            {
                "source_type": "document",
                "source_name": source_name,
                "source_filename": source_filename,
                "source_file": source_file,
                "record_type": "document_chunk",
                "segment_id": None,
                "scene_id": None,
                "page_number": chunk.get("page_number"),
                "chunk_index": chunk.get("chunk_index"),
                "start_time": None,
                "end_time": None,
                "keyframe_path": None,
                "caption": None,
                "ocr_text": None,
                "text_for_embedding": chunk_text,
            }
        )

    return records


def batched(items: List[Dict[str, Any]], batch_size: int) -> Iterator[List[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_model(model_name: str, device: str, max_seq_length: int) -> SentenceTransformer:
    model = SentenceTransformer(
        model_name,
        device=device,
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = max_seq_length
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True, help="Root processed folder")
    parser.add_argument("--output-dir", required=True, help="Folder to write embeddings export")
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--output-dim", type=int, default=1024)
    parser.add_argument("--min-chars", type=int, default=5)
    args = parser.parse_args()

    processed_root = Path(args.processed_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "embeddings.jsonl"
    manifest_path = output_dir / "manifest.json"
    errors_path = output_dir / "errors.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_name, device=device, max_seq_length=args.max_seq_length)

    result_files = discover_result_files(processed_root)
    candidate_records: List[Dict[str, Any]] = []
    errors = 0

    for result_file in result_files:
        try:
            with result_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if is_video_result(result_file):
                built = build_video_records(data, result_file)
            else:
                built = build_document_records(data, result_file)

            for row in built:
                if len(row["text_for_embedding"]) >= args.min_chars:
                    candidate_records.append(row)

        except Exception as e:
            errors += 1
            with errors_path.open("a", encoding="utf-8") as ef:
                ef.write(json.dumps({
                    "source_file": str(result_file),
                    "stage": "parse",
                    "error": str(e),
                    "created_at": utc_now(),
                }, ensure_ascii=False) + "\n")

    written = 0
    with embeddings_path.open("w", encoding="utf-8") as out_f:
        for batch in batched(candidate_records, args.batch_size):
            texts = [row["text_for_embedding"] for row in batch]
            try:
                vecs = model.encode(
                    texts,
                    batch_size=len(texts),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                for row, vec in zip(batch, vecs):
                    vec = np.asarray(vec, dtype=np.float32)
                    if args.output_dim and args.output_dim < vec.shape[0]:
                        vec = vec[: args.output_dim]

                    key = "|".join([
                        row["source_file"],
                        row["record_type"],
                        str(row.get("segment_id")),
                        str(row.get("scene_id")),
                        str(row.get("page_number")),
                        str(row.get("chunk_index")),
                        row["text_for_embedding"],
                    ])

                    record = ExportRecord(
                        content_hash=sha1_text(key),
                        source_type=row["source_type"],
                        source_name=row["source_name"],
                        source_filename=row["source_filename"],
                        source_file=row["source_file"],
                        record_type=row["record_type"],
                        segment_id=row.get("segment_id"),
                        scene_id=row.get("scene_id"),
                        page_number=row.get("page_number"),
                        chunk_index=row.get("chunk_index"),
                        start_time=row.get("start_time"),
                        end_time=row.get("end_time"),
                        keyframe_path=row.get("keyframe_path"),
                        caption=row.get("caption"),
                        ocr_text=row.get("ocr_text"),
                        text_for_embedding=row["text_for_embedding"],
                        embedding_model=args.model_name,
                        embedding_dim=int(vec.shape[0]),
                        embedding=vec.tolist(),
                        created_at=utc_now(),
                    )
                    out_f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                    written += 1

            except Exception as e:
                errors += len(batch)
                with errors_path.open("a", encoding="utf-8") as ef:
                    for row in batch:
                        ef.write(json.dumps({
                            "source_file": row["source_file"],
                            "record_type": row["record_type"],
                            "segment_id": row.get("segment_id"),
                            "scene_id": row.get("scene_id"),
                            "page_number": row.get("page_number"),
                            "chunk_index": row.get("chunk_index"),
                            "stage": "encode",
                            "error": str(e),
                            "created_at": utc_now(),
                        }, ensure_ascii=False) + "\n")

    manifest = {
        "created_at": utc_now(),
        "processed_root": str(processed_root),
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "device": device,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "output_dim": args.output_dim,
        "result_files_found": len(result_files),
        "records_to_embed": len(candidate_records),
        "records_written": written,
        "errors": errors,
    }
    with manifest_path.open("w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
