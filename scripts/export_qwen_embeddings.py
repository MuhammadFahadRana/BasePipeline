#!/usr/bin/env python3
"""
Generate embeddings from existing processed results and export to JSONL.

Expected inputs:
- processed/results/<video_name>/results.json
- processed/documents/<document_name>/results/results.json

Outputs:
- <output_dir>/embeddings.jsonl
- <output_dir>/manifest.json
- <output_dir>/errors.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


TEXT_KEYS_PRIORITY = [
    "text",
    "content",
    "transcript",
    "transcript_text",
    "ocr_text",
    "ocr",
    "caption",
    "captions",
    "description",
    "summary",
    "title",
    "heading",
    "page_text",
    "chunk_text",
    "scene_text",
    "labels",
]

LIST_HINT_KEYS = {
    "segments",
    "transcript_segments",
    "transcript",
    "scenes",
    "chunks",
    "pages",
    "paragraphs",
    "sections",
    "items",
    "results",
}

ID_HINT_KEYS = ["segment_id", "scene_id", "chunk_id", "id", "page", "page_number", "index"]
START_KEYS = ["start_time", "start", "begin"]
END_KEYS = ["end_time", "end", "stop"]


@dataclass
class EmbeddingRecord:
    source_hash: str
    source_type: str                 # "video" or "document"
    source_name: str
    source_file: str
    item_kind: str
    item_path: str
    item_index: int
    logical_id: Optional[str]
    scene_id: Optional[str]
    segment_id: Optional[str]
    start_time: Optional[float]
    end_time: Optional[float]
    text_for_embedding: str
    model_name: str
    embedding_dim: int
    embedding: List[float]
    created_at: str


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def trim_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def join_text_values(values: List[str], max_chars: int) -> str:
    seen = set()
    cleaned: List[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        value = normalize_whitespace(value)
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return trim_text("\n\n".join(cleaned), max_chars=max_chars)


def flatten_string_values(obj: Any, key_hint_only: bool = False, parent_key: str = "") -> List[str]:
    out: List[str] = []

    if isinstance(obj, str):
        if not key_hint_only or parent_key.lower() in TEXT_KEYS_PRIORITY:
            out.append(obj)
        return out

    if isinstance(obj, list):
        for item in obj:
            out.extend(flatten_string_values(item, key_hint_only=key_hint_only, parent_key=parent_key))
        return out

    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(flatten_string_values(v, key_hint_only=key_hint_only, parent_key=k))
        return out

    return out


def extract_preferred_text(item: Dict[str, Any], max_chars: int) -> str:
    values: List[str] = []

    # Preferred keys first
    for key in TEXT_KEYS_PRIORITY:
        if key in item:
            v = item[key]
            if isinstance(v, str):
                values.append(v)
            elif isinstance(v, list):
                values.extend(flatten_string_values(v))
            elif isinstance(v, dict):
                values.extend(flatten_string_values(v))

    # Fallback: all text-like values under text-ish keys
    if not values:
        values.extend(flatten_string_values(item, key_hint_only=True))

    # Last resort: all strings anywhere in the item
    if not values:
        values.extend(flatten_string_values(item, key_hint_only=False))

    return join_text_values(values, max_chars=max_chars)


def walk_nodes(obj: Any, path: str = "$") -> Iterator[Tuple[str, Any]]:
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_nodes(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_nodes(v, f"{path}[{i}]")


def guess_source_name(source_file: Path, source_type: str) -> str:
    if source_type == "video":
        # processed/results/<video_name>/results.json
        return source_file.parent.name
    # processed/documents/<doc_name>/results/results.json
    if source_file.parent.name == "results" and source_file.parent.parent:
        return source_file.parent.parent.name
    return source_file.parent.name


def discover_results_files(processed_root: Path) -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []

    video_glob = processed_root / "results"
    if video_glob.exists():
        for p in sorted(video_glob.glob("*/results.json")):
            files.append(("video", p))

    docs_glob = processed_root / "documents"
    if docs_glob.exists():
        for p in sorted(docs_glob.glob("*/results/results.json")):
            files.append(("document", p))

    return files


def extract_candidate_items(
    data: Any,
    source_type: str,
    source_file: Path,
    max_chars: int,
) -> List[Dict[str, Any]]:
    """
    Tries to find segment-like lists first.
    Falls back to one whole-file record.
    """
    records: List[Dict[str, Any]] = []

    for path, node in walk_nodes(data):
        if not isinstance(node, list) or not node:
            continue

        key_name = path.split(".")[-1]
        key_name = key_name.split("[")[0].lower()

        if key_name not in LIST_HINT_KEYS:
            continue

        if not all(isinstance(x, dict) for x in node[: min(5, len(node))]):
            continue

        for idx, item in enumerate(node):
            text = extract_preferred_text(item, max_chars=max_chars)
            if not text:
                continue

            logical_id = None
            for key in ID_HINT_KEYS:
                if key in item and item[key] is not None:
                    logical_id = str(item[key])
                    break

            scene_id = str(item.get("scene_id")) if item.get("scene_id") is not None else None
            segment_id = str(item.get("segment_id")) if item.get("segment_id") is not None else None

            start_time = None
            end_time = None
            for key in START_KEYS:
                if key in item:
                    start_time = safe_float(item[key])
                    break
            for key in END_KEYS:
                if key in item:
                    end_time = safe_float(item[key])
                    break

            records.append(
                {
                    "source_type": source_type,
                    "source_name": guess_source_name(source_file, source_type),
                    "source_file": str(source_file),
                    "item_kind": key_name,
                    "item_path": path,
                    "item_index": idx,
                    "logical_id": logical_id,
                    "scene_id": scene_id,
                    "segment_id": segment_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "text_for_embedding": text,
                }
            )

    if records:
        return records

    # Fallback: embed the whole file as one record
    if isinstance(data, dict):
        text = extract_preferred_text(data, max_chars=max_chars)
    else:
        text = join_text_values(flatten_string_values(data), max_chars=max_chars)

    if text:
        records.append(
            {
                "source_type": source_type,
                "source_name": guess_source_name(source_file, source_type),
                "source_file": str(source_file),
                "item_kind": "whole_result",
                "item_path": "$",
                "item_index": 0,
                "logical_id": None,
                "scene_id": None,
                "segment_id": None,
                "start_time": None,
                "end_time": None,
                "text_for_embedding": text,
            }
        )

    return records


def batched(items: List[Dict[str, Any]], size: int) -> Iterator[List[Dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


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
    parser.add_argument("--processed-root", required=True, help="Path to processed root folder")
    parser.add_argument("--output-dir", required=True, help="Directory for embeddings export")
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--max-text-chars", type=int, default=12000)
    parser.add_argument(
        "--output-dim",
        type=int,
        default=1024,
        help="Dimension to keep in export. Use 1024 if you want easy SQL Server VECTOR compatibility.",
    )
    parser.add_argument("--min-text-chars", type=int, default=10)
    args = parser.parse_args()

    processed_root = Path(args.processed_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "embeddings.jsonl"
    errors_path = output_dir / "errors.jsonl"
    manifest_path = output_dir / "manifest.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_name, device=device, max_seq_length=args.max_seq_length)

    discovered_files = discover_results_files(processed_root)

    all_items: List[Dict[str, Any]] = []
    file_count = 0
    error_count = 0

    for source_type, source_file in discovered_files:
        file_count += 1
        try:
            with source_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            items = extract_candidate_items(
                data=data,
                source_type=source_type,
                source_file=source_file,
                max_chars=args.max_text_chars,
            )
            for item in items:
                if len(item["text_for_embedding"].strip()) >= args.min_text_chars:
                    all_items.append(item)

        except Exception as e:
            error_count += 1
            with errors_path.open("a", encoding="utf-8") as ef:
                ef.write(
                    json.dumps(
                        {
                            "source_file": str(source_file),
                            "stage": "load_or_extract",
                            "error": str(e),
                            "created_at": now_utc_iso(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    written = 0
    with embeddings_path.open("w", encoding="utf-8") as out_f:
        for batch in batched(all_items, args.batch_size):
            texts = [x["text_for_embedding"] for x in batch]
            try:
                vectors = model.encode(
                    texts,
                    batch_size=len(texts),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                for item, vec in zip(batch, vectors):
                    vec = np.asarray(vec, dtype=np.float32)
                    if args.output_dim and args.output_dim < vec.shape[0]:
                        vec = vec[: args.output_dim]

                    payload = EmbeddingRecord(
                        source_hash=sha1_text(
                            f"{item['source_file']}|{item['item_path']}|{item['item_index']}|{item['text_for_embedding']}"
                        ),
                        source_type=item["source_type"],
                        source_name=item["source_name"],
                        source_file=item["source_file"],
                        item_kind=item["item_kind"],
                        item_path=item["item_path"],
                        item_index=item["item_index"],
                        logical_id=item["logical_id"],
                        scene_id=item["scene_id"],
                        segment_id=item["segment_id"],
                        start_time=item["start_time"],
                        end_time=item["end_time"],
                        text_for_embedding=item["text_for_embedding"],
                        model_name=args.model_name,
                        embedding_dim=int(vec.shape[0]),
                        embedding=vec.tolist(),
                        created_at=now_utc_iso(),
                    )

                    out_f.write(json.dumps(asdict(payload), ensure_ascii=False) + "\n")
                    written += 1

            except Exception as e:
                error_count += len(batch)
                with errors_path.open("a", encoding="utf-8") as ef:
                    for item in batch:
                        ef.write(
                            json.dumps(
                                {
                                    "source_file": item["source_file"],
                                    "item_path": item["item_path"],
                                    "item_index": item["item_index"],
                                    "stage": "encode",
                                    "error": str(e),
                                    "created_at": now_utc_iso(),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

    manifest = {
        "created_at": now_utc_iso(),
        "processed_root": str(processed_root),
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "device": device,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "max_text_chars": args.max_text_chars,
        "output_dim": args.output_dim,
        "result_files_found": len(discovered_files),
        "candidate_items_found": len(all_items),
        "embedding_records_written": written,
        "errors_written": error_count,
    }

    with manifest_path.open("w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()