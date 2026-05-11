"""
Evaluate ATLAS retrieval variants from processed artifacts.

This script is intentionally offline-first: it reads processed/results.json,
training/retrieval_dataset.json, and training/relevance_judgments.csv, then
evaluates BM25, dense, hybrid, and modality-ablation variants without requiring
PostgreSQL or the FastAPI service to be running.

Examples:
    python -m training.evaluate_retrieval_variants --methods lexical
    python -m training.evaluate_retrieval_variants --methods all --max-queries 50
    python -m training.evaluate_retrieval_variants --methods all --include-lora

Outputs:
    <output-dir>/retrieval_variant_summary.csv
    <output-dir>/retrieval_variant_summary.tex
    <output-dir>/retrieval_per_query.csv
    <output-dir>/retrieval_topk_results.csv
    <output-dir>/retrieval_run_config.json
    <output-dir>/research_question_evidence.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


TEXT_MISSING = {"", "none", "null", "n/a", "na", "no text", "no visible text"}
DEFAULT_K_VALUES = (1, 3, 5, 10)
DEFAULT_EVAL_DEPTH = 10


@dataclass(frozen=True)
class EvidenceUnit:
    unit_id: str
    video: str
    segment_index: int
    scene_id: Optional[int]
    start_time: float
    end_time: float
    transcript: str
    ocr: str
    caption: str
    labels: Tuple[str, ...]

    def text_for(self, evidence_mode: str) -> str:
        evidence_mode = evidence_mode.lower()
        parts: List[str] = []

        if evidence_mode in {"transcript", "transcript_ocr", "transcript_caption", "full"}:
            parts.append(self.transcript)
        if evidence_mode in {"ocr", "transcript_ocr", "full"}:
            parts.append(self.ocr)
        if evidence_mode in {"caption", "transcript_caption", "full"}:
            parts.append(self.caption)
        if evidence_mode == "full" and self.labels:
            parts.append(" ".join(self.labels))

        return " ".join(p.strip() for p in parts if _has_text(p)).strip()


@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    query: str
    query_type: str
    language: str
    video: str
    start_time: float
    end_time: float
    segment_index: Optional[int] = None


@dataclass(frozen=True)
class Variant:
    name: str
    display_name: str
    group: str
    method: str
    evidence_mode: str
    dense_model: str = "base"
    dense_weight: float = 0.65
    bm25_weight: float = 0.35
    description: str = ""


def _has_text(value: object) -> bool:
    text = str(value or "").strip()
    return text.lower() not in TEXT_MISSING


def _norm_text(value: object) -> str:
    return str(value or "").replace("\x00", " ").strip()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\w]+", (text or "").lower(), flags=re.UNICODE)


def _sha1_texts(parts: Iterable[str]) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="ignore"))
        h.update(b"\n---\n")
    return h.hexdigest()


def _round4(value: float) -> float:
    if value is None or not np.isfinite(value):
        return float("nan")
    return round(float(value), 4)


def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    intersection = max(0.0, inter_end - inter_start)
    union = (a_end - a_start) + (b_end - b_start) - intersection
    return intersection / union if union > 0 else 0.0


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def discover_result_files(processed_dir: Path, source: str) -> List[Path]:
    source = (source or "auto").lower()
    candidates: List[Path] = []

    if source in {"auto", "results"}:
        candidates = sorted((processed_dir / "results").glob("*/results.json"))
        if candidates or source == "results":
            return candidates

    if source in {"auto", "whisper"}:
        candidates = sorted((processed_dir / "Whisper-Large-v3").glob("*/results.json"))
        if candidates or source == "whisper":
            return candidates

    if source == "all":
        candidates = sorted(processed_dir.glob("*/*/results.json"))
        seen = set()
        unique = []
        for path in candidates:
            key = str(path.resolve()).lower()
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    return sorted(processed_dir.glob("*/*/results.json"))


def _scene_candidates(payload: Dict) -> List[Dict]:
    if isinstance(payload.get("scene_analysis"), dict):
        scenes = payload["scene_analysis"].get("scenes")
        if isinstance(scenes, list):
            return scenes
    if isinstance(payload.get("scenes"), list):
        return payload["scenes"]
    return []


def _scene_id_value(scene: Dict) -> Optional[int]:
    for key in ("scene_id", "id", "index"):
        if key in scene and scene[key] is not None:
            try:
                return int(scene[key])
            except Exception:
                return None
    return None


def _match_scene(segment: Dict, scenes: Sequence[Dict], scene_by_id: Dict[int, Dict]) -> Tuple[Optional[int], Optional[Dict]]:
    for key in ("scene_id", "scene", "scene_index"):
        if key in segment and segment[key] is not None:
            try:
                sid = int(segment[key])
                if sid in scene_by_id:
                    return sid, scene_by_id[sid]
            except Exception:
                pass

    start = float(segment.get("start", segment.get("start_time", 0)) or 0)
    for scene in scenes:
        try:
            scene_start = float(scene.get("start_time", scene.get("start", 0)) or 0)
            scene_end = float(scene.get("end_time", scene.get("end", 0)) or 0)
        except Exception:
            continue
        if scene_start <= start <= scene_end:
            return _scene_id_value(scene), scene
    return None, None


def load_units_from_processed(processed_dir: Path, source: str = "auto") -> List[EvidenceUnit]:
    result_files = discover_result_files(processed_dir, source)
    if not result_files:
        raise FileNotFoundError(f"No results.json files found under {processed_dir}")

    units: List[EvidenceUnit] = []
    seen = set()

    for result_file in result_files:
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"[WARN] Skipping {result_file}: {exc}")
            continue

        filename = payload.get("video", {}).get("filename", result_file.parent.name)
        video = Path(filename).stem
        scenes = _scene_candidates(payload)
        scene_by_id = {
            sid: scene
            for scene in scenes
            for sid in [_scene_id_value(scene)]
            if sid is not None
        }

        segments = payload.get("transcription", {}).get("segments", [])
        for idx, segment in enumerate(segments):
            transcript = _norm_text(segment.get("text", ""))
            if not _has_text(transcript):
                continue

            start_time = float(segment.get("start", segment.get("start_time", 0)) or 0)
            end_time = float(segment.get("end", segment.get("end_time", start_time)) or start_time)
            scene_id, scene = _match_scene(segment, scenes, scene_by_id)
            scene = scene or {}

            labels = scene.get("object_labels") or scene.get("labels") or []
            if isinstance(labels, str):
                labels = [labels]
            labels_tuple = tuple(str(item).strip() for item in labels if str(item).strip())

            ocr = _norm_text(
                segment.get("ocr_text")
                or scene.get("ocr_text")
                or scene.get("ocr")
                or scene.get("text_on_screen")
                or ""
            )
            caption = _norm_text(
                segment.get("caption")
                or scene.get("caption")
                or scene.get("description")
                or scene.get("summary")
                or ""
            )

            segment_index = int(segment.get("segment_index", idx))
            unit_id = f"{video}::{segment_index}::{start_time:.2f}-{end_time:.2f}"
            dedupe_key = (video, segment_index, round(start_time, 2), round(end_time, 2), transcript)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            units.append(
                EvidenceUnit(
                    unit_id=unit_id,
                    video=video,
                    segment_index=segment_index,
                    scene_id=scene_id,
                    start_time=start_time,
                    end_time=end_time,
                    transcript=transcript,
                    ocr=ocr,
                    caption=caption,
                    labels=labels_tuple,
                )
            )

    units.sort(key=lambda u: (u.video.lower(), u.segment_index, u.start_time))
    return units


def load_eval_queries(dataset_path: Path, judgments_path: Optional[Path]) -> List[EvalQuery]:
    queries: List[EvalQuery] = []

    if judgments_path and judgments_path.exists():
        grouped: Dict[str, List[Dict[str, str]]] = {}
        with open(judgments_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                grouped.setdefault(row.get("query_id", ""), []).append(row)

        for query_id, rows in grouped.items():
            rows.sort(key=lambda r: int(r.get("relevance", 0) or 0), reverse=True)
            best = rows[0]
            if int(best.get("relevance", 0) or 0) < 2:
                continue
            queries.append(
                EvalQuery(
                    query_id=query_id,
                    query=best.get("query_text", ""),
                    query_type=best.get("query_type", ""),
                    language=best.get("language", ""),
                    video=best.get("video", ""),
                    start_time=float(best.get("start_time", 0) or 0),
                    end_time=float(best.get("end_time", 0) or 0),
                    segment_index=int(best["segment_index"]) if best.get("segment_index", "").strip() else None,
                )
            )

    if queries:
        return queries

    with open(dataset_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for entry in payload.get("entries", []):
        pos = entry.get("positive", {}) or {}
        queries.append(
            EvalQuery(
                query_id=entry.get("query_id", ""),
                query=entry.get("query", ""),
                query_type=entry.get("query_type", ""),
                language=entry.get("language", ""),
                video=entry.get("video", ""),
                start_time=float(pos.get("start_time", 0) or 0),
                end_time=float(pos.get("end_time", 0) or 0),
                segment_index=pos.get("segment_index"),
            )
        )
    return queries


def default_variants(include_lora: bool) -> List[Variant]:
    variants = [
        Variant(
            name="bm25_only",
            display_name="BM25-only",
            group="method_baseline",
            method="bm25",
            evidence_mode="full",
            dense_model="none",
            dense_weight=0.0,
            bm25_weight=1.0,
            description="Lexical BM25 over transcript, OCR, captions, and labels.",
        ),
        Variant(
            name="dense_only",
            display_name="Dense-only",
            group="method_baseline",
            method="dense",
            evidence_mode="transcript",
            dense_model="base",
            dense_weight=1.0,
            bm25_weight=0.0,
            description="Qwen3 dense retrieval over transcript text only.",
        ),
        Variant(
            name="hybrid",
            display_name="Hybrid",
            group="method_baseline",
            method="hybrid",
            evidence_mode="transcript",
            dense_model="base",
            description="BM25 plus Qwen3 dense retrieval over transcript text.",
        ),
        Variant(
            name="transcript_only",
            display_name="Transcript only",
            group="modality_ablation",
            method="hybrid",
            evidence_mode="transcript",
            dense_model="base",
            description="Hybrid retrieval using transcript evidence only.",
        ),
        Variant(
            name="ocr_only",
            display_name="OCR only",
            group="modality_ablation",
            method="hybrid",
            evidence_mode="ocr",
            dense_model="base",
            description="Hybrid retrieval using OCR text only.",
        ),
        Variant(
            name="caption_only",
            display_name="Caption only",
            group="modality_ablation",
            method="hybrid",
            evidence_mode="caption",
            dense_model="base",
            description="Hybrid retrieval using visual captions only.",
        ),
        Variant(
            name="transcript_ocr",
            display_name="Transcript + OCR",
            group="modality_ablation",
            method="hybrid",
            evidence_mode="transcript_ocr",
            dense_model="base",
            description="Hybrid retrieval using transcript and OCR text.",
        ),
        Variant(
            name="transcript_captions",
            display_name="Transcript + captions",
            group="modality_ablation",
            method="hybrid",
            evidence_mode="transcript_caption",
            dense_model="base",
            description="Hybrid retrieval using transcript and visual captions.",
        ),
        Variant(
            name="full_multimodal_atlas",
            display_name="Full multimodal ATLAS",
            group="full_system",
            method="hybrid",
            evidence_mode="full",
            dense_model="base",
            description="Hybrid retrieval over transcript, OCR, captions, and labels.",
        ),
    ]
    if include_lora:
        variants.append(
            Variant(
                name="full_multimodal_atlas_lora",
                display_name="Full multimodal ATLAS + LoRA",
                group="full_system",
                method="hybrid",
                evidence_mode="full",
                dense_model="lora",
                description="Full hybrid retrieval with the local LoRA-adapted Qwen3 checkpoint.",
            )
        )
    return variants


def lexical_variants() -> List[Variant]:
    return [
        Variant(
            name="bm25_only",
            display_name="BM25-only",
            group="method_baseline",
            method="bm25",
            evidence_mode="full",
            dense_model="none",
            dense_weight=0.0,
            bm25_weight=1.0,
            description="Lexical BM25 over transcript, OCR, captions, and labels.",
        ),
        Variant(
            name="ocr_only_bm25",
            display_name="OCR only BM25",
            group="modality_ablation",
            method="bm25",
            evidence_mode="ocr",
            dense_model="none",
            dense_weight=0.0,
            bm25_weight=1.0,
            description="Lexical OCR-only ablation.",
        ),
        Variant(
            name="caption_only_bm25",
            display_name="Caption only BM25",
            group="modality_ablation",
            method="bm25",
            evidence_mode="caption",
            dense_model="none",
            dense_weight=0.0,
            bm25_weight=1.0,
            description="Lexical caption-only ablation.",
        ),
    ]


class EmbeddingProvider:
    def __init__(
        self,
        cache_dir: Path,
        base_model: str,
        lora_model: str,
        batch_size: int,
        device: Optional[str],
    ):
        self.cache_dir = cache_dir
        self.base_model = base_model
        self.lora_model = lora_model
        self.batch_size = batch_size
        self.device = device
        self._models: Dict[str, object] = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_model_path(self, model_spec: str) -> str:
        model_spec = str(model_spec)
        if model_spec == "base":
            model_spec = self.base_model
        elif model_spec == "lora":
            model_spec = self.lora_model

        if Path(model_spec).exists():
            return model_spec

        if "/" not in model_spec:
            return model_spec

        cached = _find_hf_snapshot(model_spec)
        return str(cached) if cached else model_spec

    def _load_model(self, model_spec: str):
        resolved = self._resolve_model_path(model_spec)
        if resolved in self._models:
            return self._models[resolved]

        from sentence_transformers import SentenceTransformer

        kwargs = {"trust_remote_code": True}
        if self.device:
            kwargs["device"] = self.device

        print(f"[dense] Loading model: {resolved}")
        model = SentenceTransformer(resolved, **kwargs)
        self._models[resolved] = model
        return model

    def encode(self, model_spec: str, texts: Sequence[str], cache_prefix: str) -> Tuple[np.ndarray, float, bool]:
        resolved = self._resolve_model_path(model_spec)
        text_hash = _sha1_texts(texts)
        key = hashlib.sha1(f"{resolved}|{cache_prefix}|{len(texts)}|{text_hash}".encode("utf-8")).hexdigest()[:20]
        cache_path = self.cache_dir / f"{cache_prefix}_{key}.npz"

        if cache_path.exists():
            payload = np.load(cache_path)
            return payload["embeddings"], 0.0, True

        model = self._load_model(model_spec)
        start = time.perf_counter()
        embeddings = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        elapsed = time.perf_counter() - start
        embeddings = np.asarray(embeddings, dtype=np.float32)
        np.savez_compressed(cache_path, embeddings=embeddings)
        return embeddings, elapsed, False


def _find_hf_snapshot(model_id: str) -> Optional[Path]:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_root / ("models--" + model_id.replace("/", "--"))
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        return None

    candidates = []
    for path in snapshots.iterdir():
        if not path.is_dir():
            continue
        if any((path / name).exists() for name in ("model.safetensors", "pytorch_model.bin", "modules.json")):
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.zeros_like(scores, dtype=np.float32)
    min_val = float(finite.min())
    max_val = float(finite.max())
    if max_val <= min_val:
        if max_val > 0:
            return np.ones_like(scores, dtype=np.float32)
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - min_val) / (max_val - min_val)).astype(np.float32)


def _rank_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return np.asarray([], dtype=np.int64)
    return np.argsort(-scores, kind="mergesort")


def _result_is_relevant(query: EvalQuery, unit: EvidenceUnit, iou_threshold: float) -> Tuple[bool, float]:
    if unit.video != query.video:
        return False, 0.0
    overlap = _iou(unit.start_time, unit.end_time, query.start_time, query.end_time)
    return overlap >= iou_threshold, overlap


def evaluate_variant(
    variant: Variant,
    all_units: Sequence[EvidenceUnit],
    queries: Sequence[EvalQuery],
    embedding_provider: Optional[EmbeddingProvider],
    k_values: Sequence[int],
    eval_depth: int,
    iou_threshold: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    texts = [unit.text_for(variant.evidence_mode) for unit in all_units]
    active_pairs = [(unit, text) for unit, text in zip(all_units, texts) if _has_text(text)]
    units = [item[0] for item in active_pairs]
    corpus_texts = [item[1] for item in active_pairs]

    if not units:
        raise ValueError(f"Variant {variant.name} has no non-empty evidence units")

    print(f"\n[{variant.name}] {variant.display_name}")
    print(f"  method={variant.method} evidence={variant.evidence_mode} corpus={len(units)}")

    build_start = time.perf_counter()
    bm25 = None
    tokenized_corpus: Optional[List[List[str]]] = None
    if variant.method in {"bm25", "hybrid"}:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [_tokenize(text) for text in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)

    corpus_embeddings = None
    query_embeddings = None
    corpus_encode_s = 0.0
    query_encode_s = 0.0
    corpus_cache_hit = False
    query_cache_hit = False

    if variant.method in {"dense", "hybrid"}:
        if embedding_provider is None:
            raise ValueError(f"Variant {variant.name} needs dense embeddings")
        corpus_embeddings, corpus_encode_s, corpus_cache_hit = embedding_provider.encode(
            variant.dense_model,
            corpus_texts,
            f"corpus_{variant.evidence_mode}_{variant.dense_model}",
        )
        query_embeddings, query_encode_s, query_cache_hit = embedding_provider.encode(
            variant.dense_model,
            [q.query for q in queries],
            f"queries_{variant.dense_model}",
        )

    build_s = time.perf_counter() - build_start

    ranks: List[float] = []
    first_ious: List[float] = []
    top1_ious: List[float] = []
    first_abs_errors: List[float] = []
    top1_abs_errors: List[float] = []
    per_query_rows: List[Dict[str, object]] = []
    topk_rows: List[Dict[str, object]] = []
    query_latencies_ms: List[float] = []
    query_encode_ms = (query_encode_s * 1000.0 / max(1, len(queries))) if query_encode_s else 0.0

    for query_idx, query in enumerate(queries):
        q_start = time.perf_counter()

        scores = np.zeros(len(units), dtype=np.float32)
        if variant.method in {"bm25", "hybrid"}:
            assert bm25 is not None
            bm25_scores = np.asarray(bm25.get_scores(_tokenize(query.query)), dtype=np.float32)
            if variant.method == "bm25":
                scores = bm25_scores
            else:
                scores += float(variant.bm25_weight) * _normalize_scores(bm25_scores)

        if variant.method in {"dense", "hybrid"}:
            assert corpus_embeddings is not None and query_embeddings is not None
            dense_scores = np.dot(corpus_embeddings, query_embeddings[query_idx]).astype(np.float32)
            if variant.method == "dense":
                scores = dense_scores
            else:
                scores += float(variant.dense_weight) * _normalize_scores(dense_scores)

        order = _rank_scores(scores)
        search_elapsed_ms = (time.perf_counter() - q_start) * 1000.0
        latency_ms = search_elapsed_ms + query_encode_ms
        query_latencies_ms.append(latency_ms)

        rank = math.inf
        first_iou = 0.0
        first_abs_error = math.inf
        top1_iou = 0.0
        top1_abs_error = math.inf
        top1_unit = units[int(order[0])] if len(order) else None
        top1_score = float(scores[int(order[0])]) if len(order) else float("nan")

        if top1_unit is not None and top1_unit.video == query.video:
            top1_iou = _iou(top1_unit.start_time, top1_unit.end_time, query.start_time, query.end_time)
            top1_abs_error = abs(top1_unit.start_time - query.start_time)

        for position, unit_idx in enumerate(order, start=1):
            unit = units[int(unit_idx)]
            relevant, overlap = _result_is_relevant(query, unit, iou_threshold)
            if relevant:
                rank = float(position)
                first_iou = overlap
                first_abs_error = abs(unit.start_time - query.start_time)
                break

        ranks.append(rank)
        first_ious.append(first_iou)
        top1_ious.append(top1_iou)
        first_abs_errors.append(first_abs_error)
        top1_abs_errors.append(top1_abs_error)

        per_query_rows.append(
            {
                "variant": variant.name,
                "display_name": variant.display_name,
                "method": variant.method,
                "evidence_mode": variant.evidence_mode,
                "query_id": query.query_id,
                "query": query.query,
                "query_type": query.query_type,
                "target_video": query.video,
                "target_start_time": query.start_time,
                "target_end_time": query.end_time,
                "target_segment_index": query.segment_index,
                "rank": "" if math.isinf(rank) else int(rank),
                "found": not math.isinf(rank),
                "first_relevant_iou": _round4(first_iou),
                "first_relevant_abs_error_s": "" if math.isinf(first_abs_error) else round(first_abs_error, 3),
                "top1_video": top1_unit.video if top1_unit else "",
                "top1_segment_index": top1_unit.segment_index if top1_unit else "",
                "top1_start_time": top1_unit.start_time if top1_unit else "",
                "top1_end_time": top1_unit.end_time if top1_unit else "",
                "top1_score": _round4(top1_score),
                "top1_iou": _round4(top1_iou),
                "top1_abs_error_s": "" if math.isinf(top1_abs_error) else round(top1_abs_error, 3),
                "query_latency_ms": round(latency_ms, 3),
            }
        )

        for top_pos, unit_idx in enumerate(order[:eval_depth], start=1):
            unit = units[int(unit_idx)]
            relevant, overlap = _result_is_relevant(query, unit, iou_threshold)
            topk_rows.append(
                {
                    "variant": variant.name,
                    "query_id": query.query_id,
                    "rank": top_pos,
                    "score": _round4(float(scores[int(unit_idx)])),
                    "relevant": relevant,
                    "iou": _round4(overlap),
                    "video": unit.video,
                    "segment_index": unit.segment_index,
                    "scene_id": unit.scene_id,
                    "start_time": unit.start_time,
                    "end_time": unit.end_time,
                    "transcript": unit.transcript,
                    "ocr": unit.ocr,
                    "caption": unit.caption,
                }
            )

    summary = summarize_variant(
        variant=variant,
        units=units,
        all_units=all_units,
        queries=queries,
        ranks=ranks,
        first_ious=first_ious,
        top1_ious=top1_ious,
        first_abs_errors=first_abs_errors,
        top1_abs_errors=top1_abs_errors,
        query_latencies_ms=query_latencies_ms,
        k_values=k_values,
        eval_depth=eval_depth,
        build_s=build_s,
        corpus_encode_s=corpus_encode_s,
        query_encode_s=query_encode_s,
        corpus_cache_hit=corpus_cache_hit,
        query_cache_hit=query_cache_hit,
    )
    return summary, per_query_rows, topk_rows


def summarize_variant(
    variant: Variant,
    units: Sequence[EvidenceUnit],
    all_units: Sequence[EvidenceUnit],
    queries: Sequence[EvalQuery],
    ranks: Sequence[float],
    first_ious: Sequence[float],
    top1_ious: Sequence[float],
    first_abs_errors: Sequence[float],
    top1_abs_errors: Sequence[float],
    query_latencies_ms: Sequence[float],
    k_values: Sequence[int],
    eval_depth: int,
    build_s: float,
    corpus_encode_s: float,
    query_encode_s: float,
    corpus_cache_hit: bool,
    query_cache_hit: bool,
) -> Dict[str, object]:
    n = len(queries)
    finite_ranks = [r for r in ranks if not math.isinf(r)]
    row: Dict[str, object] = {
        "variant": variant.name,
        "display_name": variant.display_name,
        "group": variant.group,
        "method": variant.method,
        "evidence_mode": variant.evidence_mode,
        "dense_model": variant.dense_model,
        "dense_weight": variant.dense_weight,
        "bm25_weight": variant.bm25_weight,
        "description": variant.description,
        "queries": n,
        "corpus_units": len(units),
        "all_processed_units": len(all_units),
        "coverage_pct": _round4(100.0 * len(units) / max(1, len(all_units))),
        "build_s": round(build_s, 3),
        "corpus_encode_s": round(corpus_encode_s, 3),
        "query_encode_s": round(query_encode_s, 3),
        "corpus_cache_hit": corpus_cache_hit,
        "query_cache_hit": query_cache_hit,
        "mrr": _round4(sum((1.0 / r) for r in finite_ranks) / n if n else 0.0),
        "mean_rank": _round4(float(np.mean(finite_ranks)) if finite_ranks else math.inf),
        "median_rank": _round4(float(np.median(finite_ranks)) if finite_ranks else math.inf),
        "not_found": sum(1 for r in ranks if math.isinf(r)),
        "avg_first_relevant_iou": _round4(float(np.mean(first_ious)) if first_ious else 0.0),
        "avg_top1_iou": _round4(float(np.mean(top1_ious)) if top1_ious else 0.0),
        "median_first_relevant_abs_error_s": _round4(_finite_median(first_abs_errors)),
        "median_top1_abs_error_s": _round4(_finite_median(top1_abs_errors)),
        "mean_query_latency_ms": _round4(float(np.mean(query_latencies_ms)) if query_latencies_ms else 0.0),
        "p50_query_latency_ms": _round4(float(np.percentile(query_latencies_ms, 50)) if query_latencies_ms else 0.0),
        "p95_query_latency_ms": _round4(float(np.percentile(query_latencies_ms, 95)) if query_latencies_ms else 0.0),
    }

    for k in k_values:
        hits = sum(1 for r in ranks if r <= k)
        precision_values = [(1.0 / k if r <= k else 0.0) for r in ranks]
        recall = hits / n if n else 0.0
        precision = sum(precision_values) / n if n else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        map_k = sum((1.0 / r if r <= k else 0.0) for r in ranks if not math.isinf(r)) / n if n else 0.0
        ndcg_k = sum((1.0 / math.log2(r + 1) if r <= k else 0.0) for r in ranks if not math.isinf(r)) / n if n else 0.0

        row[f"hit@{k}"] = _round4(recall)
        row[f"recall@{k}"] = _round4(recall)
        row[f"precision@{k}"] = _round4(precision)
        row[f"f1@{k}"] = _round4(f1)
        row[f"map@{k}"] = _round4(map_k)
        row[f"ndcg@{k}"] = _round4(ndcg_k)

    row[f"map@{eval_depth}"] = row.get(f"map@{eval_depth}", _round4(0.0))
    row[f"ndcg@{eval_depth}"] = row.get(f"ndcg@{eval_depth}", _round4(0.0))
    return row


def _finite_median(values: Sequence[float]) -> float:
    finite = [v for v in values if not math.isinf(v) and np.isfinite(v)]
    return float(np.median(finite)) if finite else math.inf


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = list(fieldnames or rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_latex_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    columns = [
        ("display_name", "Configuration"),
        ("method", "Method"),
        ("evidence_mode", "Evidence"),
        ("recall@1", "Recall@1"),
        ("recall@5", "Recall@5"),
        ("recall@10", "Recall@10"),
        ("precision@10", "Precision@10"),
        ("mrr", "MRR"),
        ("ndcg@10", "nDCG@10"),
        ("p95_query_latency_ms", "P95 ms"),
    ]
    lines = [
        r"\begin{table}[H]",
        r"\caption{ATLAS retrieval variant evaluation.}",
        r"\label{tab:retrieval_variant_evaluation}",
        r"\centering",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lllrrrrrrr}",
        r"\toprule",
        " & ".join(_latex_escape(label) for _, label in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.4f}" if not key.endswith("_ms") else f"{value:.1f}"
            values.append(_latex_escape(value))
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_research_question_notes(path: Path, rows: Sequence[Dict[str, object]], config: Dict[str, object]) -> None:
    if rows:
        best_recall = max(rows, key=lambda r: float(r.get("recall@10", 0) or 0))
        best_mrr = max(rows, key=lambda r: float(r.get("mrr", 0) or 0))
        fastest = min(rows, key=lambda r: float(r.get("p95_query_latency_ms", math.inf) or math.inf))
    else:
        best_recall = best_mrr = fastest = {}

    lines = [
        "# Retrieval Evaluation Evidence for Research Questions",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Queries: {config.get('queries')}",
        f"Processed units: {config.get('processed_units')}",
        "",
        "## RQ1: What multimodal fusion strategies best align text queries to timestamped video segments?",
        "",
        f"- Best Recall@10 in this run: `{best_recall.get('display_name', '')}` with `{best_recall.get('recall@10', '')}`.",
        f"- Best MRR in this run: `{best_mrr.get('display_name', '')}` with `{best_mrr.get('mrr', '')}`.",
        "- Compare `Transcript only`, `Transcript + OCR`, `Transcript + captions`, and `Full multimodal ATLAS` rows to isolate the contribution of each evidence source.",
        "",
        "## RQ2: How should we structure a persisted video index for fast retrieval and future reuse?",
        "",
        "- The corpus rows in `retrieval_topk_results.csv` expose the persisted unit shape needed for reuse: video id/name, segment index, scene id, timestamp span, transcript, OCR, caption, labels, and scores.",
        "- The summary table reports `corpus_units`, `coverage_pct`, `build_s`, and embedding-cache hits; use these to motivate precomputed text/vector fields and metadata indexes.",
        "",
        "## RQ3: Which retrieval setup optimizes Precision/Recall@k and MRR under realistic latency constraints?",
        "",
        f"- Fastest P95 query latency in this run: `{fastest.get('display_name', '')}` with `{fastest.get('p95_query_latency_ms', '')}` ms.",
        "- Use `precision@k`, `recall@k`, `mrr`, `ndcg@10`, and `p95_query_latency_ms` together; high recall with unacceptable latency is a different trade-off than a lower-recall lexical baseline.",
        "",
        "## RQ4: What are the trade-offs between on-the-fly processing and precomputed, database-backed indices?",
        "",
        "- `build_s` and `corpus_encode_s` approximate index/precompute cost. `mean_query_latency_ms` and `p95_query_latency_ms` approximate online retrieval cost once evidence and embeddings exist.",
        "- Dense and LoRA variants benefit strongly from cached/precomputed corpus embeddings; BM25 has cheaper build cost but may underperform semantic matching.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def select_variants(methods: str, include_lora: bool) -> List[Variant]:
    methods = (methods or "all").lower()
    if methods == "lexical":
        return lexical_variants()
    if methods == "core":
        return [v for v in default_variants(include_lora=False) if v.name in {"bm25_only", "dense_only", "hybrid"}]
    if methods == "modalities":
        return [v for v in default_variants(include_lora=False) if v.group == "modality_ablation"]
    return default_variants(include_lora=include_lora)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ATLAS retrieval variants offline")
    parser.add_argument("--processed-dir", default="processed", help="Processed output directory")
    parser.add_argument(
        "--processed-source",
        default="auto",
        choices=["auto", "results", "whisper", "all"],
        help="Which processed result layout to read. auto prefers processed/results.",
    )
    parser.add_argument("--dataset", default="training/retrieval_dataset.json", help="Retrieval dataset JSON")
    parser.add_argument("--judgments", default="training/relevance_judgments.csv", help="Relevance judgments CSV")
    parser.add_argument("--output-dir", default=None, help="Output directory for tables")
    parser.add_argument("--cache-dir", default="training/retrieval_eval_cache", help="Embedding cache directory")
    parser.add_argument(
        "--methods",
        default="all",
        choices=["all", "core", "modalities", "lexical"],
        help="Variant suite to run. lexical avoids dense model loading.",
    )
    parser.add_argument("--include-lora", action="store_true", help="Run Full ATLAS + LoRA variant")
    parser.add_argument("--base-model", default="Qwen/Qwen3-Embedding-0.6B", help="Base dense embedding model")
    parser.add_argument("--lora-model", default="training/checkpoints", help="Local LoRA/full SentenceTransformer checkpoint")
    parser.add_argument("--batch-size", type=int, default=16, help="Dense encoding batch size")
    parser.add_argument("--device", default=None, help="SentenceTransformer device override, e.g. cuda or cpu")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit queries for smoke tests")
    parser.add_argument("--max-corpus-units", type=int, default=None, help="Limit corpus units for smoke tests")
    parser.add_argument("--eval-depth", type=int, default=DEFAULT_EVAL_DEPTH, help="Number of ranked rows to save per query")
    parser.add_argument("--iou-threshold", type=float, default=0.10, help="Timestamp IoU threshold for a relevant hit")
    parser.add_argument("--k-values", default="1,3,5,10", help="Comma-separated k values")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = (PROJECT_ROOT / args.processed_dir).resolve()
    dataset_path = (PROJECT_ROOT / args.dataset).resolve()
    judgments_path = (PROJECT_ROOT / args.judgments).resolve() if args.judgments else None

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "processed" / "evaluation"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = tuple(int(k.strip()) for k in args.k_values.split(",") if k.strip())
    eval_depth = max(args.eval_depth, max(k_values))

    units = load_units_from_processed(processed_dir, args.processed_source)
    queries = load_eval_queries(dataset_path, judgments_path)
    if args.max_queries:
        queries = queries[: args.max_queries]
    if args.max_corpus_units:
        target_keys = {(q.video, q.segment_index) for q in queries if q.segment_index is not None}
        target_units = [
            unit
            for unit in units
            if (unit.video, unit.segment_index) in target_keys
        ]
        remaining = [unit for unit in units if unit not in target_units]
        units = (target_units + remaining)[: args.max_corpus_units]

    variants = select_variants(args.methods, args.include_lora)
    needs_dense = any(v.method in {"dense", "hybrid"} for v in variants)
    embedding_provider = (
        EmbeddingProvider(
            cache_dir=(PROJECT_ROOT / args.cache_dir).resolve(),
            base_model=args.base_model,
            lora_model=args.lora_model,
            batch_size=args.batch_size,
            device=args.device,
        )
        if needs_dense
        else None
    )

    print("ATLAS retrieval variant evaluation")
    print(f"  processed units: {len(units)}")
    print(f"  queries: {len(queries)}")
    print(f"  variants: {', '.join(v.name for v in variants)}")
    print(f"  output: {output_dir}")

    summaries: List[Dict[str, object]] = []
    per_query_rows: List[Dict[str, object]] = []
    topk_rows: List[Dict[str, object]] = []

    for variant in variants:
        summary, query_rows, ranked_rows = evaluate_variant(
            variant=variant,
            all_units=units,
            queries=queries,
            embedding_provider=embedding_provider,
            k_values=k_values,
            eval_depth=eval_depth,
            iou_threshold=args.iou_threshold,
        )
        summaries.append(summary)
        per_query_rows.extend(query_rows)
        topk_rows.extend(ranked_rows)

    summary_fields = list(summaries[0].keys()) if summaries else []
    write_csv(output_dir / "retrieval_variant_summary.csv", summaries, summary_fields)
    write_csv(output_dir / "retrieval_per_query.csv", per_query_rows)
    write_csv(output_dir / "retrieval_topk_results.csv", topk_rows)
    write_latex_summary(output_dir / "retrieval_variant_summary.tex", summaries)

    config = {
        "generated_at": datetime.now().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "processed_dir": str(processed_dir),
        "processed_source": args.processed_source,
        "dataset": str(dataset_path),
        "judgments": str(judgments_path) if judgments_path else None,
        "output_dir": str(output_dir),
        "queries": len(queries),
        "processed_units": len(units),
        "max_queries": args.max_queries,
        "max_corpus_units": args.max_corpus_units,
        "variants": [asdict(v) for v in variants],
        "k_values": k_values,
        "eval_depth": eval_depth,
        "iou_threshold": args.iou_threshold,
        "base_model": args.base_model,
        "lora_model": args.lora_model,
        "include_lora": args.include_lora,
    }
    with open(output_dir / "retrieval_run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    write_research_question_notes(output_dir / "research_question_evidence.md", summaries, config)

    print("\nDone.")
    print(f"  Summary CSV: {output_dir / 'retrieval_variant_summary.csv'}")
    print(f"  LaTeX table: {output_dir / 'retrieval_variant_summary.tex'}")
    print(f"  Per-query:   {output_dir / 'retrieval_per_query.csv'}")


if __name__ == "__main__":
    main()
