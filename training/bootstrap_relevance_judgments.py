"""
bootstrap_relevance_judgments.py

Create a starter relevance_judgments.csv from retrieval_dataset.json.

Default behavior writes one high-confidence positive row (relevance=3) per query.
This guarantees each query_id has at least one positive judgment so
`training.build_training_dataset --from-judgments` can proceed.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import re
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.build_training_dataset import load_segments_from_processed  # noqa: E402


FIELDNAMES = [
    "query_id",
    "query_text",
    "query_type",
    "language",
    "video",
    "segment_index",
    "start_time",
    "end_time",
    "relevance",
    "notes",
]


def _to_video_stem(video_value: Any) -> str:
    video = str(video_value or "").strip()
    if not video:
        return ""
    return Path(video).stem


def _norm_title(value: str) -> str:
    txt = value.lower().strip()
    txt = re.sub(r"[^a-z0-9]+", "", txt)
    return txt


def _resolve_video_name(video: str, catalog_videos: List[str]) -> str:
    if not video:
        return video
    if video in catalog_videos:
        return video

    # Prefix/contains match first (helps with truncated stems).
    prefix_matches = [
        v for v in catalog_videos if v.startswith(video) or video.startswith(v)
    ]
    if prefix_matches:
        prefix_matches.sort(key=lambda x: (abs(len(x) - len(video)), x))
        return prefix_matches[0]

    # Normalized text match (ignore punctuation/spaces).
    n_video = _norm_title(video)
    norm_matches = [v for v in catalog_videos if _norm_title(v) == n_video]
    if norm_matches:
        return sorted(norm_matches, key=lambda x: (abs(len(x) - len(video)), x))[0]

    # Loose normalized containment.
    loose_matches = [
        v
        for v in catalog_videos
        if n_video and (n_video in _norm_title(v) or _norm_title(v) in n_video)
    ]
    if loose_matches:
        loose_matches.sort(key=lambda x: (abs(len(x) - len(video)), x))
        return loose_matches[0]

    return video


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _resolve_segment_index(
    video: str,
    start_time: float,
    end_time: float,
    declared_index: int,
    video_to_segments: Dict[str, List[Tuple[int, float, float]]],
) -> int:
    """
    Resolve segment index against current processed data.
    Falls back from declared index to best timestamp match.
    """
    segs = video_to_segments.get(video, [])
    if not segs:
        return declared_index if declared_index >= 0 else -1

    if declared_index >= 0:
        for idx, s, e in segs:
            if idx == declared_index:
                return declared_index

    # 1) Exact-ish start/end match first.
    exact = [
        (idx, abs(s - start_time) + abs(e - end_time))
        for idx, s, e in segs
        if abs(s - start_time) <= 0.05 and abs(e - end_time) <= 0.05
    ]
    if exact:
        exact.sort(key=lambda x: x[1])
        return exact[0][0]

    # 2) Nearest by start (tight tolerance).
    near = [
        (idx, abs(s - start_time))
        for idx, s, _ in segs
        if abs(s - start_time) <= 0.75
    ]
    if near:
        near.sort(key=lambda x: x[1])
        return near[0][0]

    # 3) Always fall back to globally nearest segment start.
    all_near = [(idx, abs(s - start_time)) for idx, s, _ in segs]
    all_near.sort(key=lambda x: x[1])
    return all_near[0][0] if all_near else (declared_index if declared_index >= 0 else -1)


def build_rows(
    dataset: Dict[str, Any],
    video_to_segments: Optional[Dict[str, List[Tuple[int, float, float]]]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    entries = dataset.get("entries", [])
    catalog_videos = sorted((video_to_segments or {}).keys())

    for entry in entries:
        positive = entry.get("positive", {}) or {}
        video = _to_video_stem(entry.get("video", ""))
        if catalog_videos:
            video = _resolve_video_name(video, catalog_videos)
        start_time = _safe_float(positive.get("start_time", 0.0), 0.0)
        end_time = _safe_float(positive.get("end_time", 0.0), 0.0)
        declared_idx = _safe_int(positive.get("segment_index", -1), -1)
        segment_index = declared_idx
        if video_to_segments is not None:
            segment_index = _resolve_segment_index(
                video=video,
                start_time=start_time,
                end_time=end_time,
                declared_index=declared_idx,
                video_to_segments=video_to_segments,
            )

        rows.append(
            {
                "query_id": str(entry.get("query_id", "")),
                "query_text": str(entry.get("query", "")),
                "query_type": str(entry.get("query_type", "natural_search")),
                "language": str(entry.get("language", "en")),
                "video": video,
                "segment_index": segment_index,
                "start_time": start_time,
                "end_time": end_time,
                "relevance": 3,
                "notes": "auto_seed_positive",
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap training/relevance_judgments.csv from retrieval_dataset.json"
    )
    parser.add_argument(
        "--dataset",
        default="training/retrieval_dataset.json",
        help="Path to retrieval dataset JSON",
    )
    parser.add_argument(
        "--output",
        default="training/relevance_judgments.csv",
        help="Path to output judgments CSV",
    )
    parser.add_argument(
        "--processed-dir",
        default="processed",
        help="Processed directory used to resolve current segment indexes",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    segments = load_segments_from_processed(args.processed_dir)
    video_to_segments: Dict[str, List[Tuple[int, float, float]]] = {}
    for s in segments:
        video_to_segments.setdefault(s.video, []).append(
            (int(s.segment_index), float(s.start_time), float(s.end_time))
        )

    rows = build_rows(dataset, video_to_segments=video_to_segments)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
