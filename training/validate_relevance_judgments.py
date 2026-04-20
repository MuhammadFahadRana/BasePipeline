"""
validate_relevance_judgments.py

Validate relevance_judgments.csv quality before dataset build:
- each query_id has at least one row with relevance >= 2
- video values are stems (no file extension)
- (video, segment_index) exists in segment catalog when segment_index >= 0
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wav", ".mp3", ".flac"}


def _load_catalog(path: Path) -> Set[Tuple[str, int]]:
    pairs: Set[Tuple[str, int]] = set()
    if not path.exists():
        return pairs
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pairs.add((row["video"], int(row["segment_index"])))
            except Exception:
                continue
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate relevance_judgments.csv")
    parser.add_argument(
        "--judgments",
        default="training/relevance_judgments.csv",
        help="Path to relevance judgments CSV",
    )
    parser.add_argument(
        "--catalog",
        default="training/segment_catalog.csv",
        help="Path to segment catalog CSV",
    )
    args = parser.parse_args()

    judgments_path = Path(args.judgments)
    if not judgments_path.exists():
        raise FileNotFoundError(f"Judgments CSV not found: {judgments_path}")

    catalog_pairs = _load_catalog(Path(args.catalog))

    by_query: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    bad_video_format = 0
    missing_segments = 0
    total_rows = 0

    with open(judgments_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            by_query[row.get("query_id", "")].append(row)

            video = (row.get("video") or "").strip()
            if Path(video).suffix.lower() in VIDEO_EXTENSIONS:
                bad_video_format += 1

            seg_idx_raw = (row.get("segment_index") or "").strip()
            try:
                seg_idx = int(seg_idx_raw)
            except Exception:
                seg_idx = -1
            if seg_idx >= 0 and catalog_pairs:
                if (video, seg_idx) not in catalog_pairs:
                    missing_segments += 1

    queries_without_positive = 0
    for qid, rows in by_query.items():
        try:
            best_rel = max(int((r.get("relevance") or "0")) for r in rows)
        except Exception:
            best_rel = 0
        if best_rel < 2:
            queries_without_positive += 1

    print(f"Rows: {total_rows}")
    print(f"Unique query_id: {len(by_query)}")
    print(f"Queries missing relevance>=2: {queries_without_positive}")
    print(f"Rows with video extension (should be stem only): {bad_video_format}")
    print(f"Rows with missing (video,segment_index) in catalog: {missing_segments}")

    if queries_without_positive == 0 and bad_video_format == 0 and missing_segments == 0:
        print("Validation status: PASS")
    else:
        print("Validation status: WARN")


if __name__ == "__main__":
    main()
