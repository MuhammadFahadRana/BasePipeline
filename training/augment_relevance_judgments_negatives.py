"""
augment_relevance_judgments_negatives.py

Add negative rows to relevance_judgments.csv from segment_catalog.csv.

Rules:
- Keep at least one positive row (relevance >= 2) per query_id.
- Add negatives (relevance 0/1) to reach a target count per query.
- Use video/segment/timestamps exactly from segment_catalog.csv.
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _safe_int(value: str, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _row_key(row: Dict[str, str]) -> Tuple[str, int]:
    return (row.get("video", ""), _safe_int(row.get("segment_index", "-1"), -1))


def load_catalog(catalog_path: Path) -> List[Dict[str, str]]:
    with open(catalog_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def augment_judgments(
    judgments_rows: List[Dict[str, str]],
    catalog_rows: List[Dict[str, str]],
    min_negatives: int,
    max_negatives: int,
    target_negatives: int,
    seed: int,
) -> List[Dict[str, str]]:
    by_query: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in judgments_rows:
        by_query[row["query_id"]].append(row)

    catalog_by_video: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in catalog_rows:
        catalog_by_video[row["video"]].append(row)

    all_catalog = list(catalog_rows)
    out_rows: List[Dict[str, str]] = []

    # Clamp target into [min, max]
    desired_negatives = max(min_negatives, min(max_negatives, target_negatives))

    for qid in sorted(by_query.keys()):
        qrows = by_query[qid]

        # Ensure at least one positive for this query.
        positives = [r for r in qrows if _safe_int(r.get("relevance", "0"), 0) >= 2]
        if positives:
            positive_row = max(positives, key=lambda r: _safe_int(r.get("relevance", "0"), 0))
        else:
            positive_row = qrows[0]
            positive_row["relevance"] = "3"
            notes = (positive_row.get("notes") or "").strip()
            positive_row["notes"] = (notes + "; auto_promoted_positive").strip("; ").strip()

        positive_video = positive_row.get("video", "")
        positive_key = _row_key(positive_row)

        existing_negatives = [r for r in qrows if _safe_int(r.get("relevance", "0"), 0) <= 1]
        existing_keys = {_row_key(r) for r in qrows}

        need = max(0, desired_negatives - len(existing_negatives))

        if need > 0:
            # Stable but query-specific randomness.
            local_rng = random.Random(f"{seed}:{qid}")

            same_video_candidates = [
                r for r in catalog_by_video.get(positive_video, [])
                if _row_key(r) != positive_key and _row_key(r) not in existing_keys
            ]
            other_video_candidates = [
                r for r in all_catalog
                if r.get("video", "") != positive_video and _row_key(r) not in existing_keys
            ]

            # Add one harder negative from same video when possible.
            added: List[Dict[str, str]] = []
            if need > 0 and same_video_candidates:
                c = local_rng.choice(same_video_candidates)
                row = dict(positive_row)
                row["video"] = c["video"]
                row["segment_index"] = str(_safe_int(c["segment_index"], -1))
                row["start_time"] = str(_safe_float(c["start_time"], 0.0))
                row["end_time"] = str(_safe_float(c["end_time"], 0.0))
                row["relevance"] = "1"
                row["notes"] = "auto_negative_same_video"
                added.append(row)
                existing_keys.add(_row_key(row))
                need -= 1

            # Fill remaining with negatives from other videos.
            if need > 0 and other_video_candidates:
                local_rng.shuffle(other_video_candidates)
                for c in other_video_candidates:
                    if need <= 0:
                        break
                    key = (c["video"], _safe_int(c["segment_index"], -1))
                    if key in existing_keys:
                        continue
                    row = dict(positive_row)
                    row["video"] = c["video"]
                    row["segment_index"] = str(_safe_int(c["segment_index"], -1))
                    row["start_time"] = str(_safe_float(c["start_time"], 0.0))
                    row["end_time"] = str(_safe_float(c["end_time"], 0.0))
                    row["relevance"] = "0"
                    row["notes"] = "auto_negative_other_video"
                    added.append(row)
                    existing_keys.add(key)
                    need -= 1

            qrows.extend(added)

        # Keep positives first then negatives.
        qrows.sort(key=lambda r: _safe_int(r.get("relevance", "0"), 0), reverse=True)
        out_rows.extend(qrows)

    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add negatives to relevance_judgments.csv from segment_catalog.csv"
    )
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
    parser.add_argument(
        "--output",
        default="training/relevance_judgments.csv",
        help="Output path (default: in-place update)",
    )
    parser.add_argument(
        "--min-negatives",
        type=int,
        default=2,
        help="Minimum negatives per query_id",
    )
    parser.add_argument(
        "--max-negatives",
        type=int,
        default=5,
        help="Maximum negatives per query_id",
    )
    parser.add_argument(
        "--target-negatives",
        type=int,
        default=3,
        help="Target negatives per query_id (clamped to [min, max])",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    judgments_path = Path(args.judgments)
    catalog_path = Path(args.catalog)
    output_path = Path(args.output)

    if not judgments_path.exists():
        raise FileNotFoundError(f"Judgments CSV not found: {judgments_path}")
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog CSV not found: {catalog_path}")

    with open(judgments_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        judgments_rows = list(reader)
    if not fieldnames:
        fieldnames = [
            "query_id", "query_text", "query_type", "language", "video",
            "segment_index", "start_time", "end_time", "relevance", "notes"
        ]

    catalog_rows = load_catalog(catalog_path)
    out_rows = augment_judgments(
        judgments_rows=judgments_rows,
        catalog_rows=catalog_rows,
        min_negatives=max(0, args.min_negatives),
        max_negatives=max(0, args.max_negatives),
        target_negatives=max(0, args.target_negatives),
        seed=args.seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
