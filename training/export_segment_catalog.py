"""
export_segment_catalog.py

Export all existing transcript segments (from processed results) to a flat CSV.
Useful for manual relevance labeling without creating segments manually.
"""

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.build_training_dataset import load_segments_from_processed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export segment catalog for manual relevance judgments"
    )
    parser.add_argument(
        "--processed-dir",
        default="processed",
        help="Path to processed directory",
    )
    parser.add_argument(
        "--output",
        default="training/segment_catalog.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    segments = load_segments_from_processed(args.processed_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = sorted(segments, key=lambda s: (s.video, s.segment_index))

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "segment_index",
                "start_time",
                "end_time",
                "language",
                "scene_id",
                "text",
            ],
        )
        writer.writeheader()
        for s in rows:
            writer.writerow(
                {
                    "video": s.video,
                    "segment_index": s.segment_index,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "language": s.language,
                    "scene_id": s.scene_id,
                    "text": s.text,
                }
            )

    print(f"Wrote {len(rows)} segment rows to {output_path}")


if __name__ == "__main__":
    main()
