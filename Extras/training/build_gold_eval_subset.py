"""
build_gold_eval_subset.py

Create a clean gold evaluation split from manual transcript annotations.

Accepted annotation formats:
1) CSV with columns: audio_path, transcript
2) CSV with columns: video, start_time, end_time, transcript
3) JSONL with equivalent keys

Usage:
    python -m training.build_gold_eval_subset --annotations training/gold_eval.csv
    python -m training.build_gold_eval_subset --annotations training/gold_eval.csv --out-dir training/asr_data_gold
"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple


def _norm_path(p: str) -> str:
    return str(Path(p)).replace("\\", "/").lower().strip()


def _load_manifest(manifest_path: Path) -> List[Dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("samples", [])


def _load_annotations(path: Path) -> List[Dict]:
    rows: List[Dict] = []

    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
        return rows

    raise ValueError("Unsupported annotations format. Use .csv or .jsonl")


def _key_video_time(video: str, start_time: float, end_time: float) -> Tuple[str, float, float]:
    return (video.strip().lower(), round(float(start_time), 2), round(float(end_time), 2))


def build_gold_eval(
    manifest_path: str,
    annotations_path: str,
    out_dir: str,
) -> Dict:
    manifest_samples = _load_manifest(Path(manifest_path))
    annotations = _load_annotations(Path(annotations_path))

    by_audio: Dict[str, str] = {}
    by_video_time: Dict[Tuple[str, float, float], str] = {}

    for row in annotations:
        transcript = (row.get("transcript") or row.get("sentence") or "").strip()
        if not transcript:
            continue

        audio_path = row.get("audio_path") or row.get("audio")
        if audio_path:
            by_audio[_norm_path(audio_path)] = transcript
            continue

        if all(k in row for k in ("video", "start_time", "end_time")):
            key = _key_video_time(row["video"], row["start_time"], row["end_time"])
            by_video_time[key] = transcript

    gold_samples: List[Dict] = []
    unmatched = 0

    for s in manifest_samples:
        matched_text = None

        audio_key = _norm_path(s.get("audio_path", ""))
        if audio_key in by_audio:
            matched_text = by_audio[audio_key]
        else:
            key = _key_video_time(s.get("video", ""), s.get("start_time", 0), s.get("end_time", 0))
            if key in by_video_time:
                matched_text = by_video_time[key]

        if matched_text is None:
            unmatched += 1
            continue

        s2 = dict(s)
        s2["transcript"] = matched_text
        s2["split"] = "eval"
        gold_samples.append(s2)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_from": str(manifest_path),
        "annotations": str(annotations_path),
        "stats": {
            "total_manifest_samples": len(manifest_samples),
            "total_annotations": len(annotations),
            "gold_eval_samples": len(gold_samples),
            "unmatched_manifest_samples": unmatched,
        },
        "samples": gold_samples,
    }

    manifest_out = out / "manifest_gold_eval.json"
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    jsonl_out = out / "eval_gold.jsonl"
    with open(jsonl_out, "w", encoding="utf-8") as f:
        for s in gold_samples:
            f.write(json.dumps({
                "audio": s["audio_path"],
                "sentence": s["transcript"],
                "language": s.get("language", "en"),
            }, ensure_ascii=False) + "\n")

    return manifest["stats"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gold ASR eval set from manual transcripts")
    parser.add_argument("--manifest", default="training/asr_data/manifest.json", help="Source ASR manifest")
    parser.add_argument("--annotations", required=True, help="Manual annotations CSV/JSONL")
    parser.add_argument("--out-dir", default="training/asr_data_gold", help="Output directory")
    args = parser.parse_args()

    stats = build_gold_eval(args.manifest, args.annotations, args.out_dir)

    print("\nGold eval subset created")
    print(f"  Source samples: {stats['total_manifest_samples']}")
    print(f"  Manual rows:    {stats['total_annotations']}")
    print(f"  Gold eval:      {stats['gold_eval_samples']}")
    print(f"  Unmatched:      {stats['unmatched_manifest_samples']}")


if __name__ == "__main__":
    main()
