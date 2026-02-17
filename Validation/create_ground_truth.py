"""
Auto-generate ground truth template files for all processed videos.

Reads each video's results.json, extracts the transcript text,
and saves a ground truth JSON file in the ground_truth/ folder.

Usage:
    python Validation/create_ground_truth.py
    python Validation/create_ground_truth.py --force   # overwrite existing files
"""

import json
from pathlib import Path


RESULTS_DIR = Path("processed/results")
OUTPUT_DIR = Path("ground_truth")
TRANSCRIPTS_DIR = OUTPUT_DIR / "transcripts"


def read_results(results_file: Path) -> dict | None:
    """Read a results.json with encoding fallbacks."""
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
        try:
            with open(results_file, "r", encoding=encoding) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    print(f"  ✗ Could not read {results_file} (encoding error)")
    return None


def create_ground_truth(video_name: str, results: dict, transcript_dir: Path) -> dict:
    """
    Build a ground truth template.
    If {video_name}.txt exists in transcript_dir, use that as transcript.
    Otherwise uses pipeline results.
    """
    txt_file = transcript_dir / f"{video_name}.txt"
    
    # 1. Create the text file if it doesn't exist (pre-fill with pipeline output)
    pipeline_text = results.get("transcription", {}).get("text", "")
    
    if not txt_file.exists():
        try:
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(pipeline_text)
            print(f"  + Created transcript file: {txt_file.name}")
        except Exception as e:
            print(f"  ! Failed to create text file: {e}")
            
    # 2. Read the text file (user might have edited it)
    if txt_file.exists():
        try:
            transcript_text = txt_file.read_text(encoding="utf-8").strip()
        except:
            transcript_text = txt_file.read_text(encoding="latin-1").strip()

    return {
        "video": video_name,
        "ground_truth_transcript": [transcript_text] if transcript_text else [""],
        "instructions": "Manually transcribe the video audio exactly as spoken, including filler words.",
        "scene_annotations": [],
        "scene_annotation_instructions": "Mark scene changes with timestamps (seconds) and scene id in the format: [scene_id] [timestamp]",
    }


def main(force: bool = False):
    if not RESULTS_DIR.exists():
        print(f"No results directory found at: {RESULTS_DIR.absolute()}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(video_dirs)} processed videos\n")

    created, skipped, failed = 0, 0, 0

    for video_dir in video_dirs:
        video_name = video_dir.name
        output_file = OUTPUT_DIR / f"{video_name}_gt.json"

        # Skip if already exists (unless --force)
        if output_file.exists() and not force:
            print(f"  ⏭ {video_name} — already exists, skipping")
            skipped += 1
            continue

        results_file = video_dir / "results.json"
        if not results_file.exists():
            print(f"  ✗ {video_name} — no results.json found")
            failed += 1
            continue

        results = read_results(results_file)
        if results is None:
            failed += 1
            continue

        gt = create_ground_truth(video_name, results, TRANSCRIPTS_DIR)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(gt, f, indent=2, ensure_ascii=False)

        word_count = len(gt["ground_truth_transcript"][0].split())
        print(f"  ✓ {video_name} — {word_count} words")
        created += 1

    print(f"\nDone: {created} created, {skipped} skipped, {failed} failed")
    print(f"Ground truth files saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ground truth files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    main(force=args.force)
