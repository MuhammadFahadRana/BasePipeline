"""
Auto-generate ground truth template files for videos missing ground truth.

Scans processed/transcripts/Whisper-Large-v3 for video folders,
checks ground_truth/ for existing files, and creates template JSON
files for any missing videos using the Whisper transcript as placeholder.
You can then manually correct the transcript in each _gt.json file.

Usage:
    python Validation/create_ground_truth.py
    python Validation/create_ground_truth.py --force   # overwrite existing files
"""

import json
from pathlib import Path


WHISPER_DIR = Path("processed/transcripts/Whisper-Large-v3")
OUTPUT_DIR = Path("ground_truth")


def folder_to_video_name(folder_name: str) -> str:
    """Convert folder name (underscores) back to video name (spaces)."""
    return folder_name.replace("_", " ")


def get_existing_gt_video_names() -> set[str]:
    """Collect video names that already have ground truth files."""
    existing = set()
    if not OUTPUT_DIR.exists():
        return existing
    for f in OUTPUT_DIR.glob("*.json"):
        name = f.stem
        # Handle both "VideoName_gt.json" and "VideoName.json" conventions
        if name.endswith("_gt"):
            existing.add(name[:-3])
        else:
            existing.add(name)
    return existing


def read_transcript(transcript_dir: Path) -> str:
    """Read transcript text from a Whisper transcript folder."""
    json_file = transcript_dir / "transcript.json"
    txt_file = transcript_dir / "transcript.txt"

    if json_file.exists():
        for encoding in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
            try:
                with open(json_file, "r", encoding=encoding) as f:
                    data = json.load(f)
                return data.get("text", "").strip()
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

    if txt_file.exists():
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return txt_file.read_text(encoding=encoding).strip()
            except UnicodeDecodeError:
                continue

    return ""


def create_ground_truth_template(video_name: str, transcript_text: str) -> dict:
    """Build a ground truth template JSON matching the existing format."""
    return {
        "video": video_name,
        "ground_truth_transcript": [transcript_text] if transcript_text else [""],
        "instructions": "Manually transcribe the video audio exactly as spoken, including filler words.",
        "scene_annotations": [],
        "scene_annotation_instructions": "Mark scene changes with timestamps (seconds) and scene id in the format: [scene_id] [timestamp]",
    }


def main(force: bool = False):
    if not WHISPER_DIR.exists():
        print(f"Whisper transcript directory not found: {WHISPER_DIR.absolute()}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_folders = sorted([d for d in WHISPER_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(video_folders)} videos in {WHISPER_DIR}\n")

    existing_gt = get_existing_gt_video_names()
    print(f"Found {len(existing_gt)} existing ground truth files in {OUTPUT_DIR}\n")

    created, skipped, failed = 0, 0, 0

    for folder in video_folders:
        video_name = folder_to_video_name(folder.name)
        output_file = OUTPUT_DIR / f"{video_name}_gt.json"

        if video_name in existing_gt and not force:
            print(f"  ⏭ {video_name} — ground truth exists, skipping")
            skipped += 1
            continue

        transcript_text = read_transcript(folder)
        if not transcript_text:
            print(f"  ✗ {video_name} — no transcript found in {folder.name}/")
            failed += 1
            continue

        gt = create_ground_truth_template(video_name, transcript_text)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(gt, f, indent=2, ensure_ascii=False)

        word_count = len(transcript_text.split())
        print(f"  ✓ {video_name} — {word_count} words")
        created += 1

    print(f"\nDone: {created} created, {skipped} skipped, {failed} failed")
    print(f"Ground truth files saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ground truth template files from Whisper transcripts"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    main(force=args.force)
