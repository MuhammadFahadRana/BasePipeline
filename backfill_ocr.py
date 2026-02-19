"""
Backfill OCR text on existing keyframes.

Reads all processed scene keyframes, runs EasyOCR on each,
and updates both results.json files and the database.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from scene_detector import SceneDetector, SceneConfig
from database.config import SessionLocal, test_connection
from database.models import Scene


def backfill_ocr(dry_run: bool = False):
    """Run OCR on all existing keyframes and update results + DB."""

    # Initialize scene detector (OCR only, no YOLO/CLIP needed)
    config = SceneConfig(
        enable_refinement=False,
        enable_ocr=True,
        ocr_languages=["en"],
        ocr_use_gpu=True,
        ocr_confidence_threshold=0.5,
    )
    detector = SceneDetector(config=config)

    results_dir = Path("processed/results")
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    video_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"\n{'='*60}")
    print(f"OCR BACKFILL")
    print(f"{'='*60}")
    print(f"Found {len(video_dirs)} video result directories")
    print(f"Dry run: {dry_run}")
    print(f"{'='*60}\n")

    # Connect to database
    db = None
    if not dry_run:
        if test_connection():
            db = SessionLocal()
            print("[OK] Connected to database")
        else:
            print("[!] Database not available, will only update results.json files")

    total_scenes = 0
    total_ocr_found = 0
    total_updated_db = 0

    for i, video_dir in enumerate(video_dirs, 1):
        results_file = video_dir / "results.json"
        if not results_file.exists():
            continue

        print(f"\n[{i}/{len(video_dirs)}] {video_dir.name}")

        # Load results
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                results = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  Skipping (corrupted results.json): {e}")
            continue

        scenes = results.get("scene_analysis", {}).get("scenes", [])
        if not scenes:
            print(f"  No scenes found, skipping")
            continue

        # Check how many already have OCR
        existing_ocr = sum(1 for s in scenes if s.get("ocr_text"))
        print(f"  {len(scenes)} scenes, {existing_ocr} already have OCR")

        if existing_ocr == len(scenes):
            print(f"  All scenes already have OCR, skipping")
            total_scenes += len(scenes)
            total_ocr_found += existing_ocr
            continue

        # Run OCR on scenes that don't have it yet
        ocr_added = 0
        for scene in scenes:
            if scene.get("ocr_text"):
                continue  # Already has OCR

            kf = scene.get("keyframe_path")
            if not kf:
                scene["ocr_text"] = None
                continue

            kf_path = Path(kf)
            if not kf_path.is_absolute():
                kf_path = Path.cwd() / kf

            if not kf_path.exists():
                scene["ocr_text"] = None
                continue

            if dry_run:
                print(f"  [DRY RUN] Would OCR: {kf_path.name}")
                continue

            try:
                ocr_reader = detector._ensure_ocr()
                if ocr_reader is None:
                    print("  [!] OCR reader not available")
                    break

                text = ocr_reader.extract_text(
                    str(kf_path),
                    confidence_threshold=config.ocr_confidence_threshold,
                    clean=True,
                )
                scene["ocr_text"] = text if text else None
                if text:
                    ocr_added += 1
                    print(f"  Scene {scene.get('scene_id', '?')}: \"{text[:60]}\"")
            except Exception as e:
                print(f"  [!] OCR failed for scene {scene.get('scene_id', '?')}: {e}")
                scene["ocr_text"] = None

        if not dry_run and ocr_added > 0:
            # Update results.json
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [OK] Updated results.json (+{ocr_added} OCR entries)")

            # Also update the scenes cache file
            scenes_cache = Path("processed/scenes") / video_dir.name / f"{video_dir.name}_scenes.json"
            if scenes_cache.exists():
                with open(scenes_cache, "w", encoding="utf-8") as f:
                    json.dump(scenes, f, indent=2, ensure_ascii=False)

            # Update database
            if db:
                try:
                    from database.models import Video
                    video = db.query(Video).filter(
                        Video.filename.like(f"%{video_dir.name}%")
                    ).first()

                    if video:
                        db_scenes = db.query(Scene).filter(
                            Scene.video_id == video.id
                        ).all()

                        db_updated = 0
                        for db_scene in db_scenes:
                            # Match by scene_id
                            matching = [s for s in scenes if s.get("scene_id") == db_scene.scene_id]
                            if matching and matching[0].get("ocr_text"):
                                db_scene.ocr_text = matching[0]["ocr_text"]
                                db_scene.ocr_processed_at = datetime.utcnow()
                                db_updated += 1

                        if db_updated > 0:
                            db.commit()
                            total_updated_db += db_updated
                            print(f"  [OK] Updated {db_updated} DB records")
                    else:
                        print(f"  [!] Video not found in DB for: {video_dir.name}")
                except Exception as e:
                    db.rollback()
                    print(f"  [!] DB update failed: {e}")

        total_scenes += len(scenes)
        total_ocr_found += existing_ocr + ocr_added

    # Summary
    print(f"\n{'='*60}")
    print("OCR BACKFILL COMPLETE")
    print(f"{'='*60}")
    print(f"Total scenes processed: {total_scenes}")
    print(f"Scenes with OCR text: {total_ocr_found}")
    print(f"DB records updated: {total_updated_db}")

    if db:
        db.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backfill OCR on existing keyframes")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    args = parser.parse_args()

    backfill_ocr(dry_run=args.dry_run)
