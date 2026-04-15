"""
Backfill scene enrichment data (caption, OCR, object labels) and embeddings.

This script repairs existing DB rows where scenes are missing enrichment text and
ensures three retrieval signals are available:
1. Scene enrichment text in `scenes` (caption/object_labels/ocr_text)
2. Scene text embeddings in `embeddings` (scene_id rows)
3. Spatiotemporal visual embeddings in `visual_embeddings` (start/mid/end roles)
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Optional, List

# Avoid Windows cp1252 crashes when helper modules print unicode symbols.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.config import SessionLocal, test_connection
from database.models import Video, Scene
from database.ingest import DataIngester
from sqlalchemy import text


def _read_results_for_video(video: Video) -> Optional[List[Dict]]:
    stem = Path(video.filename).stem
    candidates = [
        Path("processed") / "results" / stem / "results.json",
        Path("processed") / stem / "results.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            scenes = data.get("scene_analysis", {}).get("scenes", [])
            if isinstance(scenes, list):
                return scenes
        except Exception:
            continue
    return None


def _write_results_from_db(db, video: Video) -> bool:
    stem = Path(video.filename).stem
    results_file = Path("processed") / "results" / stem / "results.json"
    if not results_file.exists():
        return False

    try:
        data = json.loads(results_file.read_text(encoding="utf-8"))
    except Exception:
        return False

    scenes = data.get("scene_analysis", {}).get("scenes", [])
    if not isinstance(scenes, list):
        return False

    db_scenes = (
        db.query(Scene)
        .filter(Scene.video_id == video.id)
        .order_by(Scene.scene_id.asc())
        .all()
    )
    db_by_scene_id = {s.scene_id: s for s in db_scenes}
    changed = False

    for row in scenes:
        scene_id = row.get("scene_id")
        if scene_id not in db_by_scene_id:
            continue
        db_scene = db_by_scene_id[scene_id]
        new_caption = db_scene.caption
        new_labels = db_scene.object_labels if db_scene.object_labels is not None else []
        new_ocr = db_scene.ocr_text
        new_conf = db_scene.ocr_confidence

        if row.get("caption") != new_caption:
            row["caption"] = new_caption
            changed = True
        if row.get("object_labels") != new_labels:
            row["object_labels"] = new_labels
            changed = True
        if row.get("ocr_text") != new_ocr:
            row["ocr_text"] = new_ocr
            changed = True
        if row.get("ocr_confidence") != new_conf:
            row["ocr_confidence"] = new_conf
            changed = True

    if not changed:
        return False

    results_file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    scenes_cache = Path("processed") / "scenes" / stem / f"{stem}_scenes.json"
    if scenes_cache.exists():
        scenes_cache.write_text(
            json.dumps(scenes, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return True


def _snapshot_scene_signal_stats(db) -> Dict[str, int]:
    row = db.execute(
        text(
            """
            SELECT
                (SELECT COUNT(*) FROM scenes) AS scenes_total,
                (SELECT COUNT(*) FROM scenes
                 WHERE caption IS NOT NULL
                   AND BTRIM(caption) <> ''
                   AND LOWER(BTRIM(caption)) NOT IN ('none', 'null', 'n/a', 'na')
                ) AS scenes_caption,
                (SELECT COUNT(*) FROM scenes
                 WHERE ocr_text IS NOT NULL
                   AND BTRIM(ocr_text) <> ''
                   AND LOWER(BTRIM(ocr_text)) NOT IN ('none', 'null', 'n/a', 'na')
                ) AS scenes_ocr,
                (SELECT COUNT(*) FROM scenes
                 WHERE object_labels IS NOT NULL
                   AND object_labels::text <> '[]'
                ) AS scenes_with_object_labels,
                (SELECT COUNT(*) FROM embeddings
                 WHERE scene_id IS NOT NULL AND segment_id IS NULL
                ) AS scene_embeddings,
                (SELECT COUNT(*) FROM visual_embeddings) AS visual_embeddings
            """
        )
    ).mappings().first()
    return dict(row) if row else {}


def main():
    parser = argparse.ArgumentParser(
        description="Backfill scene OCR/captions/object labels and embeddings."
    )
    parser.add_argument("--video-id", type=int, default=None, help="Repair one video only")
    parser.add_argument("--limit", type=int, default=0, help="Max videos to process (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Report only, no writes")
    parser.add_argument(
        "--no-results-sync",
        action="store_true",
        help="Skip syncing enriched fields back into processed/results/*.json",
    )
    parser.add_argument(
        "--disable-qwen",
        action="store_true",
        help="Skip Qwen caption/OCR inference and use OCR/transcript fallbacks only.",
    )
    args = parser.parse_args()

    if not test_connection():
        raise SystemExit("Database connection failed.")

    if args.disable_qwen:
        os.environ["VISUAL_ENRICHMENT_ENABLED"] = "false"
        print("Qwen visual enrichment disabled for this run (fallback mode).")

    db = SessionLocal()
    try:
        query = db.query(Video).order_by(Video.id.asc())
        if args.video_id is not None:
            query = query.filter(Video.id == args.video_id)
        videos = query.all()
        if args.limit and args.limit > 0:
            videos = videos[: args.limit]

        if not videos:
            print("No matching videos found.")
            return

        with DataIngester(db=db) as ingester:
            total = len(videos)
            repaired = 0
            total_scene_enriched = 0
            total_scene_emb_added = 0
            total_text_emb_added = 0
            total_visual_emb_added = 0
            total_segments_relinked = 0
            total_results_synced = 0
            stats_before = _snapshot_scene_signal_stats(db)

            for idx, video in enumerate(videos, start=1):
                missing_text = ingester._count_missing_transcript_embeddings(video.id)
                missing_enrichment = ingester._count_scenes_missing_enrichment(video.id)
                missing_scene_emb = ingester._count_scenes_missing_text_embeddings(video.id)
                missing_visual_scenes = ingester._scenes_missing_visual_coverage(video.id)

                if (
                    missing_text == 0
                    and missing_enrichment == 0
                    and missing_scene_emb == 0
                    and len(missing_visual_scenes) == 0
                ):
                    print(f"[{idx}/{total}] {video.filename}: up to date")
                    continue

                print(
                    f"[{idx}/{total}] {video.filename}: "
                    f"missing_text={missing_text}, "
                    f"missing_enrichment={missing_enrichment}, "
                    f"missing_scene_emb={missing_scene_emb}, "
                    f"missing_visual_scenes={len(missing_visual_scenes)}"
                )

                if args.dry_run:
                    continue

                scenes_data = _read_results_for_video(video) or []
                out = ingester._fill_missing_embeddings(
                    video=video,
                    need_text=missing_text > 0,
                    need_visual=len(missing_visual_scenes) > 0,
                    need_scene_enrichment=missing_enrichment > 0,
                    need_scene_text_embeddings=missing_scene_emb > 0,
                    scenes_data=scenes_data,
                    precomputed_missing_visual_scenes=missing_visual_scenes,
                )

                repaired += 1
                total_scene_enriched += out.get("scenes_enriched", 0)
                total_scene_emb_added += out.get("scene_text_embeddings_added", 0)
                total_text_emb_added += out.get("text_embeddings_added", 0)
                total_visual_emb_added += out.get("visual_embeddings_added", 0)
                total_segments_relinked += out.get("segments_relinked", 0)

                if not args.no_results_sync and _write_results_from_db(db, video):
                    total_results_synced += 1

            print("\nRepair summary")
            print("--------------")
            print(f"Videos scanned: {total}")
            print(f"Videos repaired: {repaired}")
            print(f"Scenes enriched: {total_scene_enriched}")
            print(f"Scene text embeddings added: {total_scene_emb_added}")
            print(f"Transcript embeddings added: {total_text_emb_added}")
            print(f"Visual embeddings added: {total_visual_emb_added}")
            print(f"Transcript segments relinked: {total_segments_relinked}")
            if not args.no_results_sync:
                print(f"results.json files synced: {total_results_synced}")

            if not args.dry_run:
                stats_after = _snapshot_scene_signal_stats(db)
                print("\nSignal coverage (before -> after)")
                print("--------------------------------")
                for key in (
                    "scenes_total",
                    "scenes_caption",
                    "scenes_ocr",
                    "scenes_with_object_labels",
                    "scene_embeddings",
                    "visual_embeddings",
                ):
                    print(f"{key}: {stats_before.get(key, 0)} -> {stats_after.get(key, 0)}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
