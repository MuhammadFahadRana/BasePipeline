"""
enrich_scenes_with_captions.py

Backfill script: generates captions and object labels for all existing
scene keyframes using Qwen2-VL-2B-Instruct (runs on RTX 4070 8 GB),
then generates and stores a semantic text embedding for each scene so
that the vector similarity search immediately benefits.

Usage:
    # Full run (all scenes without a caption):
    python enrich_scenes_with_captions.py

    # Limit to N scenes (useful for testing):
    python enrich_scenes_with_captions.py --limit 10

    # Dry-run: prints captions but does NOT write to DB:
    python enrich_scenes_with_captions.py --limit 5 --dry-run

    # Use a different model (default: Qwen/Qwen2-VL-2B-Instruct):
    python enrich_scenes_with_captions.py --model Qwen/Qwen2-VL-7B-Instruct

    # Target a specific video only:
    python enrich_scenes_with_captions.py --video "AkerBP 1.mp4"

    # Skip embedding generation (captions only):
    python enrich_scenes_with_captions.py --skip-embeddings
"""

import sys
import argparse
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from database.config import SessionLocal, test_connection
from database.models import Scene, Video, Embedding
from sqlalchemy.orm import Session
from sqlalchemy import text


# ── Default model ─────────────────────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"


def parse_args():
    p = argparse.ArgumentParser(description="Enrich scene keyframes with Qwen2-VL captions")
    p.add_argument("--model",   default=DEFAULT_MODEL, help="HuggingFace model ID")
    p.add_argument("--limit",   type=int, default=None,  help="Max scenes to process (None = all)")
    p.add_argument("--dry-run", action="store_true",     help="Print captions but do not update DB")
    p.add_argument("--video",   default=None,            help="Only process scenes from this video filename")
    p.add_argument("--batch-size", type=int, default=1,  help="Images per Qwen2-VL call (keep 1 for stability)")
    p.add_argument("--force",   action="store_true",     help="Re-run even if caption already exists")
    p.add_argument("--skip-embeddings", action="store_true",
                   help="Skip text embedding generation (captions only, no vector search indexing)")
    return p.parse_args()


def load_extractor(model_name: str):
    """Load Qwen2-VL. Uses 4-bit by default to fit in 8 GB VRAM."""
    from extract_visual_features import VisualFeatureExtractor
    return VisualFeatureExtractor(
        model_name=model_name,
        device="auto",
        load_in_4bit=True,
    )


def load_embedding_generator():
    """Load the text embedding model used by the search engine."""
    from embeddings.text_embeddings import get_embedding_generator
    return get_embedding_generator()


def get_scenes_to_process(db: Session, video_filter, force: bool, limit):
    """Query scenes that need captioning."""
    q = db.query(Scene).join(Video)

    if not force:
        # Only process scenes that don't have a caption yet
        q = q.filter((Scene.caption == None) | (Scene.caption == ""))  # noqa: E711

    if video_filter:
        q = q.filter(Video.filename == video_filter)

    # Only scenes with a keyframe on disk
    q = q.filter(Scene.keyframe_path != None)  # noqa: E711

    if limit:
        q = q.limit(limit)

    return q.all()


def update_scene(db: Session, scene: Scene, result: dict, dry_run: bool):
    """Write caption/object_labels/ocr_text back to the DB."""
    caption       = result.get("caption") or None
    object_labels = result.get("object_labels") or []
    ocr_text      = result.get("ocr_text") or None

    if dry_run:
        return

    # Use raw SQL UPDATE to avoid SQLAlchemy column mapping issues
    db.execute(
        text("""
            UPDATE scenes
            SET caption       = :caption,
                object_labels = CAST(:labels AS jsonb),
                ocr_text      = COALESCE(ocr_text, :ocr_text)
            WHERE id = :scene_id
        """),
        {
            "caption":  caption,
            "labels":   json.dumps(object_labels),
            "ocr_text": ocr_text,
            "scene_id": scene.id,
        },
    )


def upsert_caption_embedding(db: Session, scene: Scene, result: dict, emb_gen, dry_run: bool) -> bool:
    """
    Generate and upsert a text embedding from the scene's caption +
    object labels + OCR text into the embeddings table.
    Returns True if an embedding was (would be) saved.
    """
    parts = []
    caption = result.get("caption")
    if caption:
        parts.append(caption)

    labels = result.get("object_labels") or []
    if labels:
        if isinstance(labels, list):
            parts.append(" ".join(str(lbl) for lbl in labels))
        else:
            parts.append(str(labels))

    ocr_text = result.get("ocr_text")
    if ocr_text:
        parts.append(ocr_text)

    if not parts:
        return False

    if dry_run:
        return True  # Would have created one

    embed_text = " ".join(parts)
    vec = emb_gen.encode_single(embed_text)

    # Upsert: update if a scene-level embedding already exists, else insert
    existing = db.query(Embedding).filter(
        Embedding.scene_id == scene.id,
        Embedding.segment_id == None,  # noqa: E711
    ).first()

    if existing:
        existing.embedding = vec.tolist()
    else:
        db.add(Embedding(
            scene_id=scene.id,
            segment_id=None,
            embedding=vec.tolist(),
            embedding_model=emb_gen.model_name,
        ))

    return True


def main():
    args = parse_args()

    print("=" * 60)
    print("Scene Caption Backfill — Qwen2-VL")
    print("=" * 60)
    print(f"Model      : {args.model}")
    print(f"Dry-run    : {args.dry_run}")
    print(f"Limit      : {args.limit or 'all'}")
    print(f"Force      : {args.force}")
    print(f"Video      : {args.video or 'all'}")
    print(f"Embeddings : {'skip' if args.skip_embeddings else 'yes — generate + store for vector search'}")
    print("=" * 60)

    if not test_connection():
        print("✗ Database connection failed. Check .env config.")
        sys.exit(1)

    db = SessionLocal()

    try:
        scenes = get_scenes_to_process(db, args.video, args.force, args.limit)
        total = len(scenes)
        print(f"\nScenes to process: {total}")

        if total == 0:
            print("Nothing to do. All scenes already have captions.")
            print("Run with --force to re-caption existing scenes.")
            return

        # Filter to scenes whose keyframe file actually exists on disk
        valid_scenes = []
        for s in scenes:
            kf_path = Path(s.keyframe_path)
            if not kf_path.is_absolute():
                kf_path = Path.cwd() / kf_path
            if kf_path.exists():
                valid_scenes.append((s, str(kf_path)))
            else:
                print(f"  ⚠ Keyframe not found, skipping scene {s.id}: {s.keyframe_path}")

        print(f"Valid keyframes on disk: {len(valid_scenes)} / {total}")

        if not valid_scenes:
            print("No valid keyframes to process.")
            return

        if args.dry_run:
            print("\n[DRY-RUN] Loading model for test...")
        else:
            print("\nLoading Qwen2-VL model (first time may download weights ~4-8 GB)...")

        extractor = load_extractor(args.model)

        # Load embedding model only if needed
        emb_gen = None
        if not args.skip_embeddings:
            print("\nLoading text embedding model...")
            emb_gen = load_embedding_generator()
            print(f"✓ Embedding model ready: {emb_gen.model_name}")

        print(f"\nStarting enrichment of {len(valid_scenes)} scenes...\n")
        success   = 0
        failed    = 0
        emb_saved = 0
        start_time = time.time()

        for idx, (scene, kf_path) in enumerate(valid_scenes, 1):
            try:
                t0 = time.time()
                result = extractor.analyze_image(kf_path)
                elapsed = time.time() - t0

                caption  = result.get("caption", "")
                labels   = result.get("object_labels", [])
                ocr_text = result.get("ocr_text", "")

                status = "[DRY-RUN]" if args.dry_run else "✓"
                print(
                    f"[{idx}/{len(valid_scenes)}] {status} Scene {scene.id} "
                    f"(video_id={scene.video_id}) | {elapsed:.1f}s\n"
                    f"  Caption : {caption[:120]}\n"
                    f"  Labels  : {', '.join(str(l) for l in labels[:8])}\n"
                    f"  OCR     : {ocr_text[:80] if ocr_text else '(none)'}\n"
                )

                # 1. Save caption/labels/ocr_text to scenes table
                update_scene(db, scene, result, dry_run=args.dry_run)

                # 2. Generate + store semantic embedding for this caption
                if emb_gen is not None:
                    saved = upsert_caption_embedding(db, scene, result, emb_gen, dry_run=args.dry_run)
                    if saved:
                        emb_saved += 1

                if not args.dry_run and idx % 10 == 0:
                    db.commit()
                    print(f"  — committed {idx} scenes so far —")

                success += 1

            except Exception as e:
                print(f"  ✗ Scene {scene.id} failed: {e}")
                failed += 1
                try:
                    db.rollback()
                except Exception:
                    pass

        # Final commit
        if not args.dry_run:
            db.commit()

        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("ENRICHMENT COMPLETE")
        print("=" * 60)
        print(f"Success          : {success}")
        print(f"Failed           : {failed}")
        print(f"Embeddings saved : {emb_saved}")
        print(f"Time             : {total_time/60:.1f} min ({total_time/max(success, 1):.1f}s/scene avg)")
        if not args.dry_run:
            print(f"\nNext step: restart api/app.py to see improved search results.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
