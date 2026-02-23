"""Ingest processed video data into the database."""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Dict, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session

from database.config import SessionLocal
from database.models import Video, Scene, TranscriptSegment, Embedding, VisualEmbedding
from embeddings.text_embeddings import get_embedding_generator
from embeddings.vision_embeddings import get_vision_embedding_generator


class DataIngester:
    """Ingest processed video data into database."""

    def __init__(self, db: Optional[Session] = None):
        """
        Initialize data ingester.

        Args:
            db: Database session (creates new if not provided)
        """
        self.db = db or SessionLocal()
        self.own_session = db is None
        self.embedding_gen = get_embedding_generator()
        self.vision_gen = None  # Lazy load

    def _get_vision_gen(self):
        if self.vision_gen is None:
            self.vision_gen = get_vision_embedding_generator()
        return self.vision_gen

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.own_session:
            self.db.close()

    def ingest_video(
        self,
        results_file: Path,
        generate_embeddings: bool = True,
        generate_visual_embeddings: bool = True,
        skip_existing: bool = True,
        update_existing: bool = False,
    ) -> Dict:
        """
        Ingest a single video's results into database.

        Args:
            results_file: Path to results.json (e.g., processed/results/AkerBP_1/results.json)
            generate_embeddings: Whether to generate embeddings for transcript segments
            generate_visual_embeddings: Whether to generate visual embeddings for keyframes
            skip_existing: Skip if video already exists in database
            update_existing: If True, checks if file is newer than DB record and updates if so.

        Returns:
            Dict with ingestion statistics
        """
        results_file = Path(results_file)

        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")

        # Load results
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        video_info = results["video"]
        video_name = video_info["filename"]

        # Simple fingerprint, using mtime_iso as a proxy for now
        video_fingerprint_val = video_info.get("mtime_iso")

        # Check existing
        existing_video = (
            self.db.query(Video).filter(Video.filename == video_name).first()
        )

        if existing_video:
            # ── Compare video fingerprints (content-based, not filesystem mtime) ──
            should_update = False
            reason = "no_changes"

            if update_existing:
                new_fingerprint = video_info.get("mtime_iso")
                db_fingerprint = existing_video.video_fingerprint

                if new_fingerprint != db_fingerprint:
                    # Video file itself has changed → full re-ingest
                    should_update = True
                    reason = "video_changed"
                    print(f"Video content changed for {video_name} "
                          f"(DB: {db_fingerprint} → New: {new_fingerprint})")
                else:
                    # Fingerprint matches – check if embeddings are present
                    has_text_embs = (
                        self.db.query(Embedding)
                        .join(TranscriptSegment)
                        .filter(TranscriptSegment.video_id == existing_video.id)
                        .first() is not None
                    )
                    has_visual_embs = (
                        self.db.query(VisualEmbedding)
                        .join(Scene)
                        .filter(Scene.video_id == existing_video.id)
                        .first() is not None
                    )

                    need_text = generate_embeddings and not has_text_embs
                    need_visual = generate_visual_embeddings and not has_visual_embs

                    if need_text or need_visual:
                        # Fill only the missing embeddings, don't delete anything
                        print(f"Filling missing embeddings for {video_name} "
                              f"(text={need_text}, visual={need_visual})")
                        filled = self._fill_missing_embeddings(
                            existing_video, need_text, need_visual
                        )
                        return {
                            "video": video_name,
                            "status": "filled_embeddings",
                            "video_id": existing_video.id,
                            **filled,
                        }
                    else:
                        reason = "up_to_date"

            if should_update:
                print(f"Updating existing video: {video_name} (Replacing record)")
                self.db.delete(existing_video)
                self.db.commit()  # Commit deletion to ensure clean slate
            elif skip_existing:
                if update_existing and reason == "up_to_date":
                    print(f"  ✓ Video up to date: {video_name}")
                else:
                    print(f"  ✓ Video already in database: {video_name}")

                return {
                    "video": video_name,
                    "status": "skipped",
                    "video_id": existing_video.id,
                    "reason": reason,
                }
            else:
                # Not updating and not skipping explicitly -> Skip
                print(f"  ✓ Video already in database: {video_name} (Skipping)")
                return {
                    "video": video_name,
                    "status": "skipped",
                    "video_id": existing_video.id,
                    "reason": "already_exists",
                }


        print(f"\n{'='*60}")
        print(f"Ingesting: {video_name}")
        print(f"{'='*60}")

        # Create video record
        video = Video(
            filename=video_name,
            file_path=video_info["path"],
            file_size_mb=video_info.get("size_mb"),
            duration_seconds=results["scene_analysis"].get("total_duration"),
            whisper_model=results["processing_info"].get("whisper_model"),
            scene_threshold=results["processing_info"].get("scene_threshold"),
            video_fingerprint=video_fingerprint_val,
            processed_at=datetime.fromisoformat(video_info.get("mtime_iso"))
            if video_info.get("mtime_iso")
            else None,
        )

        self.db.add(video)
        self.db.flush()  # Get video.id

        print(f"Video record created (ID: {video.id})")

        # Ingest scenes
        scenes_data = results["scene_analysis"]["scenes"]
        scene_db_objects = []

        for scene_data in scenes_data:
            scene = Scene(
                video_id=video.id,
                scene_id=scene_data["scene_id"],
                start_time=scene_data["start_time"],
                end_time=scene_data["end_time"],
                duration=scene_data["duration"],
                start_frame=scene_data.get("start_frame"),
                end_frame=scene_data.get("end_frame"),
                keyframe_path=scene_data.get("keyframe_path"),
                ocr_text=scene_data.get("ocr_text"),
                object_labels=scene_data.get("object_labels", []),
                caption=scene_data.get("caption"),
            )
            self.db.add(scene)
            scene_db_objects.append(scene)

        self.db.flush()
        print(f"{len(scenes_data)} scenes ingested")

        # Ingest transcript segments
        segments_data = results["transcription"]["segments"]
        transcript_segments = []

        for idx, seg_data in enumerate(segments_data):
            # Find corresponding scene (if any)
            scene_db_id = None
            for s_db in scene_db_objects:
                if (
                    seg_data["start"] >= s_db.start_time
                    and seg_data["start"] <= s_db.end_time
                ):
                    scene_db_id = s_db.id
                    break
            
            segment = TranscriptSegment(
                video_id=video.id,
                scene_id=scene_db_id,
                segment_index=idx,
                start_time=seg_data["start"],
                end_time=seg_data["end"],
                text=seg_data["text"],
                language=results["transcription"].get("language", "en"),
            )

            self.db.add(segment)
            transcript_segments.append(segment)

        self.db.flush()
        print(f"{len(transcript_segments)} transcript segments ingested")

        # Text Embeddings
        if generate_embeddings and transcript_segments:
            print("Generating text embeddings (batch mode)...")
            texts = [seg.text for seg in transcript_segments]
            embeddings = self.embedding_gen.encode(
                texts, 
                batch_size=32,
                show_progress=True,
            )

            for segment, embedding in zip(transcript_segments, embeddings):
                emb = Embedding(
                    segment_id=segment.id,
                    embedding=embedding.tolist(),
                    embedding_model=self.embedding_gen.model_name,
                )
                self.db.add(emb)

            print(f"✓ {len(embeddings)} text embeddings generated")

        # Visual Embeddings
        visual_count = 0
        if generate_visual_embeddings:
            visual_count = self.ingest_visual_embeddings(scene_db_objects)

        # Commit all changes
        self.db.commit()
        print(f"Successfully ingested: {video_name}\n")

        return {
            "video": video_name,
            "status": "success",
            "video_id": video.id,
            "scenes_count": len(scenes_data),
            "segments_count": len(transcript_segments),
            "text_embeddings": generate_embeddings,
            "visual_embeddings": visual_count,
        }

    def _fill_missing_embeddings(
        self, video: Video, need_text: bool, need_visual: bool
    ) -> Dict:
        """
        Generate only the missing embeddings for an existing video record.
        Does NOT delete/re-create the video, scenes, or segments.
        """
        result = {"text_embeddings_added": 0, "visual_embeddings_added": 0}

        if need_text:
            # Get segments that don't have embeddings yet
            segments = (
                self.db.query(TranscriptSegment)
                .filter(TranscriptSegment.video_id == video.id)
                .all()
            )
            segments_without_emb = [
                seg for seg in segments
                if not self.db.query(Embedding)
                    .filter(Embedding.segment_id == seg.id)
                    .first()
            ]

            if segments_without_emb:
                print(f"  Generating text embeddings for {len(segments_without_emb)} segments...")
                texts = [seg.text for seg in segments_without_emb]
                embeddings = self.embedding_gen.encode(
                    texts, batch_size=32, show_progress=True
                )
                for seg, emb_vec in zip(segments_without_emb, embeddings):
                    emb = Embedding(
                        segment_id=seg.id,
                        embedding=emb_vec.tolist(),
                        embedding_model=self.embedding_gen.model_name,
                    )
                    self.db.add(emb)
                result["text_embeddings_added"] = len(segments_without_emb)
                print(f"  ✓ {len(segments_without_emb)} text embeddings generated")

        if need_visual:
            scenes = (
                self.db.query(Scene)
                .filter(Scene.video_id == video.id)
                .all()
            )
            scenes_without_emb = [
                s for s in scenes
                if s.keyframe_path
                and not self.db.query(VisualEmbedding)
                    .filter(VisualEmbedding.scene_id == s.id)
                    .first()
            ]

            if scenes_without_emb:
                count = self.ingest_visual_embeddings(scenes_without_emb)
                result["visual_embeddings_added"] = count

        self.db.commit()
        print(f"  ✓ Missing embeddings filled for {video.filename}")
        return result

    def ingest_visual_embeddings(self, scenes: List[Scene], batch_size: int = 32) -> int:
        """Generate and store visual embeddings for a list of scenes."""
        scenes_with_keyframes = [s for s in scenes if s.keyframe_path]
        if not scenes_with_keyframes:
            return 0

        print(f"Generating visual embeddings for {len(scenes_with_keyframes)} scenes...")
        vision_gen = self._get_vision_gen()
        
        embeddings_created = 0
        
        for i in range(0, len(scenes_with_keyframes), batch_size):
            batch = scenes_with_keyframes[i:i+batch_size]
            batch_paths = []
            valid_scenes = []
            
            for scene in batch:
                full_path = Path(scene.keyframe_path)
                if not full_path.is_absolute():
                    full_path = Path.cwd() / scene.keyframe_path
                
                if full_path.exists():
                    batch_paths.append(str(full_path))
                    valid_scenes.append(scene)
                else:
                    print(f"Warning: Keyframe not found: {scene.keyframe_path}")

            if not batch_paths:
                continue

            try:
                embeddings = vision_gen.encode_images(
                    batch_paths,
                    batch_size=len(batch_paths),
                    show_progress=False,
                    normalize=True
                )
                
                for scene, embedding in zip(valid_scenes, embeddings):
                    visual_emb = VisualEmbedding(
                        scene_id=scene.id,
                        keyframe_path=scene.keyframe_path,
                        embedding=embedding.tolist(),
                        embedding_model=vision_gen.model_name
                    )
                    self.db.add(visual_emb)
                    embeddings_created += 1
                
                # Flush batches to database
                self.db.flush()
                
            except Exception as e:
                print(f"Error processing visual batch: {e}")
                continue
                
        print(f"✓ {embeddings_created} visual embeddings generated")
        return embeddings_created

    def ingest_batch(
        self,
        processed_dir: str = "processed",
        generate_embeddings: bool = True,
        skip_existing: bool = True,
        update_existing: bool = True, # Default to True to handle updates
        force: bool = False,
    ) -> Dict:
        """
        Batch ingest all processed videos.
        """
        processed_dir = Path(processed_dir)
        results_dir = processed_dir / "results"

        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        # Find all results.json files
        results_files = list(results_dir.glob("*/results.json"))

        print(f"\n{'='*60}")
        print(f"BATCH INGESTION")
        print(f"{'='*60}")
        print(f"Found {len(results_files)} videos to ingest")
        print(f"{'='*60}\n")

        stats = {
            "total": len(results_files),
            "success": 0,
            "skipped": 0,
            "failed": 0,
            "failed_videos": [],
        }

        for i, results_file in enumerate(results_files, 1):
            print(f"[{i}/{len(results_files)}]")
            try:
                result = self.ingest_video(
                    results_file,
                    generate_embeddings=generate_embeddings,
                    skip_existing=not force,
                    update_existing=update_existing or force,
                )

                if result["status"] in ("success", "filled_embeddings"):
                    stats["success"] += 1
                elif result["status"] == "skipped":
                    stats["skipped"] += 1
                else:
                    stats["failed"] += 1
                    stats["failed_videos"].append(result["video"])

            except Exception as e:
                self.db.rollback()
                print(f"Failed to ingest {results_file.parent.name}: {e}")
                import traceback
                traceback.print_exc()
                stats["failed"] += 1
                stats["failed_videos"].append(
                    {"video": results_file.parent.name, "error": str(e)}
                )

        print("\n" + "=" * 60)
        print("BATCH INGESTION COMPLETE")
        print("=" * 60)
        print(f"Total: {stats['total']}")
        print(f"Success: {stats['success']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"Failed: {stats['failed']}")
        if stats["failed_videos"]:
            print("\nFailed videos:")
            for fail in stats["failed_videos"]:
                # The structure of failed_videos might be a dict or just the name,
                # depending on where the error occurred.
                if isinstance(fail, dict):
                    print(f"  - {fail['video']}: {fail.get('error', 'Unknown error')}")
                else:
                    print(f"  - {fail}: Unknown error")

        return stats


if __name__ == "__main__":
    from database.config import test_connection, init_db

    # Test database connection
    if not test_connection():
        print("Please configure your database connection in .env file")
        exit(1)

    # Initialize database
    init_db()

    # Ingest all processed videos
    with DataIngester() as ingester:
        stats = ingester.ingest_batch(
            processed_dir="processed",
            generate_embeddings=True,
            skip_existing=True,
        )

    print(f"\nIngestion complete: {stats['success']} videos in database")
