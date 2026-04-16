"""Ingest processed video data into SQL Server Express."""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Dict, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text

from database.mssql_connection import SessionLocal
from database.models_sqlserver import Video, Scene, TranscriptSegment, Embedding, VisualEmbedding
from embeddings.text_embeddings import get_embedding_generator
from embeddings.vision_embeddings import get_vision_embedding_generator


class SQLDataIngester:
    """Ingest processed video data into SQL Server."""

    def __init__(self, db: Optional[Session] = None):
        """Initialize data ingester."""
        self.db = db or SessionLocal()
        self.own_session = db is None
        self.embedding_gen = get_embedding_generator()
        self.vision_gen = None
        # Ensure SQL Server schema compatibility (add columns if missing)
        try:
            self._ensure_sqlserver_schema()
        except Exception:
            # Do not fail initialization for schema check failures; ingestion will report errors
            pass

    def _ensure_sqlserver_schema(self):
        """Add missing columns to SQL Server tables if they don't exist (safe to run repeatedly)."""
        engine = None
        try:
            engine = self.db.get_bind()
        except Exception:
            engine = getattr(self.db, "bind", None)
        if engine is None:
            return

        # Check if engine is already a connection or an Engine
        from sqlalchemy.engine import Engine, Connection
        if isinstance(engine, Connection):
            conn = engine
        else:
            conn = engine.connect()
        
        try:
            # Check and add `object_labels` and `caption` on `scenes`
                for col_name, col_def in (("object_labels", "NVARCHAR(MAX) NULL"), ("caption", "NVARCHAR(MAX) NULL")):
                    q = "SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='scenes' AND COLUMN_NAME = :col"
                    res = conn.execute(text(q), {"col": col_name}).fetchone()
                    if not res:
                        alter = f"ALTER TABLE dbo.scenes ADD {col_name} {col_def};"
                        conn.execute(text(alter))
        finally:
                if isinstance(engine, Engine):
                    conn.close()

    def _get_vision_gen(self):
        if self.vision_gen is None:
            self.vision_gen = get_vision_embedding_generator()
        return self.vision_gen

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.own_session:
            self.db.close()

    def ingest_all(self, results_files, generate_embeddings=True, generate_visual_embeddings=True, skip_existing=True):
        total_videos = 0
        total_scenes = 0
        total_segments = 0
        failed_files = []
        for results_file in results_files:
            try:
                result = self.ingest_video(
                    results_file,
                    generate_embeddings=generate_embeddings,
                    generate_visual_embeddings=generate_visual_embeddings,
                    skip_existing=skip_existing,
                )
                total_videos += 1
                total_scenes += result.get("scenes_count", 0)
                total_segments += result.get("segments_count", 0)
                print(f"Ingested: {result['video']} (scenes: {result['scenes_count']}, segments: {result['segments_count']})")
            except Exception as e:
                print(f"Failed to ingest {results_file}: {e}")
                failed_files.append(str(results_file))
        print("\n--- Ingestion Summary ---")
        print(f"Videos processed: {total_videos}")
        print(f"Total scenes: {total_scenes}")
        print(f"Total segments: {total_segments}")
        if failed_files:
            print("Failed files:")
            for f in failed_files:
                print(f"  {f}")
                
    def ingest_video(
        self,
        results_file: Path,
        generate_embeddings: bool = True,
        generate_visual_embeddings: bool = True,
        skip_existing: bool = True,
    ) -> Dict:
        """Ingest a single video's results into SQL Server."""
        results_file = Path(results_file)
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")

        # Try several encodings when loading JSON files to avoid UnicodeDecodeError
        results = None
        encodings_to_try = ["utf-8", "utf-8-sig", "utf-16", "latin-1"]
        for enc in encodings_to_try:
            try:
                with open(results_file, "r", encoding=enc) as f:
                    results = json.load(f)
                break
            except UnicodeDecodeError:
                # try next encoding
                continue
            except json.JSONDecodeError as jde:
                # If file isn't valid JSON even after decoding, re-raise
                raise jde
        if results is None:
            # final attempt: read as binary and try utf-8 with errors replaced
            with open(results_file, "rb") as f:
                raw = f.read()
            try:
                text = raw.decode("utf-8", errors="replace")
                results = json.loads(text)
            except Exception as e:
                raise e
            print(f"DEBUG: Loaded results from {results_file.name}")
            print(f"DEBUG: Transcription keys: {list(results.get('transcription', {}).keys())}")
            segments = results.get('transcription', {}).get('segments', [])
            print(f"DEBUG: Segments type: {type(segments)}, length: {len(segments)}")
            if segments and len(segments) > 0:
                print(f"DEBUG: First segment type: {type(segments[0])}")
                if isinstance(segments[0], dict):
                    print(f"DEBUG: First segment keys: {list(segments[0].keys())}")

        video_info = results["video"]
        video_name = video_info["filename"]

        # Check existing
        existing_video = self.db.query(Video).filter(Video.filename == video_name).first()
        if existing_video and skip_existing:
            print(f"  ✓ Video already in database: {video_name}")
            return {"video": video_name, "status": "skipped", "video_id": existing_video.id}

        print(f"Ingesting: {video_name} into SQL Server")

        # Create video record
        # Ensure `video_fingerprint` stores valid JSON (DB constraint requires ISJSON=1)
        raw_fprint = video_info.get("mtime_iso")
        if raw_fprint is None:
            video_fingerprint = None
        elif isinstance(raw_fprint, (dict, list)):
            video_fingerprint = raw_fprint
        else:
            # store as an object rather than a plain string to satisfy JSON CHECK
            video_fingerprint = {"mtime_iso": raw_fprint}

        video = Video(
            filename=video_name,
            file_path=video_info["path"],
            file_size_mb=video_info.get("size_mb"),
            duration_seconds=results["scene_analysis"].get("total_duration"),
            whisper_model=results["processing_info"].get("whisper_model"),
            scene_threshold=results["processing_info"].get("scene_threshold"),
            video_fingerprint=video_fingerprint,
            processed_at=datetime.fromisoformat(video_info.get("mtime_iso"))
            if video_info.get("mtime_iso") else datetime.utcnow(),
        )

        try:
            self.db.add(video)
            self.db.flush()

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

            # Ingest transcript segments
            segments_data = results["transcription"]["segments"]
            transcript_segments = []
            
            # Debug prints removed

            # CRITICAL: Extract all data into simple local variables before comparison
            scene_lookup = []
            for scene_obj in scene_db_objects:
                try:
                    # Using getattr to avoid any potential shadowing issues with .id or .start_time
                    s_id_val = int(getattr(scene_obj, 'id'))
                    s_st_val = float(getattr(scene_obj, 'start_time'))
                    s_en_val = float(getattr(scene_obj, 'end_time'))
                    scene_lookup.append((s_id_val, s_st_val, s_en_val))
                except Exception as e:
                    print(f"DEBUG: Failed to extract data from scene object: {e}")
            
            # Debug prints removed

            def to_f(val):
                if hasattr(val, '__call__'):
                    return 0.0
                try:
                    return float(val)
                except:
                    return 0.0

            for idx, seg_data in enumerate(segments_data):
                matched_scene_id = None
                
                # Get values from dict and ensure they are floats
                current_v_st = to_f(seg_data.get("start", 0))
                current_v_en = to_f(seg_data.get("end", 0))

                for s_id, s_st, s_en in scene_lookup:
                    try:
                        # Direct comparison of float primitives with unique names
                        if current_v_st >= s_st and current_v_st <= s_en:
                            matched_scene_id = s_id
                            break
                    except Exception as e:
                        # If it still fails, we skip this comparison
                        pass

                segment = TranscriptSegment(
                    video_id=video.id,
                    scene_id=matched_scene_id,
                    segment_index=idx,
                    start_time=current_v_st,
                    end_time=current_v_en,
                    text=seg_data["text"],
                    language=results["transcription"].get("language", "en"),
                )
                self.db.add(segment)
                transcript_segments.append(segment)

            self.db.flush()

            # Text Embeddings
            if generate_embeddings and transcript_segments:
                texts = [seg.text for seg in transcript_segments]
                embeddings = self.embedding_gen.encode(texts, batch_size=32)
                for segment, embedding in zip(transcript_segments, embeddings):
                    emb = Embedding(
                        segment_id=segment.id,
                        embedding=json.dumps(embedding.tolist()),
                        embedding_model=self.embedding_gen.model_name,
                    )
                    self.db.add(emb)

            # Visual Embeddings
            visual_count = 0
            if generate_visual_embeddings:
                visual_count = self.ingest_visual_embeddings(scene_db_objects)

            self.db.commit()
            return {
                "video": video_name,
                "status": "success",
                "video_id": video.id,
                "scenes_count": len(scenes_data),
                "segments_count": len(transcript_segments),
                "visual_embeddings": visual_count,
            }
        except Exception as e:
            self.db.rollback()
            print(f"Error ingesting {results_file}: {e}")
            raise

    def ingest_visual_embeddings(self, scenes: List[Scene], batch_size: int = 32) -> int:
        """Generate and store visual embeddings for a list of scenes."""
        # Use getattr to avoid Pylance treating the attribute as a Column on the class
        scenes_with_keyframes = [s for s in scenes if getattr(s, "keyframe_path", None)]
        if not scenes_with_keyframes:
            return 0

        vision_gen = self._get_vision_gen()
        count = 0
        for i in range(0, len(scenes_with_keyframes), batch_size):
            batch = scenes_with_keyframes[i : i + batch_size]

            batch_paths = []
            valid_scenes = []
            for s in batch:
                kp = getattr(s, "keyframe_path", None)
                if not kp:
                    continue
                p = Path(kp)
                if p.exists():
                    batch_paths.append(str(p.absolute()))
                    valid_scenes.append(s)

            if not batch_paths:
                continue

            embeddings = vision_gen.encode_images(batch_paths, batch_size=len(batch_paths))
            for scene, embedding in zip(valid_scenes, embeddings):
                visual_emb = VisualEmbedding(
                    scene_id=scene.id,
                    keyframe_path=getattr(scene, "keyframe_path", None),
                    embedding=json.dumps(embedding.tolist()),
                    embedding_model=vision_gen.model_name,
                )
                self.db.add(visual_emb)
                count += 1
        return count

    def ingest_batch(self, processed_dir: str = "processed") -> Dict:
        """Batch ingest all processed videos."""
        results_dir = Path(processed_dir) / "results"
        results_files = list(results_dir.glob("*/results.json"))
        
        stats = {"total": len(results_files), "success": 0, "skipped": 0, "failed": 0}
        for results_file in results_files:
            try:
                result = self.ingest_video(results_file)
                if result["status"] == "success": stats["success"] += 1
                else: stats["skipped"] += 1
            except Exception as e:
                import traceback
                print(f"Failed to ingest {results_file}: {e}")
                traceback.print_exc()
                stats["failed"] += 1
        return stats

if __name__ == "__main__":
    with SQLDataIngester() as ingester:
        processed_path = Path("processed")
        results_dir = processed_path / "results"
        if results_dir.exists():
            results_files = list(results_dir.glob("*/results.json"))
            ingester.ingest_all(results_files)
        else:
            print(f"Results directory not found: {results_dir}")
