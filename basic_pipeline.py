# ATLAS – AI-driven Temporal Linking and Search, this is the name of the project. 

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import timedelta, datetime
import time

# Import from transcribe_all for multi-model support
#from transcribe_all import SimpleTranscriber

from transcriber import SimpleTranscriber
from scene_detector import SceneDetector, SceneConfig

# Database ingestion
try:
    from database.ingest import DataIngester
    HAS_DB = True
except ImportError:
    HAS_DB = False


class BasicVideoPipeline:
    """
    Basic pipeline that:
    1. Transcribes video using Whisper
    2. Detects scenes/shots
    3. Aligns transcripts with scenes
    4. Extracts keyframes
    5. Ingests into database (new!)
    """

    def __init__(
        self,
        backend: str = "whisper",
        model_variant: dict = None,
        scene_threshold: float = 30.0,
        device: str = "auto",
        skip_ingest: bool = False,
    ):
        """
        Initialize pipeline with selected ASR backend and scene detector.

        Args:
            backend: ASR backend (whisper/whisperx/distil-whisper/wav2vec/nemo)
            model_variant: Model variant dict (e.g., {'name': 'base'})
            scene_threshold: Scene detection threshold
            device: Device for models ("auto", "cpu", "cuda")
            skip_ingest: Skip database ingestion if True
        """
        # Default to Whisper base if no variant specified
        if model_variant is None:
            model_variant = {"name": "base", "description": "Fast, good for simple audio"}

        self.transcriber = SimpleTranscriber(
            backend=backend, model_variant=model_variant, device=device
        )
        scene_cfg = SceneConfig(
            threshold=scene_threshold,
            yolo_model="yolov8n.pt",
            yolo_person_conf=0.35,
            clip_sim_merge_threshold=0.90,
            device="cuda" if device in ("auto", "cuda") else "cpu",
        )
        self.scene_detector = SceneDetector(config=scene_cfg)
        self.backend = backend
        self.model_variant = model_variant
        self.skip_ingest = skip_ingest

    # ---------------------------
    # Caching helpers (FIXED)
    # ---------------------------
    def _video_fingerprint(self, video_path: Path, use_hash: bool = False) -> Dict:
        """
        Fast mode: size + mtime
        Strict mode: also sha256 (slower for big files)
        """
        stat = video_path.stat()
        fp = {"size_bytes": stat.st_size, "mtime": stat.st_mtime}

        if use_hash:
            h = hashlib.sha256()
            with open(video_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            fp["sha256"] = h.hexdigest()

        return fp

    def _expected_outputs_exist(
        self, transcript_dir: Path, scenes_dir: Path, results_dir: Path
    ) -> bool:
        required = [
            transcript_dir / "transcript.json",
            scenes_dir / "scenes.json",
            results_dir / "results.json",
            results_dir / "report.html",
        ]
        return all(p.exists() for p in required)

    def _load_manifest(self, manifest_path: Path) -> Optional[Dict]:
        if not manifest_path.exists():
            return None
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_manifest(self, manifest_path: Path, manifest: Dict) -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ---------------------------
    # Core pipeline
    # ---------------------------
    def process_video(
        self,
        video_path: str,
        output_base: str = "processed",
        use_hash: bool = False,
        force: bool = False,
        generate_embeddings: bool = True,
    ):
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        output_base = Path(output_base)
        output_base.mkdir(parents=True, exist_ok=True)

        video_name = video_path.stem

        transcript_dir = output_base / "transcripts" / video_name
        scenes_dir = output_base / "scenes" / video_name
        results_dir = output_base / "results" / video_name
        manifest_path = results_dir / "manifest.json"

        # Cache check
        current_fp = self._video_fingerprint(video_path, use_hash=use_hash)

        # current_cfg 
        # TODO: Make this more dynamic
        current_cfg = {
            "whisper_model": getattr(self.transcriber, "model_name", "unknown"),
            "scene_threshold": self.scene_detector.config.threshold,
            "semantic_refine": False,
            "yolo_model": "yolov8n.pt",
            "yolo_person_conf": 0.35,
            "vision_model": "google/siglip-base-patch16-224",
            "text_embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "clip_sim_merge_threshold": 0.90,
        }
        manifest = self._load_manifest(manifest_path)

        cache_hit = (
            (not force)
            and (manifest is not None)
            and (manifest.get("video_fingerprint") == current_fp)
            and (manifest.get("pipeline_config") == current_cfg)
            and self._expected_outputs_exist(transcript_dir, scenes_dir, results_dir)
        )

        if cache_hit:
            results_file = results_dir / "results.json"
            print(f"\n✓ Skipping (cached): {video_name}")
            print(f"  Using existing results: {results_file}")
            
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                
                duration = results.get("processing_info", {}).get("processing_duration")
                if duration is not None:
                    is_estimated = results.get("processing_info", {}).get("duration_is_estimated", False)
                    est_str = " (estimated)" if is_estimated else ""
                    print(f"  Processing time: {duration:.2f}s{est_str}")
                
                # Still check if we need to ingest into DB if HAS_DB
                if HAS_DB and not self.skip_ingest:
                    self._ingest_results(results_file, generate_embeddings)
                
                return results
            except Exception as e:
                print(f"  ! Error reading cached results: {e}")
                return json.loads(results_file.read_text(encoding="utf-8"))

        # Otherwise process
        transcript_dir.mkdir(parents=True, exist_ok=True)
        scenes_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Processing: {video_name}")
        print(f"{'='*50}")

        start_time = time.time()

        print("\n1. Transcribing audio...")
        try:
            transcript = self.transcriber.transcribe_video(
                str(video_path), output_dir="processed"
            )
        except Exception as e:
            print(f"  ! Transcription failed: {e}")
            print("    Continuing with empty transcript.")
            transcript = {"text": "", "segments": [], "language": "unknown"}

        # Check if audio-only
        is_audio = video_path.suffix.lower() in self.scene_detector.config.audio_extensions

        if is_audio:
            print("\n2. Audio file detected - Skipping scene detection & refinement.")
            # Create synthetic scene for the whole file
            last_end = 0.0
            if transcript.get("segments"):
                last_end = transcript["segments"][-1]["end"]
            
            scenes = [{
                "scene_id": 0,
                "start_time": 0.0,
                "end_time": last_end,
                "duration": last_end,
                "keyframe_path": None,
                "ocr_text": None
            }]
        else:
            print("\n2. Detecting & refining scenes...")
            scenes = self.scene_detector.detect_scenes(
                video_path, 
                base_output_dir=str(output_base / "scenes")
            )

            print("\n2b. Refining scenes (YOLO + CLIP)...")
            try:
                scenes = self.scene_detector.refine_scenes(scenes)
            except Exception as e:
                print(f"Scene refinement failed: {e}")

            print("\n2c. Enriching scenes with OCR...")
            try:
                scenes = self.scene_detector.enrich_with_ocr(scenes)
                # Re-save scenes cache with OCR data included
                scenes_cache = output_base / "scenes" / video_path.stem / f"{video_path.stem}_scenes.json"
                if scenes_cache.exists():
                    with open(scenes_cache, "w", encoding="utf-8") as f:
                        json.dump(scenes, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"OCR enrichment failed: {e}")


        print("\n3. Aligning transcripts with scenes...")
        aligned_data = self.align_transcript_with_scenes(transcript, scenes)

        end_time = time.time()
        processing_duration = end_time - start_time

        print("\n4. Saving results...")
        results = self.save_results(
            video_path,
            transcript,
            scenes,
            aligned_data,
            results_dir=results_dir,
            transcript_dir=transcript_dir,
            scenes_dir=scenes_dir,
            processing_duration=processing_duration,
        )

        results_file = results_dir / "results.json"

        # 5. Database Ingestion
        if HAS_DB and not self.skip_ingest:
            self._ingest_results(results_file, generate_embeddings)

        # Save manifest for caching
        new_manifest = {
            "video_filename": video_path.name,
            "video_path": str(video_path),
            "video_fingerprint": current_fp,
            "pipeline_config": current_cfg,
            "saved_at_iso": datetime.now().isoformat(),
            "use_hash": use_hash,
        }
        self._save_manifest(manifest_path, new_manifest)
        print(f"✓ Manifest saved to: {manifest_path}")

        return results

    def _ingest_results(self, results_file: Path, generate_embeddings: bool = True):
        print("\n5. Ingesting into database...")
        try:
            with DataIngester() as ingester:
                ingester.ingest_video(
                    results_file,
                    generate_embeddings=generate_embeddings,
                    generate_visual_embeddings=generate_embeddings,
                    update_existing=True
                )
        except Exception as e:
            print(f"  ! Ingestion failed: {e}")

    def align_transcript_with_scenes(
        self, transcript: Dict, scenes: List[Dict]
    ) -> List[Dict]:
        transcript_segments = transcript.get("segments", [])

        for scene in scenes:
            scene_start = scene["start_time"]
            scene_end = scene["end_time"]

            scene_segments = []
            for seg in transcript_segments:
                seg_start = seg["start"]
                seg_end = seg["end"]

                if (
                    (seg_start >= scene_start and seg_start <= scene_end)
                    or (seg_end >= scene_start and seg_end <= scene_end)
                    or (seg_start <= scene_start and seg_end >= scene_end)
                ):
                    scene_segments.append(
                        {
                            "text": seg["text"],
                            "start": seg_start,
                            "end": seg_end,
                            "start_str": str(timedelta(seconds=seg_start)),
                            "end_str": str(timedelta(seconds=seg_end)),
                        }
                    )

            scene["transcript_segments"] = scene_segments

        return scenes

    def save_results(
        self,
        video_path: Path,
        transcript: Dict,
        scenes: list,
        aligned_data: list,
        results_dir: Path,
        transcript_dir: Path,
        scenes_dir: Path,
        processing_duration: float = 0.0,
    ) -> Dict:
        stat = video_path.stat()
        total_duration = sum(s.get("duration", 0) for s in scenes) if scenes else 0
        avg_scene_duration = (total_duration / len(scenes)) if scenes else 0

        results = {
            "video": {
                "filename": video_path.name,
                "path": str(video_path),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "mtime": stat.st_mtime,
                "mtime_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            },
            "transcription": {
                "language": transcript.get("language", "en"),
                "text": transcript.get("text", ""),
                "num_segments": len(transcript.get("segments", [])),
                "segments": transcript.get("segments", []),
            },
            "scene_analysis": {
                "num_scenes": len(scenes),
                "scenes": scenes,
                "total_duration": total_duration,
                "avg_scene_duration": avg_scene_duration,
            },
            "alignment": {
                "scenes_with_transcript": len(
                    [s for s in scenes if s.get("transcript_segments")]
                ),
                "aligned_scenes": aligned_data,
            },
            "processing_info": {
                "whisper_model": getattr(self.transcriber, "model_name", "unknown"),
                "scene_threshold": self.scene_detector.config.threshold,
                "processing_duration": round(processing_duration, 2),
            },
        }

        # Ensure dirs exist (safe if already created)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        scenes_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save transcript.json
        transcript_file = transcript_dir / "transcript.json"
        with open(transcript_file, "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        print(f"✓ Transcript saved to: {transcript_file}")

        # Save scenes.json
        scenes_file = scenes_dir / "scenes.json"
        with open(scenes_file, "w", encoding="utf-8") as f:
            json.dump(scenes, f, indent=2, ensure_ascii=False)
        print(f"✓ Scenes saved to: {scenes_file}")

        # Save results.json
        results_file = results_dir / "results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Full results saved to: {results_file}")
        print(f"  Processing time: {processing_duration:.2f}s")

        # Save HTML report
        report_file = results_dir / "report.html"
        self.create_html_report(results, report_file)

        return results

    def create_html_report(self, results: Dict, output_file: Path):
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Video Analysis Report: {results['video']['filename']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .scene {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007bff; }}
                .transcript {{ background: #f8f9fa; padding: 5px 10px; margin: 5px 0; border-radius: 3px; }}
                .keyframe {{ max-width: 300px; margin: 10px 0; }}
                .stats {{ display: flex; gap: 20px; flex-wrap: wrap; }}
                .stat-box {{ background: #e9ecef; padding: 10px; border-radius: 3px; min-width: 150px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Video Analysis Report</h1>
                <h2>{results['video']['filename']}</h2>
            </div>

            <div class="stats">
                <div class="stat-box">
                    <h3>Statistics</h3>
                    <p>Duration: {results['scene_analysis']['total_duration']:.1f}s</p>
                    <p>Scenes: {results['scene_analysis']['num_scenes']}</p>
                    <p>Transcript Segments: {results['transcription']['num_segments']}</p>
                </div>
            </div>

            <div class="section">
                <h2>Detected Scenes</h2>
        """

        for scene in results["scene_analysis"]["scenes"]:
            # safer if duration missing
            duration = scene.get("duration", 0.0)

            html_content += f"""
                <div class="scene">
                    <h3>Scene {scene.get('scene_id', '')}</h3>
                    <p>{scene.get('start_time', 0):.1f}s - {scene.get('end_time', 0):.1f}s ({duration:.1f}s)</p>
            """

            if scene.get("keyframe_path"):
                html_content += f"""
                    <div class="keyframe">
                        <img src="{scene['keyframe_path']}" alt="Keyframe" style="max-width: 300px;">
                    </div>
                """

            if scene.get("transcript_segments"):
                html_content += "<h4>Transcript Segments:</h4>"
                for seg in scene["transcript_segments"]:
                    html_content += f"""
                        <div class="transcript">
                            <strong>[{seg['start_str']}]</strong> {seg['text']}
                        </div>
                    """

            html_content += "</div>"

        html_content += """
            </div>

            <div class="section">
                <h2>Full Transcript</h2>
                <pre style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
        """

        for segment in results["transcription"]["segments"]:
            start_time = str(timedelta(seconds=segment["start"])).split(".")[0]
            html_content += f"[{start_time}] {segment['text']}\n"

        html_content += """
                </pre>
            </div>

            <div class="section">
                <h2> Processing Info</h2>
                <ul>
        """

        for key, value in results["processing_info"].items():
            html_content += f"<li><strong>{key}:</strong> {value}</li>"

        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✓ HTML report saved to: {output_file}")

    def batch_process(
        self,
        video_folder: str = "videos",
        selected: Optional[List[str]] = None,
        limit: Optional[int] = None,
        output_base: str = "processed",
        use_hash: bool = False,
        force: bool = False,
    ):
        video_folder = Path(video_folder)

        if selected:
            videos = [Path(s) if Path(s).suffix else (video_folder / s) for s in selected]
        else:
            videos = list(video_folder.glob("*.*"))

        if limit is not None:
            videos = videos[:limit]

        print(f"\n{'='*60}")
        print(f"Starting batch processing of {len(videos)} videos")
        print(f"{'='*60}")

        results = []
        for i, video_path in enumerate(videos, 1):
            print(f"\nVideo {i}/{len(videos)}: {video_path.name}")
            try:
                _ = self.process_video(
                    str(video_path),
                    output_base=output_base,
                    use_hash=use_hash,
                    force=force,
                )
                results.append({"video": video_path.name, "success": True})
            except Exception as e:
                print(f"✗ Processing failed: {str(e)}")
                results.append({"video": video_path.name, "success": False, "error": str(e)})

        self.create_batch_summary(results, output_base=output_base)
        return results

    def create_batch_summary(self, results: List[Dict], output_base: str = "processed"):
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        summary = {
            "total_videos": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "failed_videos": [r["video"] for r in failed],
            "saved_at_iso": datetime.now().isoformat(),
        }

        summary_dir = Path(output_base)
        summary_dir.mkdir(parents=True, exist_ok=True)

        summary_file = summary_dir / "batch_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Summary saved to: {summary_file}")

        if failed:
            print("\nFailed videos:")
            for fail in failed:
                print(f"  - {fail['video']}: {fail.get('error', 'Unknown error')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Basic Video Pipeline with SOTA Embeddings")
    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--folder", type=str, default="videos", help="Folder containing videos to batch process")
    parser.add_argument("--limit", type=int, help="Limit number of videos in batch process")
    parser.add_argument("--force", action="store_true", help="Force re-processing")
    parser.add_argument("--use-hash", action="store_true", help="Use SHA256 hashing for input change detection")
    parser.add_argument("--skip-db", action="store_true", help="Skip database ingestion")
    parser.add_argument("--ingest-only", action="store_true", help="Only perform database ingestion (results must exist)")
    parser.add_argument("--backend", type=str, default="whisper", help="ASR backend")
    parser.add_argument("--model", type=str, default="large-v3", help="Whisper model variant")
    parser.add_argument("--threshold", type=float, default=20.0, help="Scene detection threshold")

    args = parser.parse_args()

    pipeline = BasicVideoPipeline(
        backend=args.backend,
        model_variant={"name": args.model},
        scene_threshold=args.threshold,
        skip_ingest=args.skip_db,
    )

    if args.ingest_only:
        print("\nRunning Ingestion Only mode")
        if args.video:
            video_path = Path(args.video)
            results_file = Path("processed/results") / video_path.stem / "results.json"
            pipeline._ingest_results(results_file)
        else:
            # Batch ingest from processed/results
            try:
                with DataIngester() as ingester:
                    ingester.ingest_batch(
                        processed_dir="processed",
                        update_existing=True,
                        force=args.force
                    )
            except Exception as e:
                print(f"Batch ingestion failed: {e}")
    elif args.video:
        pipeline.process_video(
            args.video,
            force=args.force,
            use_hash=args.use_hash
        )
    else:
        pipeline.batch_process(
            video_folder=args.folder,
            limit=args.limit,
            force=args.force,
            use_hash=args.use_hash
        )
