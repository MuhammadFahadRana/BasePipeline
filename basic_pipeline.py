# -*- coding: utf-8 -*-
# ATLAS - AI-driven Temporal Linking and Search

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Callable, Dict, Optional, List
from datetime import timedelta, datetime
import time

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

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

    TRANSCRIPTION_MODEL_NAME = "whisper-large-v3"

    def __init__(
        self,
        scene_threshold: float = 30.0,
        device: str = "auto",
        skip_ingest: bool = False,
        ingest_target: Optional[str] = None,
        visual_enrichment: Optional[bool] = None,
    ):
        """
        Initialize pipeline with selected ASR backend and scene detector.

        Args:
            scene_threshold: Scene detection threshold
            device: Preferred device ("auto" recommended).
            skip_ingest: Skip database ingestion if True
            ingest_target: "postgres", "sqlserver", "both", or "none".
            visual_enrichment: Enable Qwen visual captions/OCR if True.
        """
        if visual_enrichment is None:
            visual_enrichment = (
                os.getenv("VISUAL_ENRICHMENT_ENABLED", "true").strip().lower()
                in ("1", "true", "yes", "on")
            )

        self.transcriber = SimpleTranscriber(
            backend="whisper",
            model_variant={"name": "large-v3"},
            device=device,
        )
        scene_device = (
            "cuda" if getattr(self.transcriber, "device", "cpu") == "cuda" else "cpu"
        )
        scene_cfg = SceneConfig(
            threshold=scene_threshold,
            clip_sim_merge_threshold=0.90,
            device=scene_device,
            enable_visual_enrichment=visual_enrichment,
        )
        self.scene_detector = SceneDetector(config=scene_cfg)
        self.backend = "whisper"
        self.model_variant = {"name": "large-v3"}
        self.skip_ingest = skip_ingest
        target = ingest_target or os.getenv("PIPELINE_INGEST_TARGET", "postgres")
        self.ingest_target = target.strip().lower()
        if self.ingest_target not in {"postgres", "sqlserver", "both", "none"}:
            self.ingest_target = "postgres"

        # Formats that should be auto-converted to .mp4 before processing
        self.CONVERT_EXTENSIONS = {
            ".ts",
            ".mp2t",
            ".m2ts",
            ".mts",
            ".avi",
            ".mkv",
            ".mov",
            ".webm",
            ".flv",
            ".wmv",
        }

    @staticmethod
    def _report_progress(
        progress_callback: Callable[[int, str, str], None] | None,
        percent: int,
        stage: str,
        message: str,
    ) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(percent, stage, message)
        except Exception:
            # Progress reporting must never fail the pipeline.
            return

    @staticmethod
    def _load_sqlserver_ingester():
        """Lazy-load SQL Server ingester to keep PostgreSQL-only mode isolated."""
        try:
            from database.SQL.ingest_sqlserver import SqlServerIngester

            return SqlServerIngester
        except Exception:
            return None

    # ---------------------------
    # Format conversion
    # ---------------------------
    def _convert_to_mp4(self, video_path: Path) -> Path:
        """
        Convert a non-mp4 video to .mp4 using FFmpeg stream-copy.
        Returns the path to the .mp4 file (original if already mp4).
        The original file is kept alongside the new .mp4.
        """
        if video_path.suffix.lower() not in self.CONVERT_EXTENSIONS:
            return video_path

        mp4_path = video_path.with_suffix(".mp4")

        if mp4_path.exists():
            print(f"  [skip] MP4 already exists: {mp4_path.name}")
            return mp4_path

        print(f"  [convert] Converting {video_path.suffix} -> .mp4: {video_path.name}")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-movflags",
            "faststart",
            str(mp4_path),
        ]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                err = result.stderr.decode("utf-8", errors="replace")[:300]
                print(f"  [warn] FFmpeg stream-copy failed, trying re-encode: {err}")
                # Fallback: re-encode (slower but handles incompatible codecs)
                cmd_reencode = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video_path),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-c:a",
                    "aac",
                    "-movflags",
                    "faststart",
                    str(mp4_path),
                ]
                subprocess.run(
                    cmd_reencode,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            size_mb = mp4_path.stat().st_size / (1024 * 1024)
            print(f"  [ok] Converted: {mp4_path.name} ({size_mb:.1f} MB)")
            return mp4_path

        except FileNotFoundError:
            print(
                "  [error] ffmpeg not found! Install it: https://ffmpeg.org/download.html"
            )
            print("      Continuing with original file (may cause issues)...")
            return video_path
        except subprocess.CalledProcessError as e:
            print(f"  [error] Conversion failed: {e}")
            print("      Continuing with original file (may cause issues)...")
            return video_path

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
            transcript_dir / "transcript.txt",
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

    def _configs_match(self, saved_cfg: Dict, current_cfg: Dict) -> bool:
        """Compare only the core processing fields that would require reprocessing."""
        core_keys = ("transcription_model", "scene_threshold")
        return all(saved_cfg.get(k) == current_cfg.get(k) for k in core_keys)

    def _manifest_ingested_for_target(self, manifest: Dict) -> bool:
        if not manifest.get("ingested", False):
            return False
        saved_target = str(manifest.get("ingest_target", "postgres")).lower()
        target = self.ingest_target
        if target == "both":
            return saved_target == "both"
        if saved_target == "both":
            return target in {"postgres", "sqlserver"}
        return saved_target == target

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
        generate_visual_embeddings: bool = True,
        _ingester=None,
        progress_callback: Callable[[int, str, str], None] | None = None,
    ):
        self._report_progress(progress_callback, 1, "init", "Preparing pipeline run")
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Auto-convert non-mp4 formats to mp4 before processing
        video_path = self._convert_to_mp4(video_path)

        output_base = Path(output_base)
        output_base.mkdir(parents=True, exist_ok=True)

        video_name = video_path.stem
        model_name = getattr(self.transcriber, "model_name", "unknown")

        transcript_dir = output_base / "transcripts" / model_name / video_name
        scenes_dir = output_base / "scenes" / video_name
        results_dir = output_base / "results" / video_name
        manifest_path = results_dir / "manifest.json"

        # Cache check
        current_fp = self._video_fingerprint(video_path, use_hash=use_hash)

        current_cfg = {
            "transcription_model": self.TRANSCRIPTION_MODEL_NAME,
            "scene_threshold": self.scene_detector.config.threshold,
            "clip_sim_merge_threshold": 0.90,
        }
        manifest = self._load_manifest(manifest_path)

        cache_hit = (
            (not force)
            and (manifest is not None)
            and (manifest.get("video_fingerprint") == current_fp)
            and self._configs_match(manifest.get("pipeline_config", {}), current_cfg)
            and self._expected_outputs_exist(transcript_dir, scenes_dir, results_dir)
        )

        if cache_hit:
            self._report_progress(
                progress_callback, 100, "done", "Using cached results"
            )
            results_file = results_dir / "results.json"
            already_ingested = self._manifest_ingested_for_target(manifest)

            if already_ingested:
                print(f"\n[cached] Skipping (cached + ingested): {video_name}")
            else:
                print(f"\n[cached] Skipping (cached): {video_name}")

            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    results = json.load(f)

                # Only ingest if not already done
                if not already_ingested and not self.skip_ingest and self.ingest_target != "none":
                    ingest_report = self._ingest_results(
                        results_file,
                        generate_embeddings=generate_embeddings,
                        generate_visual_embeddings=generate_visual_embeddings,
                        ingester=_ingester,
                    )
                    # Mark ingestion complete in manifest
                    manifest["ingested"] = bool(ingest_report.get("ok"))
                    manifest["ingest_target"] = self.ingest_target
                    manifest["ingest_report"] = ingest_report
                    self._save_manifest(manifest_path, manifest)

                return results
            except Exception as e:
                print(f"  ! Error reading cached results: {e}")
                return json.loads(results_file.read_text(encoding="utf-8"))

        # Otherwise process
        transcript_dir.mkdir(parents=True, exist_ok=True)
        scenes_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 50}")
        print(f"Processing: {video_name}")
        print(f"{'=' * 50}")

        start_time = time.time()

        self._report_progress(
            progress_callback, 10, "transcription", "Starting transcription"
        )
        print("\n1. Transcribing audio...")
        try:
            transcript = self.transcriber.transcribe_video(
                str(video_path), output_dir="processed"
            )
        except Exception as e:
            print(f"  ! Transcription failed: {e}")
            print("    Continuing with empty transcript.")
            transcript = {"text": "", "segments": [], "language": "unknown"}
        self._report_progress(
            progress_callback, 40, "transcription", "Transcription completed"
        )

        # Check if audio-only
        is_audio = (
            video_path.suffix.lower() in self.scene_detector.config.audio_extensions
        )

        if is_audio:
            self._report_progress(
                progress_callback, 55, "scenes", "Audio detected, skipping scene detection"
            )
            print("\n2. Audio file detected - Skipping scene detection & refinement.")
            # Create synthetic scene for the whole file
            last_end = 0.0
            if transcript.get("segments"):
                last_end = transcript["segments"][-1]["end"]

            scenes = [
                {
                    "scene_id": 0,
                    "start_time": 0.0,
                    "end_time": last_end,
                    "duration": last_end,
                    "keyframe_path": None,
                    "ocr_text": None,
                }
            ]
        else:
            self._report_progress(
                progress_callback, 50, "scenes", "Detecting scenes"
            )
            print("\n2. Detecting & refining scenes...")
            scenes = self.scene_detector.detect_scenes(
                video_path,
                base_output_dir=str(output_base / "scenes"),
                force_reprocess=force,
            )

            print("\n2b. Refining scenes...")
            try:
                scenes = self.scene_detector.refine_scenes(scenes)
            except Exception as e:
                print(f"Scene refinement failed: {e}")

            print("\n2c. Enriching scenes (captions, labels, OCR)...")
            try:
                scenes = self.scene_detector.enrich_with_visual_features(scenes)
                # Re-save scenes cache with enrichment data included
                scenes_cache = scenes_dir / f"{video_path.stem}_scenes.json"
                if scenes_cache.exists():
                    with open(scenes_cache, "w", encoding="utf-8") as f:
                        json.dump(scenes, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Visual enrichment failed: {e}")
            self._report_progress(
                progress_callback, 70, "scenes", "Scene analysis completed"
            )

        self._report_progress(progress_callback, 75, "alignment", "Aligning transcript")
        print("\n3. Aligning transcripts with scenes...")
        aligned_data = self.align_transcript_with_scenes(transcript, scenes)

        end_time = time.time()
        processing_duration = end_time - start_time

        self._report_progress(progress_callback, 82, "saving", "Saving output files")
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
        ingested = False
        ingest_report = {"ok": False, "target": self.ingest_target}
        if not self.skip_ingest and self.ingest_target != "none":
            self._report_progress(
                progress_callback, 90, "ingestion", "Ingesting into database"
            )
            ingest_report = self._ingest_results(
                results_file,
                generate_embeddings=generate_embeddings,
                generate_visual_embeddings=generate_visual_embeddings,
                ingester=_ingester,
            )
            ingested = bool(ingest_report.get("ok"))

        # Save manifest for caching
        new_manifest = {
            "video_filename": video_path.name,
            "video_path": str(video_path),
            "video_fingerprint": current_fp,
            "pipeline_config": current_cfg,
            "ingested": ingested,
            "ingest_target": self.ingest_target,
            "ingest_report": ingest_report,
            "saved_at_iso": datetime.now().isoformat(),
            "use_hash": use_hash,
        }
        self._save_manifest(manifest_path, new_manifest)
        print(f"[ok] Manifest saved to: {manifest_path}")
        self._report_progress(progress_callback, 100, "done", "Pipeline completed")

        return results

    def _ingest_results(
        self,
        results_file: Path,
        generate_embeddings: bool = True,
        generate_visual_embeddings: bool = True,
        ingester: "DataIngester | None" = None,
    ) -> Dict[str, object]:
        print(f"\n5. Ingesting into database (target={self.ingest_target})...")
        report: Dict[str, object] = {
            "ok": False,
            "target": self.ingest_target,
            "postgres_ok": False,
            "sqlserver_ok": False,
            "errors": [],
        }

        def _append_error(msg: str) -> None:
            print(f"  ! Ingestion failed: {msg}")
            report["errors"].append(msg)

        if self.ingest_target in {"postgres", "both"}:
            if not HAS_DB:
                _append_error("PostgreSQL ingester unavailable (database.ingest import failed)")
            else:
                try:
                    if ingester is not None:
                        ingester.ingest_video(
                            results_file,
                            generate_embeddings=generate_embeddings,
                            generate_visual_embeddings=generate_visual_embeddings,
                            update_existing=True,
                        )
                        report["postgres_ok"] = True
                    else:
                        db_error = self._postgres_connection_error()
                        if db_error:
                            _append_error(
                                "postgres unavailable before loading embedding models: "
                                f"{db_error}"
                            )
                        else:
                            with DataIngester() as ing:
                                ing.ingest_video(
                                    results_file,
                                    generate_embeddings=generate_embeddings,
                                    generate_visual_embeddings=generate_visual_embeddings,
                                    update_existing=True,
                                )
                            report["postgres_ok"] = True
                except Exception as e:
                    _append_error(f"postgres: {e}")

        if self.ingest_target in {"sqlserver", "both"}:
            sql_ingester_cls = self._load_sqlserver_ingester()
            if sql_ingester_cls is None:
                _append_error("SQL Server ingester unavailable (database.SQL.ingest_sqlserver import failed)")
            else:
                try:
                    sql_ing = sql_ingester_cls(
                        enable_text_embeddings=generate_embeddings,
                        enable_visual_embeddings=generate_visual_embeddings,
                    )
                    sql_ing.ingest_video_result_file(Path(results_file))
                    report["sqlserver_ok"] = True
                except Exception as e:
                    _append_error(f"sqlserver: {e}")

        if self.ingest_target == "postgres":
            report["ok"] = bool(report["postgres_ok"])
        elif self.ingest_target == "sqlserver":
            report["ok"] = bool(report["sqlserver_ok"])
        elif self.ingest_target == "both":
            report["ok"] = bool(report["postgres_ok"] and report["sqlserver_ok"])
        else:
            report["ok"] = True

        return report

    @staticmethod
    def _postgres_connection_error() -> Optional[str]:
        """Return a connection error string, or None when PostgreSQL is reachable."""
        try:
            from sqlalchemy import text
            from database.config import SessionLocal

            db = SessionLocal()
            try:
                db.execute(text("SELECT 1"))
            finally:
                db.close()
            return None
        except Exception as exc:
            return str(exc)

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
                "whisper_model": self.TRANSCRIPTION_MODEL_NAME,
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
        print(f"[ok] Transcript saved to: {transcript_file}")

        # Save transcript.txt (human-readable with timestamps)
        text_file = transcript_dir / "transcript.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(f"Transcription for {results['video']['filename']}\n")
            f.write("=" * 50 + "\n\n")
            for seg in transcript.get("segments", []):
                start = str(timedelta(seconds=seg["start"])).split(".")[0]
                text = seg.get("text", "").strip()
                f.write(f"[{start}] {text}\n")
        print(f"[ok] Transcript (text) saved to: {text_file}")

        # Save scenes.json
        scenes_file = scenes_dir / "scenes.json"
        with open(scenes_file, "w", encoding="utf-8") as f:
            json.dump(scenes, f, indent=2, ensure_ascii=False)
        print(f"[ok] Scenes saved to: {scenes_file}")

        # Save results.json
        results_file = results_dir / "results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[ok] Full results saved to: {results_file}")
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
            <title>Video Analysis Report: {results["video"]["filename"]}</title>
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
                <h2>{results["video"]["filename"]}</h2>
            </div>

            <div class="stats">
                <div class="stat-box">
                    <h3>Statistics</h3>
                    <p>Duration: {results["scene_analysis"]["total_duration"]:.1f}s</p>
                    <p>Scenes: {results["scene_analysis"]["num_scenes"]}</p>
                    <p>Transcript Segments: {results["transcription"]["num_segments"]}</p>
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
                    <h3>Scene {scene.get("scene_id", "")}</h3>
                    <p>{scene.get("start_time", 0):.1f}s - {scene.get("end_time", 0):.1f}s ({duration:.1f}s)</p>
            """

            if scene.get("keyframe_path"):
                html_content += f"""
                    <div class="keyframe">
                        <img src="{scene["keyframe_path"]}" alt="Keyframe" style="max-width: 300px;">
                    </div>
                """

            if scene.get("transcript_segments"):
                html_content += "<h4>Transcript Segments:</h4>"
                for seg in scene["transcript_segments"]:
                    html_content += f"""
                        <div class="transcript">
                            <strong>[{seg["start_str"]}]</strong> {seg["text"]}
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

        print(f"[ok] HTML report saved to: {output_file}")

    def batch_process(
        self,
        video_folder: str = "videos",
        selected: Optional[List[str]] = None,
        limit: Optional[int] = None,
        output_base: str = "processed",
        use_hash: bool = False,
        force: bool = False,
        generate_embeddings: bool = True,
        generate_visual_embeddings: bool = True,
    ):
        video_folder = Path(video_folder)

        if selected:
            videos = [
                Path(s) if Path(s).suffix else (video_folder / s) for s in selected
            ]
        else:
            videos = list(video_folder.glob("*.*"))

        if limit is not None:
            videos = videos[:limit]

        print(f"\n{'=' * 60}")
        print(f"Starting batch processing of {len(videos)} videos")
        print(f"{'=' * 60}")

        # Share a single DataIngester across the batch to avoid reloading
        # embedding models for every video.
        batch_ingester = None
        if HAS_DB and not self.skip_ingest and self.ingest_target in {"postgres", "both"}:
            try:
                db_error = self._postgres_connection_error()
                if db_error:
                    print(
                        "  ! Could not initialise PostgreSQL ingester before loading "
                        f"embedding models: {db_error}"
                    )
                else:
                    batch_ingester = DataIngester()
            except Exception as e:
                print(f"  ! Could not initialise ingester: {e}")

        results = []
        batch_start_time = time.time()
        for i, video_path in enumerate(videos, 1):
            print(f"\nVideo {i}/{len(videos)}: {video_path.name}")
            video_start_time = time.time()
            try:
                result = self.process_video(
                    str(video_path),
                    output_base=output_base,
                    use_hash=use_hash,
                    force=force,
                    generate_embeddings=generate_embeddings,
                    generate_visual_embeddings=generate_visual_embeddings,
                    _ingester=batch_ingester,
                )
                video_elapsed = time.time() - video_start_time
                processing_time = result.get("processing_info", {}).get(
                    "processing_duration", video_elapsed
                )
                results.append(
                    {
                        "video": video_path.name,
                        "success": True,
                        "processing_time": processing_time,
                        "wall_clock_time": video_elapsed,
                    }
                )
            except Exception as e:
                video_elapsed = time.time() - video_start_time
                print(f"Processing failed: {str(e)}")
                results.append(
                    {
                        "video": video_path.name,
                        "success": False,
                        "error": str(e),
                        "wall_clock_time": video_elapsed,
                    }
                )

        # Close the shared ingester
        if batch_ingester is not None:
            batch_ingester.__exit__(None, None, None)

        batch_total_time = time.time() - batch_start_time
        self.create_batch_summary(
            results, output_base=output_base, batch_total_time=batch_total_time
        )
        return results

    def create_batch_summary(
        self,
        results: List[Dict],
        output_base: str = "processed",
        batch_total_time: float = 0.0,
    ):
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        # Calculate timing statistics
        processing_times = [
            r.get("processing_time", 0) for r in successful if "processing_time" in r
        ]
        min_time = min(processing_times) if processing_times else 0
        max_time = max(processing_times) if processing_times else 0
        avg_time = (
            sum(processing_times) / len(processing_times) if processing_times else 0
        )

        summary = {
            "total_videos": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "failed_videos": [r["video"] for r in failed],
            "saved_at_iso": datetime.now().isoformat(),
            "timing": {
                "batch_total_time": round(batch_total_time, 2),
                "processing_time_min": round(min_time, 2),
                "processing_time_max": round(max_time, 2),
                "processing_time_avg": round(avg_time, 2),
                "total_processing_time": round(sum(processing_times), 2),
            },
        }

        summary_dir = Path(output_base)
        summary_dir.mkdir(parents=True, exist_ok=True)

        summary_file = summary_dir / "batch_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Save detailed timing CSV
        import csv

        csv_file = summary_dir / "batch_timing_details.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "video",
                    "success",
                    "processing_time_s",
                    "wall_clock_time_s",
                    "error",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "video": r["video"],
                        "success": "Yes" if r["success"] else "No",
                        "processing_time_s": round(r.get("processing_time", 0), 2),
                        "wall_clock_time_s": round(r.get("wall_clock_time", 0), 2),
                        "error": r.get("error", ""),
                    }
                )

        print(f"\n{'=' * 60}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'=' * 60}")
        print(f"Successful: {len(successful)}/{len(results)}")
        print(f"Failed: {len(failed)}/{len(results)}")
        print(f"\nTiming Statistics:")
        print(f"  Batch Total Time: {batch_total_time:.2f}s")
        print(f"  Min Time: {min_time:.2f}s")
        print(f"  Max Time: {max_time:.2f}s")
        print(f"  Avg Time: {avg_time:.2f}s")
        print(f"  Total Processing Time: {sum(processing_times):.2f}s")
        print(f"\nSummary saved to: {summary_file}")
        print(f"Detailed timing saved to: {csv_file}")

        if failed:
            print("\nFailed videos:")
            for fail in failed:
                print(f"  - {fail['video']}: {fail.get('error', 'Unknown error')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Basic Video Pipeline (Whisper Large v3)"
    )
    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument(
        "--folder",
        type=str,
        default="videos",
        help="Folder containing videos to batch process",
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of videos in batch process"
    )
    parser.add_argument("--force", action="store_true", help="Force re-processing")
    parser.add_argument(
        "--use-hash",
        action="store_true",
        help="Use SHA256 hashing for input change detection",
    )
    parser.add_argument(
        "--skip-db", action="store_true", help="Skip database ingestion"
    )
    parser.add_argument(
        "--ingest-target",
        choices=["postgres", "sqlserver", "both", "none"],
        help="Database target for ingestion. Overrides PIPELINE_INGEST_TARGET.",
    )
    parser.add_argument(
        "--no-visual-enrichment",
        action="store_true",
        help="Skip Qwen visual captions/object labels/OCR enrichment.",
    )
    parser.add_argument(
        "--text-embedding-model",
        help="Override TEXT_EMBEDDING_MODEL for database ingestion.",
    )
    parser.add_argument(
        "--vision-embedding-model",
        help="Override VISION_EMBEDDING_MODEL for database ingestion.",
    )
    parser.add_argument(
        "--no-text-embeddings",
        action="store_true",
        help="Skip text embedding generation during database ingestion.",
    )
    parser.add_argument(
        "--no-visual-embeddings",
        action="store_true",
        help="Skip visual embedding generation during database ingestion.",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only perform database ingestion (results must exist)",
    )
    parser.add_argument(
        "--threshold", type=float, default=20.0, help="Scene detection threshold"
    )

    args = parser.parse_args()

    if args.text_embedding_model:
        os.environ["TEXT_EMBEDDING_MODEL"] = args.text_embedding_model
    if args.vision_embedding_model:
        os.environ["VISION_EMBEDDING_MODEL"] = args.vision_embedding_model

    pipeline = BasicVideoPipeline(
        scene_threshold=args.threshold,
        skip_ingest=args.skip_db,
        ingest_target=args.ingest_target,
        visual_enrichment=False if args.no_visual_enrichment else None,
    )

    if args.ingest_only:
        print("\nRunning Ingestion Only mode")
        if args.video:
            video_path = Path(args.video)
            results_file = (
                Path("processed") / "results" / video_path.stem / "results.json"
            )
            pipeline._ingest_results(
                results_file,
                generate_embeddings=not args.no_text_embeddings,
                generate_visual_embeddings=not args.no_visual_embeddings,
            )
        else:
            # Batch ingest from processed/results
            try:
                if pipeline.ingest_target in {"none", "sqlserver"}:
                    print(
                        "Batch ingest-only mode currently uses the PostgreSQL "
                        "ingester; set --ingest-target postgres/both or pass --video."
                    )
                else:
                    db_error = pipeline._postgres_connection_error()
                    if db_error:
                        print(
                            "Batch ingestion skipped because PostgreSQL is not "
                            f"reachable: {db_error}"
                        )
                    else:
                        with DataIngester() as ingester:
                            ingester.ingest_batch(
                                processed_dir="processed",
                                update_existing=True,
                                force=args.force,
                            )
            except Exception as e:
                print(f"Batch ingestion failed: {e}")
    elif args.video:
        pipeline.process_video(
            args.video,
            force=args.force,
            use_hash=args.use_hash,
            generate_embeddings=not args.no_text_embeddings,
            generate_visual_embeddings=not args.no_visual_embeddings,
        )
    else:
        pipeline.batch_process(
            video_folder=args.folder,
            limit=args.limit,
            force=args.force,
            use_hash=args.use_hash,
            generate_embeddings=not args.no_text_embeddings,
            generate_visual_embeddings=not args.no_visual_embeddings,
        )
