"""
scene_detector.py

Unified scene processing pipeline:
  1. Format conversion  – FFmpeg-based, handles .ts/.avi/.mkv/etc.
  2. Scene detection     – PySceneDetect (ContentDetector)
  3. Semantic refinement – YOLO person detection + CLIP similarity merging
  4. OCR enrichment      – Text extraction from keyframes via EasyOCR
"""

import cv2
import subprocess
import shutil
import tempfile
import numpy as np
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from pathlib import Path
import json
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Lazy imports for optional heavy dependencies
_torch = None
_open_clip = None
_YOLO = None


def _ensure_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _ensure_open_clip():
    global _open_clip
    if _open_clip is None:
        import open_clip
        _open_clip = open_clip
    return _open_clip


def _ensure_yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO
    return _YOLO


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
@dataclass
class SceneConfig:
    """All scene-related settings in one place."""
    # Detection
    threshold: float = 30.0
    min_scene_len: int = 15
    max_scene_duration: float = 60.0  # split scenes longer than this (seconds)

    # Semantic refinement
    enable_refinement: bool = True
    yolo_model: str = "yolov8n.pt"
    yolo_person_conf: float = 0.35
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    clip_sim_merge_threshold: float = 0.90

    # OCR enrichment
    enable_ocr: bool = True
    ocr_languages: List[str] = field(default_factory=lambda: ["en"])
    ocr_use_gpu: bool = True
    ocr_confidence_threshold: float = 0.5

    # Visual Enrichment (Qwen2-VL)
    enable_visual_enrichment: bool = True
    qwen_vl_model: str = "Qwen/Qwen2-VL-7B-Instruct"
    qwen_vl_load_in_4bit: bool = True

    # Format conversion
    ffmpeg_path: str = "ffmpeg"  # assumes ffmpeg is on PATH
    compatible_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov")
    audio_extensions: Tuple[str, ...] = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma")

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"

    def get_device(self) -> str:
        if self.device == "auto":
            torch = _ensure_torch()
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


# ──────────────────────────────────────────────
# Scene Detector (unified)
# ──────────────────────────────────────────────
class SceneDetector:
    """
    Unified scene processing:
    detect → refine (YOLO+CLIP) → enrich (OCR) → save
    """

    def __init__(self, config: Optional[SceneConfig] = None, threshold: float = None):
        """
        Initialize scene detector.

        Args:
            config: Full SceneConfig object
            threshold: Shorthand to override detection threshold only
        """
        self.config = config or SceneConfig()
        if threshold is not None:
            self.config.threshold = threshold

        # Lazy-loaded models
        self._yolo = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._ocr = None
        self._qwen_vl = None

    # ── Lazy model loaders ──────────────────────

    def _ensure_yolo(self):
        if self._yolo is None:
            YOLO = _ensure_yolo()
            self._yolo = YOLO(self.config.yolo_model)
        return self._yolo

    def _ensure_clip(self):
        if self._clip_model is None:
            torch = _ensure_torch()
            open_clip = _ensure_open_clip()
            device = self.config.get_device()

            model, _, preprocess = open_clip.create_model_and_transforms(
                self.config.clip_model, pretrained=self.config.clip_pretrained
            )
            self._clip_model = model.to(device).eval()
            self._clip_preprocess = preprocess
            self._clip_tokenizer = open_clip.get_tokenizer(self.config.clip_model)
        return self._clip_model, self._clip_preprocess

    def _ensure_ocr(self):
        if self._ocr is None:
            try:
                from embeddings.ocr import get_ocr_reader
                self._ocr = get_ocr_reader(
                    languages=self.config.ocr_languages,
                    use_gpu=self.config.ocr_use_gpu,
                )
            except ImportError:
                print("Warning: EasyOCR not found. OCR enrichment disabled.")
                self.config.enable_ocr = False
                return None
        return self._ocr

    def _ensure_qwen_vl(self):
        if self._qwen_vl is None:
            try:
                from extract_visual_features import VisualFeatureExtractor
                self._qwen_vl = VisualFeatureExtractor(
                    model_name=self.config.qwen_vl_model,
                    device=self.config.get_device(),
                    load_in_4bit=self.config.qwen_vl_load_in_4bit
                )
            except ImportError:
                print("Warning: VisualFeatureExtractor dependencies not found. Visual enrichment disabled.")
                self.config.enable_visual_enrichment = False
                return None
        return self._qwen_vl

    # ── 1. Format Conversion ────────────────────

    def _ensure_compatible_format(self, video_path: Path) -> Tuple[Path, bool]:
        """
        Convert video to mp4 if not in a compatible format.

        Returns:
            (path_to_use, is_temp) — if is_temp, caller should clean up.
        """
        if video_path.suffix.lower() in self.config.compatible_extensions:
            return video_path, False

        # Check ffmpeg is available
        ffmpeg = self.config.ffmpeg_path
        if not shutil.which(ffmpeg):
            print(f"Warning: ffmpeg not found on PATH, proceeding with original file")
            return video_path, False

        # Convert to temp mp4
        temp_dir = Path(tempfile.mkdtemp(prefix="scene_conv_"))
        temp_path = temp_dir / f"{video_path.stem}_converted.mp4"

        print(f"  Converting {video_path.suffix} → .mp4 (FFmpeg)...")
        cmd = [
            ffmpeg, "-y",
            "-i", str(video_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-loglevel", "warning",
            str(temp_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"  FFmpeg error: {result.stderr[:500]}")
                # Fall back to original
                shutil.rmtree(temp_dir, ignore_errors=True)
                return video_path, False

            print(f"  ✓ Converted successfully ({temp_path.stat().st_size / 1e6:.1f} MB)")
            return temp_path, True

        except subprocess.TimeoutExpired:
            print(f"  FFmpeg timed out for {video_path.name}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return video_path, False
        except Exception as e:
            print(f"  FFmpeg failed: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return video_path, False

    def _cleanup_temp(self, temp_path: Path):
        """Remove temp converted video and its parent dir."""
        try:
            parent = temp_path.parent
            temp_path.unlink(missing_ok=True)
            if parent.name.startswith("scene_conv_"):
                shutil.rmtree(parent, ignore_errors=True)
        except Exception:
            pass

    # ── 2. Scene Detection ──────────────────────

    def detect_scenes(
        self,
        video_path: str,
        base_output_dir: str = "processed/scenes",
        force_reprocess: bool = False,
    ) -> List[Dict]:
        """
        Detect scenes in a video. Handles format conversion automatically.

        Args:
            video_path: Path to video file (any format)
            base_output_dir: Base directory for output
            force_reprocess: Whether to overwrite existing results

        Returns:
            List of scene dicts with start/end times and keyframes
        """
        video_path = Path(video_path)
        output_dir = Path(base_output_dir) / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        scene_file = output_dir / f"{video_path.stem}_scenes.json"

        # Check if already processed
        if scene_file.exists() and not force_reprocess:
            print(f"Scenes already detected for {video_path.name}. Skipping reprocessing.")
            try:
                with open(scene_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Corrupt scene file found for {video_path.name}, reprocessing...")

        # Check for audio-only files
        if video_path.suffix.lower() in self.config.audio_extensions:
            print(f"Audio file detected: {video_path.name}. Skipping scene detection.")
            return []

        print(f"Detecting scenes in: {video_path.name}")

        # Convert if needed
        working_path, is_temp = self._ensure_compatible_format(video_path)

        try:
            # Setup video using modern scenedetect API
            video = open_video(str(working_path))
            scene_manager = SceneManager()
            scene_manager.add_detector(
                ContentDetector(
                    threshold=self.config.threshold,
                    min_scene_len=self.config.min_scene_len,
                )
            )

            # Detect scenes
            scene_manager.detect_scenes(frame_source=video)
            scene_list = scene_manager.get_scene_list()

            # Post-process: split long scenes
            final_scenes = []
            current_scene_idx = 0

            for i, (start_frame, end_frame) in enumerate(scene_list):
                start_time = start_frame.get_seconds()
                end_time = end_frame.get_seconds()
                duration = end_time - start_time

                if duration > self.config.max_scene_duration:
                    num_splits = int(np.ceil(duration / self.config.max_scene_duration))
                    split_duration = duration / num_splits

                    for k in range(num_splits):
                        sub_start = start_time + (k * split_duration)
                        sub_end = min(end_time, sub_start + split_duration)

                        # Use original video path for keyframe extraction
                        keyframe_path = self.extract_keyframe(
                            working_path, sub_start, sub_end,
                            output_dir, scene_idx=current_scene_idx,
                        )

                        scene_data = {
                            "scene_id": current_scene_idx,
                            "start_time": sub_start,
                            "end_time": sub_end,
                            "duration": sub_end - sub_start,
                            "keyframe_path": str(keyframe_path) if keyframe_path else None,
                        }
                        final_scenes.append(scene_data)
                        print(
                            f"  Scene {current_scene_idx} (split {k + 1}/{num_splits}): "
                            f"{sub_start:.1f}s - {sub_end:.1f}s ({scene_data['duration']:.1f}s)"
                        )
                        current_scene_idx += 1
                else:
                    keyframe_path = self.extract_keyframe(
                        working_path, start_time, end_time,
                        output_dir, scene_idx=current_scene_idx,
                    )

                    scene_data = {
                        "scene_id": current_scene_idx,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration,
                        "start_frame": start_frame.get_frames(),
                        "end_frame": end_frame.get_frames(),
                        "keyframe_path": str(keyframe_path) if keyframe_path else None,
                    }
                    final_scenes.append(scene_data)
                    print(
                        f"  Scene {current_scene_idx}: {start_time:.1f}s - {end_time:.1f}s "
                        f"({scene_data['duration']:.1f}s)"
                    )
                    current_scene_idx += 1

            # Save scene information
            with open(scene_file, "w") as f:
                json.dump(final_scenes, f, indent=2)

            print(f"Detected {len(final_scenes)} scenes (after post-processing)")
            print(f"Scene info saved to: {scene_file}")

            return final_scenes

        finally:
            if is_temp:
                self._cleanup_temp(working_path)

    def extract_keyframe(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        output_dir: Path,
        scene_idx: int,
    ) -> Optional[Path]:
        """
        Extract a keyframe from the middle of a scene.

        Returns:
            Path to saved keyframe image
        """
        video_path = Path(video_path)
        mid_time = (start_time + end_time) / 2

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        mid_frame = int(mid_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Save keyframe — use the original video stem for naming
        keyframe_file = output_dir / f"{output_dir.name}_scene_{scene_idx}.jpg"
        cv2.imwrite(str(keyframe_file), frame)
        return keyframe_file

    # ── 3. Semantic Refinement ──────────────────

    def _person_present(self, image_path: str) -> Tuple[bool, float]:
        """Detect person using YOLO. Returns (present, max_confidence)."""
        if not image_path or not Path(image_path).exists():
            return False, 0.0

        yolo = self._ensure_yolo()
        results = yolo(image_path, verbose=False)

        max_conf = 0.0
        present = False

        for r in results:
            if r.boxes is None:
                continue
            cls = r.boxes.cls.detach().cpu().numpy().astype(int)
            conf = r.boxes.conf.detach().cpu().numpy()
            person_confs = conf[cls == 0] if len(cls) else np.array([])
            if person_confs.size > 0:
                max_conf = float(person_confs.max())
                if max_conf >= self.config.yolo_person_conf:
                    present = True

        return present, max_conf

    def _clip_embed(self, image_path: str) -> Optional[np.ndarray]:
        """Get CLIP embedding for an image."""
        if not image_path or not Path(image_path).exists():
            return None

        torch = _ensure_torch()
        model, preprocess = self._ensure_clip()
        device = self.config.get_device()

        img = Image.open(image_path).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feats = model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.squeeze(0).detach().cpu().numpy()

    @staticmethod
    def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def refine_scenes(self, scenes: List[Dict]) -> List[Dict]:
        """
        Refine scenes with semantic signals:
          - Tag each scene with speaker_present / speaker_conf
          - Compute CLIP similarity between consecutive scenes
          - Merge near-identical consecutive speaker scenes

        Args:
            scenes: List of scene dicts from detect_scenes()

        Returns:
            Refined (possibly merged) list of scene dicts
        """
        if not self.config.enable_refinement or not scenes:
            return scenes

        print("  Annotating scenes with YOLO + CLIP...")

        # 1) Annotate with YOLO + CLIP
        clip_embs: List[Optional[np.ndarray]] = []

        for i, s in enumerate(scenes):
            kf = s.get("keyframe_path")

            speaker_present, speaker_conf = self._person_present(kf) if kf else (False, 0.0)
            s["speaker_present"] = speaker_present
            s["speaker_conf"] = speaker_conf

            emb = self._clip_embed(kf) if kf else None
            clip_embs.append(emb)

            if i > 0 and emb is not None and clip_embs[i - 1] is not None:
                s["clip_sim_to_prev"] = self._cos_sim(clip_embs[i - 1], emb)
            else:
                s["clip_sim_to_prev"] = None

        # 2) Merge consecutive similar speaker scenes
        merged: List[Dict] = []
        i = 0
        while i < len(scenes):
            cur = scenes[i]
            cur_emb = clip_embs[i]

            j = i + 1
            while j < len(scenes):
                nxt = scenes[j]
                nxt_emb = clip_embs[j]

                if not (cur.get("speaker_present") and nxt.get("speaker_present")):
                    break
                if cur_emb is None or nxt_emb is None:
                    break

                sim = self._cos_sim(cur_emb, nxt_emb)
                if sim < self.config.clip_sim_merge_threshold:
                    break

                # Merge next into current
                cur["end_time"] = nxt["end_time"]
                cur["duration"] = float(cur["end_time"] - cur["start_time"])
                j += 1

            merged.append(cur)
            i = j

        # Reassign scene IDs
        for idx, s in enumerate(merged):
            s["scene_id"] = idx

        merge_count = len(scenes) - len(merged)
        if merge_count > 0:
            print(f"  Merged {merge_count} similar consecutive scenes → {len(merged)} scenes")
        else:
            print(f"  No scenes merged ({len(merged)} scenes)")

        return merged

    # ── 4. OCR Enrichment ───────────────────────

    def enrich_with_ocr(self, scenes: List[Dict]) -> List[Dict]:
        """
        Run OCR on each scene keyframe and add ocr_text field.

        Args:
            scenes: List of scene dicts (must have keyframe_path)

        Returns:
            Same list with ocr_text added to each scene
        """
        if not self.config.enable_ocr:
            return scenes

        ocr = self._ensure_ocr()
        if ocr is None:
            return scenes

        print(f"  Running OCR on {len(scenes)} scene keyframes...")
        ocr_count = 0

        for scene in scenes:
            kf = scene.get("keyframe_path")
            if not kf or not Path(kf).exists():
                scene["ocr_text"] = None
                continue

            try:
                text = ocr.extract_text(
                    kf,
                    confidence_threshold=self.config.ocr_confidence_threshold,
                    clean=True,
                )
                scene["ocr_text"] = text if text else None
                if text:
                    ocr_count += 1
            except Exception as e:
                print(f"    OCR failed for scene {scene.get('scene_id')}: {e}")
                scene["ocr_text"] = None

        print(f"  ✓ OCR complete: {ocr_count}/{len(scenes)} scenes had text")
        return scenes

    # ── 5. Visual Enrichment (Qwen2-VL) ─────────

    def enrich_with_visual_features(self, scenes: List[Dict]) -> List[Dict]:
        """
        Enrich scenes with captions and object labels using Qwen2-VL.

        Args:
            scenes: List of scene dicts (must have keyframe_path)

        Returns:
            Same list with caption and object_labels added
        """
        if not self.config.enable_visual_enrichment:
            return scenes

        qwen = self._ensure_qwen_vl()
        if qwen is None:
            return scenes

        print(f"  Running visual enrichment (Qwen2-VL) on {len(scenes)} scenes...")
        count = 0

        for scene in scenes:
            kf = scene.get("keyframe_path")
            if not kf or not Path(kf).exists():
                scene["caption"] = None
                scene["object_labels"] = []
                continue

            try:
                result = qwen.analyze_image(kf)
                scene.update({
                    "caption": result.get("caption"),
                    "object_labels": result.get("object_labels", []),
                    "ocr_text": result.get("ocr_text")  # New: Qwen2-VL OCR
                })
                count += 1
            except Exception as e:
                print(f"    Visual enrichment failed for scene {scene.get('scene_id')}: {e}")
                scene["caption"] = None
                scene["object_labels"] = []

        print(f"  ✓ Visual enrichment complete: {count}/{len(scenes)} scenes processed")
        return scenes

    # ── 6. Full Pipeline ────────────────────────

    def process_video(
        self,
        video_path: str,
        base_output_dir: str = "processed/scenes",
        force_reprocess: bool = False,
        run_refinement: bool = None,
        run_ocr: bool = None,
    ) -> List[Dict]:
        """
        Full scene processing pipeline: detect → refine → OCR.

        Args:
            video_path: Path to video file (any format)
            base_output_dir: Output directory for scene data
            force_reprocess: Overwrite existing results
            run_refinement: Override config.enable_refinement
            run_ocr: Override config.enable_ocr

        Returns:
            List of fully processed scene dicts
        """
        # Detect
        scenes = self.detect_scenes(
            video_path,
            base_output_dir=base_output_dir,
            force_reprocess=force_reprocess,
        )

        # Refine
        do_refine = run_refinement if run_refinement is not None else self.config.enable_refinement
        if do_refine and scenes:
            try:
                scenes = self.refine_scenes(scenes)
            except Exception as e:
                print(f"  ! Refinement failed: {e}")

        # OCR (EasyOCR) - only if Qwen-VL is not fulfilling enrichment or specifically forced
        do_ocr = run_ocr if run_ocr is not None else self.config.enable_ocr
        if do_ocr and scenes and not self.config.enable_visual_enrichment:
            try:
                scenes = self.enrich_with_ocr(scenes)
            except Exception as e:
                print(f"  ! OCR enrichment failed: {e}")

        # Visual Enrichment (Qwen2-VL) - includes Caption, Labels, and OCR
        if self.config.enable_visual_enrichment and scenes:
            try:
                # This now provides ocr_text as well, replacing the need for EasyOCR
                scenes = self.enrich_with_visual_features(scenes)
            except Exception as e:
                print(f"  ! Visual enrichment failed: {e}")

        # Save enriched scenes
        video_path = Path(video_path)
        output_dir = Path(base_output_dir) / video_path.stem
        scene_file = output_dir / f"{video_path.stem}_scenes.json"
        with open(scene_file, "w") as f:
            json.dump(scenes, f, indent=2)

        return scenes

    # ── 6. Visualization ────────────────────────

    def visualize_scenes(
        self, video_path: str, scenes: list, output_file: str = None
    ):
        """
        Create a visualization of scene boundaries.
        Generates a strip of thumbnails for each scene.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        thumb_width = 160
        thumb_height = 90
        thumbnails = []

        for scene in scenes:
            mid_time = (scene["start_time"] + scene["end_time"]) / 2
            mid_frame = int(mid_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            if ret:
                thumb = cv2.resize(frame, (thumb_width, thumb_height))
                thumbnails.append(thumb)

        cap.release()

        if not thumbnails:
            print("No thumbnails to visualize")
            return

        # Create a strip of thumbnails
        strip = np.hstack(thumbnails)

        if output_file:
            cv2.imwrite(output_file, strip)
            print(f"Scene visualization saved to: {output_file}")
        else:
            output_file = f"processed/scenes/{video_path.stem}/visualization.png"
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_file, strip)
            print(f"Scene visualization saved to: {output_file}")

    # ── 7. Batch Processing ─────────────────────

    def batch_detect(self, video_folder: str = "videos"):
        """Detect scenes for all videos in a folder."""
        video_folder = Path(video_folder)
        videos = sorted(video_folder.glob("*.*"))

        # Filter out non-video files
        skip_ext = {".json", ".txt", ".wav", ".mp3", ".srt", ".vtt", ".log"}
        videos = [v for v in videos if v.suffix.lower() not in skip_ext and "test_audio" not in v.name]

        print(f"Found {len(videos)} videos for scene detection")

        all_scenes = []
        for i, video_path in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] Processing: {video_path.name}")

            try:
                scenes = self.detect_scenes(video_path)
                all_scenes.append({
                    "video": video_path.name,
                    "scenes_file": f"processed/scenes/{video_path.stem}/{video_path.stem}_scenes.json",
                    "num_scenes": len(scenes),
                    "success": True,
                })
            except Exception as e:
                print(f"Failed to detect scenes in {video_path.name}: {str(e)}")
                all_scenes.append({
                    "video": video_path.name,
                    "error": str(e),
                    "success": False,
                })

        return all_scenes


if __name__ == "__main__":
    detector = SceneDetector(threshold=20.0)

    # # Test with a single video
    # scenes = detector.process_video("videos/Risk management.mp4")

    # Run batch detection
    detector.batch_detect()