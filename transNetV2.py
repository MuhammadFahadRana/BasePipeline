"""
transNetV2.py

Scene (shot boundary) detection using TransNetV2.
Paper: TransNet V2: An effective deep network architecture for fast shot
       transition detection (https://arxiv.org/abs/2008.04838)
Repo:  https://github.com/soCzech/TransNetV2

Install:
    pip install tensorflow==2.1   # or compatible version
    pip install ffmpeg-python pillow
    # Clone repo and install:
    #   git clone https://github.com/soCzech/TransNetV2.git
    #   cd TransNetV2 && python setup.py install
    # (make sure git-lfs weights are pulled: git lfs pull)

Output layout:
    processed/scenes/transNetV2/{VideoName}/
        {VideoName}_scenes.json
        {VideoName}_scene_0.jpg
        {VideoName}_scene_1.jpg
        ...
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

BASE_OUTPUT = Path("processed/scenes/transNetV2")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}


def _load_model(model_dir: Optional[str] = None):
    """Lazy-load the TransNetV2 TF model."""
    from transnetv2 import TransNetV2  # installed via setup.py
    return TransNetV2() if model_dir is None else TransNetV2(model_dir=model_dir)


def extract_keyframe(
    video_path: Path,
    start_time: float,
    end_time: float,
    output_dir: Path,
    scene_idx: int,
) -> Optional[Path]:
    """Extract a keyframe from the middle of a scene and save it as JPEG."""
    mid_time = (start_time + end_time) / 2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid_time * fps))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    keyframe_file = output_dir / f"{output_dir.name}_scene_{scene_idx}.jpg"
    cv2.imwrite(str(keyframe_file), frame)
    return keyframe_file


def detect_scenes(
    video_path: str,
    model=None,
    model_dir: Optional[str] = None,
    threshold: float = 0.5,
    force_reprocess: bool = False,
) -> List[Dict]:
    """
    Detect shot boundaries in *video_path* using TransNetV2.

    Args:
        video_path:       Path to the video file.
        model:            Pre-loaded TransNetV2 model (avoids reloading per video).
        model_dir:        Path to transnetv2-weights/ (auto-detected when None).
        threshold:        Prediction threshold for shot boundaries (default 0.5).
        force_reprocess:  Overwrite existing results.

    Returns:
        List of scene dicts compatible with the rest of the pipeline.
    """
    video_path = Path(video_path)

    if video_path.suffix.lower() in AUDIO_EXTENSIONS:
        print(f"Audio file detected: {video_path.name}. Skipping scene detection.")
        return []

    output_dir = BASE_OUTPUT / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_file = output_dir / f"{video_path.stem}_scenes.json"

    # Cache check
    if scene_file.exists() and not force_reprocess:
        print(f"Scenes already detected for {video_path.name}. Skipping.")
        try:
            with open(scene_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Corrupt scene file for {video_path.name}, reprocessing...")

    # Load model if not provided
    if model is None:
        model = _load_model(model_dir)

    print(f"[TransNetV2] Detecting scenes in: {video_path.name}")

    # Run TransNetV2 prediction
    video_frames, single_frame_preds, all_frame_preds = model.predict_video(
        str(video_path)
    )
    scenes_frames = model.predictions_to_scenes(
        single_frame_preds, threshold=threshold
    )
    # scenes_frames: np.ndarray of shape (N, 2) — [start_frame, end_frame] per scene

    # Get FPS for time conversion
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Build scene dicts with keyframes
    scenes: List[Dict] = []
    for idx, (sf, ef) in enumerate(scenes_frames):
        start_time = float(sf) / fps
        end_time = float(ef) / fps
        duration = end_time - start_time

        keyframe_path = extract_keyframe(
            video_path, start_time, end_time, output_dir, scene_idx=idx
        )

        scenes.append({
            "scene_id": idx,
            "start_time": round(start_time, 4),
            "end_time": round(end_time, 4),
            "duration": round(duration, 4),
            "start_frame": int(sf),
            "end_frame": int(ef),
            "keyframe_path": str(keyframe_path) if keyframe_path else None,
        })

    # Persist
    with open(scene_file, "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=2)

    print(
        f"[TransNetV2] {len(scenes)} scenes detected "
        f"({total_frames} frames, {fps:.2f} fps)"
    )
    print(f"[TransNetV2] Saved to: {scene_file}")
    return scenes


def batch_detect(
    video_folder: str = "videos",
    model_dir: Optional[str] = None,
    threshold: float = 0.5,
    force_reprocess: bool = False,
) -> List[Dict]:
    """Run TransNetV2 scene detection on every video in *video_folder*."""
    video_folder = Path(video_folder)
    skip_ext = {".json", ".txt", ".srt", ".vtt", ".log"} | AUDIO_EXTENSIONS
    videos = sorted(
        v for v in video_folder.glob("*.*")
        if v.suffix.lower() not in skip_ext
    )

    print(f"[TransNetV2] Found {len(videos)} videos in {video_folder}")

    model = _load_model(model_dir)
    summary: List[Dict] = []

    for i, vp in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {vp.name}")
        try:
            scenes = detect_scenes(
                vp,
                model=model,
                threshold=threshold,
                force_reprocess=force_reprocess,
            )
            summary.append({
                "video": vp.name,
                "num_scenes": len(scenes),
                "scenes_file": str(BASE_OUTPUT / vp.stem / f"{vp.stem}_scenes.json"),
                "success": True,
            })
        except Exception as e:
            print(f"  Failed: {e}")
            summary.append({
                "video": vp.name,
                "error": str(e),
                "success": False,
            })

    succeeded = sum(1 for s in summary if s["success"])
    print(f"\n[TransNetV2] Done: {succeeded}/{len(summary)} videos processed")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TransNetV2 shot boundary detection"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to a single video file. If omitted, batch-processes videos/",
    )
    parser.add_argument(
        "--video-folder", type=str, default="videos",
        help="Folder with videos for batch mode (default: videos/)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Prediction threshold (default: 0.5)",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Path to transnetv2-weights/ directory",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process even if results exist",
    )
    args = parser.parse_args()

    if args.video:
        detect_scenes(
            args.video,
            model_dir=args.model_dir,
            threshold=args.threshold,
            force_reprocess=args.force,
        )
    else:
        batch_detect(
            video_folder=args.video_folder,
            model_dir=args.model_dir,
            threshold=args.threshold,
            force_reprocess=args.force,
        )
