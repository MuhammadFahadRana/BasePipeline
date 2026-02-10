"""
NVIDIA NeMo ASR Validation
Uses state-of-the-art NeMo models (Parakeet/Canary) for validation.
"""

import nemo.collections.asr as nemo_asr
import torch
import json
from pathlib import Path
import difflib
import numpy as np
from typing import Dict, List, Optional
import shutil
import warnings
import os
import tempfile
import shutil
import subprocess
import imageio_ffmpeg

# Suppress NeMo warnings
warnings.filterwarnings("ignore")


# Ensure ffmpeg is in PATH so pydub can find it
def prepare_ffmpeg():
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)

        if os.name == "nt":
            ffmpeg_name = os.path.basename(ffmpeg_exe)
            if ffmpeg_name != "ffmpeg.exe":
                dest = os.path.join(ffmpeg_dir, "ffmpeg.exe")
                if not os.path.exists(dest):
                    try:
                        shutil.copy(ffmpeg_exe, dest)
                    except Exception as e:
                        print(f"Warning: Could not copy ffmpeg to ffmpeg.exe: {e}")

        if ffmpeg_dir not in os.environ["PATH"]:
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

    except Exception as e:
        print(f"Warning: Failed to setup ffmpeg path: {e}")


prepare_ffmpeg()


class NeMoValidator:
    """
    Validate/Transcribe using NVIDIA NeMo ASR models.

    Supported Models:
    - Parakeet-TDT (English): nvidia/parakeet-tdt-1.1b  (Fast & Accurate)
    - Canary (Multilingual): nvidia/canary-1b           (Best for multi-lang)
    - Parakeet-CTC (Smaller): nvidia/parakeet-ctc-1.1b  (Faster)
    """

    def __init__(self, model_name: str = "nvidia/parakeet-tdt-1.1b"):
        """
        Initialize NeMo ASR model.
        """
        print(f"Loading NeMo model: {model_name}")

        # Load pre-trained model from NGC
        try:
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name)

            # Explicitly check for CUDA and print status
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"Loading NeMo on GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                print("Warning: CUDA not detected. Loading NeMo on CPU (will be slow).")

            self.model.to(self.device)
            self.model.eval()
            self.model_name = model_name
            print(f"Model loaded and ready")
        except Exception as e:
            print(f"Error loading NeMo model: {e}")
            raise

    def transcribe_audio(
        self, audio_path: str, video_name: str = None, use_cache: bool = True
    ) -> str:
        """
        Transcribe audio file using NeMo with robust caching.

        Args:
            audio_path: Path to audio/video file
            video_name: Name of video (for caching). If None, extracted from path.
            use_cache: If True, load cached transcription if available and valid

        Returns:
            Transcription text
        """
        audio_path = Path(audio_path)

        # Determine video name for caching
        if video_name is None:
            video_name = audio_path.stem.replace(" ", "_")

        # Define cache location: processed/transcripts/NeMo-{model}/videoname/transcript.json
        model_short_name = self.model_name.split("/")[-1]
        cache_dir = (
            Path("processed/transcripts") / f"NeMo-{model_short_name}" / video_name
        )
        cache_file = cache_dir / "transcript.json"

        # Check for cached transcript
        if use_cache and cache_file.exists():
            try:
                # Check modification times
                video_mtime = audio_path.stat().st_mtime
                transcript_mtime = cache_file.stat().st_mtime

                if transcript_mtime > video_mtime:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                    print(f"Using cached NeMo transcription")
                    return cached_data.get("text", "")
                else:
                    print(
                        f"Cached transcript found but video is newer. Re-transcribing..."
                    )

            except Exception as e:
                print(f" Warning: Failed to load cache, will re-transcribe: {e}")

        print(f"  Transcribing with NeMo: {audio_path.name}", flush=True)

        # Extract audio to temp WAV using subprocess and imageio_ffmpeg directly
        # This resolves [WinError 2] issues caused by missing ffprobe
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                tmp_audio_path = tmp_audio.name

            # Explicitly close the file handle before subprocess tries to write to it
            # though ffmpeg -y should handle it, on Windows it can be locked.

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

            # Use ffmpeg to extract 16kHz mono WAV
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i",
                str(audio_path.absolute()),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                tmp_audio_path,
            ]

            print(f"  DEBUG: Extracting audio to {tmp_audio_path}...", flush=True)
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"  DEBUG ERROR: FFmpeg failed: {result.stderr}", flush=True)
                return ""

            if (
                not Path(tmp_audio_path).exists()
                or Path(tmp_audio_path).stat().st_size == 0
            ):
                print(
                    f"  DEBUG ERROR: Extracted audio file is missing or empty!",
                    flush=True,
                )
                return ""

            print(
                f"  DEBUG: Extraction successful ({Path(tmp_audio_path).stat().st_size} bytes)",
                flush=True,
            )

            # Transcribe
            print(f"  DEBUG: Running model.transcribe...", flush=True)
            # Pass as list with batch_size=1 to avoid lhotse dynamic sampler issues
            results = self.model.transcribe([tmp_audio_path], batch_size=1)

            print(f"  DEBUG: transcribe() returned type {type(results)}", flush=True)
            print(f"  DEBUG: transcribe() result content: {results}", flush=True)

            if not results:
                print("  DEBUG ERROR: NeMo returned no results!", flush=True)
                return ""

            transcription_result = results[0]

            # Handle potential list or tensor output
            if isinstance(transcription_result, list):
                transcription_result = transcription_result[0]

            # If it returns a tensor, convert to string (though usually it returns text list)
            if not isinstance(transcription_result, str):
                transcription_result = str(transcription_result)

            print(
                f"  DEBUG: Transcription string (first 50 chars): {transcription_result[:50]}...",
                flush=True,
            )

            # Cache results
            print(f"  DEBUG: Creating cache dir: {cache_dir}", flush=True)
            cache_dir.mkdir(parents=True, exist_ok=True)

            cache_data = {
                "text": transcription_result,
                "model": self.model_name,
                "video_file": audio_path.name,
            }

            print(f"  DEBUG: Writing to {cache_file}...", flush=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            print(f"  ✓ Cached to {cache_file}", flush=True)

            return transcription_result

        except Exception as e:
            print(f"  Error during NeMo transcription: {e}", flush=True)
            import traceback

            traceback.print_exc()
            return ""

        finally:
            # Cleanup temp file
            if "tmp_audio_path" in locals():
                print(f"  DEBUG: Cleaning up {tmp_audio_path}", flush=True)
                Path(tmp_audio_path).unlink(missing_ok=True)

    def process_all_videos(self, video_dir: str = "videos"):
        """
        Process all videos in the directory and generate NeMo transcripts.
        """
        video_dir = Path(video_dir)
        if not video_dir.exists():
            # Try exploring for videos
            found_dirs = list(Path(".").glob("**/videos"))
            if found_dirs:
                video_dir = found_dirs[0]
            else:
                # Fallback to local
                video_dir = Path(".")

        print(f"\n{'=' * 60}")
        print(f"NeMo BULK TRANSCRIPTION")
        print(f"Model: {self.model_name}")
        print(f"Source: {video_dir.absolute()}")
        print(f"{'=' * 60}")

        # Clean up search pattern to find video files
        extensions = ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.wav"]
        video_files = []
        for ext in extensions:
            video_files.extend(list(video_dir.glob(ext)))

        print(f"Found {len(video_files)} media files.")

        for i, video_path in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")
            self.transcribe_audio(str(video_path))


if __name__ == "__main__":
    print("\nDEBUG: Main block started")
    # Initialize validator
    validator = NeMoValidator(model_name="nvidia/parakeet-tdt-1.1b")
    print("DEBUG: Validator initialized")

    # 1. Run for a Single Video
    target_video = "videos/AkerBP 1.mp4"
    print(f"DEBUG: Target video set to: {target_video}")

    if os.path.exists(target_video):
        print(f"DEBUG: File found. Starting transcription...")
        validator.transcribe_audio(target_video)
        print("DEBUG: Transcription call finished.")
    else:
        print(f"DEBUG: ERROR - File not found: {target_video}")

    # 2. Run for All Videos in Folder
    # validator.process_all_videos("videos")
