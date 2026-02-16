#!/usr/bin/env python3
"""
NVIDIA ASR Transcriber (Local NeMo Inference)

Transcribe VIDEO files using NVIDIA Parakeet models locally via NeMo.

What it does:
1) Scans a video folder for supported video/audio files
2) Extracts + converts audio to 16kHz mono WAV using ffmpeg
3) Runs local NeMo ASR inference (no API key needed)
4) Saves results (JSON + text) per video

Requirements:
- ffmpeg installed and available on PATH
- Python packages: pip install nemo_toolkit[asr] torch torchaudio
- GPU recommended (CUDA) but CPU works too

Run:
    python nvidia_transcriber.py --video-folder videos
    python nvidia_transcriber.py --video-folder videos --limit 2 --model parakeet-ctc-1.1b
    python nvidia_transcriber.py --single videos/my_video.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import torch


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".opus"}

# Map short model names to HuggingFace repo IDs
MODEL_MAP = {
    "parakeet-ctc-1.1b": "nvidia/parakeet-ctc-1.1b",
    "parakeet-ctc-0.6b": "nvidia/parakeet-ctc-0.6b",
}

AVAILABLE_MODELS = list(MODEL_MAP.keys())


# ── ffmpeg helpers ──────────────────────────────────────────────────────────


def ensure_ffmpeg_available() -> str:
    """Check ffmpeg is on PATH and return its path."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg was not found on PATH.\n"
            "Install ffmpeg and ensure 'ffmpeg -version' works in your terminal.\n"
            "Windows tip: download ffmpeg, add its 'bin' folder to PATH."
        )
    return ffmpeg_path


def has_audio_stream(input_path: Path) -> bool:
    """Check if a media file contains an audio stream using ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        # If ffprobe isn't available, assume audio exists and let ffmpeg handle it
        return True

    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(input_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0 and "audio" in proc.stdout.strip()
    except Exception:
        return True  # Assume audio exists if probe fails


def run_ffmpeg_extract_to_wav(
    input_path: Path,
    output_wav: Path,
    sample_rate: int = 16000,
) -> None:
    """Convert input media to a mono PCM WAV at sample_rate."""
    ffmpeg = ensure_ffmpeg_available()

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(output_wav),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as e:
        raise RuntimeError(f"Failed to run ffmpeg: {e}") from e

    if proc.returncode != 0 or not output_wav.exists() or output_wav.stat().st_size == 0:
        raise RuntimeError(
            "ffmpeg failed to extract/convert audio.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDERR:\n{proc.stderr.strip()}\n"
        )


# ── Transcriber class ──────────────────────────────────────────────────────


class NvidiaTranscriber:
    """Transcribe video/audio files using NVIDIA Parakeet models locally via NeMo."""

    MODEL_DISPLAY_NAME = "NVIDIA-ASR"

    def __init__(
        self,
        model: str = "parakeet-ctc-1.1b",
        device: str = "cuda",
    ) -> None:
        self.model_name = model
        self.device = self._select_device(device)

        # Resolve HF repo ID
        hf_model = MODEL_MAP.get(model, model)

        # Load the NeMo CTC model locally
        print(f"Loading NeMo model '{hf_model}' on {self.device}...")
        from nemo.collections.asr.models import EncDecCTCModelBPE

        try:
            self.model = EncDecCTCModelBPE.from_pretrained(
                model_name=hf_model, map_location=self.device
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{hf_model}' locally.\n"
                "Make sure nemo_toolkit[asr] is installed:\n"
                "  pip install nemo_toolkit[asr] torch torchaudio\n"
                f"Original error: {e}"
            ) from e

        self.model.eval()

        # Monkeypatch: bypass broken lhotse DynamicCutSampler
        # NeMo hardcodes use_lhotse=True in _setup_transcribe_dataloader,
        # but the installed lhotse version has an incompatible CutSampler.
        # Override to use the standard NeMo dataloader instead.
        import types
        from omegaconf import DictConfig

        _original_setup = self.model._setup_transcribe_dataloader

        def _patched_setup_transcribe_dataloader(config):
            if 'manifest_filepath' in config:
                manifest_filepath = config['manifest_filepath']
                batch_size = config['batch_size']
            else:
                manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
                batch_size = min(config['batch_size'], len(config['paths2audio_files']))

            dl_config = {
                'manifest_filepath': manifest_filepath,
                'sample_rate': self.model.preprocessor._sample_rate,
                'batch_size': batch_size,
                'shuffle': False,
                'num_workers': config.get('num_workers', min(batch_size, max(os.cpu_count() - 1, 1))),
                'pin_memory': True,
                'channel_selector': config.get('channel_selector', None),
                'use_start_end_token': self.model.cfg.validation_ds.get('use_start_end_token', False),
            }
            return self.model._setup_dataloader_from_config(config=DictConfig(dl_config))

        self.model._setup_transcribe_dataloader = _patched_setup_transcribe_dataloader

        print(f"Model loaded successfully on {self.device}")

    @staticmethod
    def _select_device(device: str) -> str:
        device = device.strip().lower()
        if device not in {"cuda", "cpu"}:
            raise ValueError("device must be 'cuda' or 'cpu'")

        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
            return "cpu"
        return device

    # ── Core transcription ──────────────────────────────────────────────

    def transcribe_audio(self, wav_path: Path) -> Dict:
        """
        Transcribe a WAV file using the local NeMo model.

        Returns:
            Dict with 'text' and 'segments' keys.
        """
        print(f"  Transcribing locally (model={self.model_name})...")

        with torch.no_grad():
            result = self.model.transcribe([str(wav_path)])

        # NeMo CTC models return a list of strings
        if isinstance(result, list):
            text = result[0] if result else ""
        else:
            text = str(result)

        # CTC models don't produce segment-level timestamps
        return {"text": text, "segments": []}

    # ── Single video processing ─────────────────────────────────────────

    def transcribe_video(
        self,
        video_path: str | Path,
        output_dir: str = "processed",
    ) -> Dict:
        """
        Process a single video file:
        1) Check for audio stream
        2) Convert to WAV via ffmpeg
        3) Transcribe locally via NeMo
        4) Save JSON + text transcript

        Returns:
            Transcription result dict.
        """
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"File not found: {video_path}")

        ext = video_path.suffix.lower()
        if ext not in VIDEO_EXTS and ext not in AUDIO_EXTS:
            raise ValueError(
                f"Unsupported file extension '{ext}'.\n"
                f"Supported video: {sorted(VIDEO_EXTS)}\n"
                f"Supported audio: {sorted(AUDIO_EXTS)}"
            )

        video_name = video_path.stem
        transcript_dir = Path(output_dir) / "transcripts" / self.MODEL_DISPLAY_NAME / video_name
        transcript_dir.mkdir(parents=True, exist_ok=True)

        # Check if already processed
        json_file = transcript_dir / "transcript.json"
        if json_file.exists():
            print(f"  ⏭ Already processed (found {json_file}), skipping.")
            with open(json_file, "r", encoding="utf-8") as f:
                return json.load(f)

        # ── Check for audio stream ──────────────────────────────────────
        if ext in VIDEO_EXTS and not has_audio_stream(video_path):
            print(f"  ⚠ No audio stream found in '{video_path.name}', skipping.")
            result = {
                "text": "",
                "segments": [],
                "processing_time_seconds": 0,
                "model": self.model_name,
                "source_file": video_path.name,
                "note": "No audio stream detected in video",
            }
            self._save_results(result, transcript_dir, video_name)
            return result

        # ── Step 1: Convert to WAV ──────────────────────────────────────
        temp_dir = Path(tempfile.mkdtemp(prefix="nvidia_asr_"))
        wav_path = temp_dir / f"{video_name}_16k_mono.wav"

        needs_conversion = ext in VIDEO_EXTS or ext != ".wav"
        if needs_conversion:
            print(f"  Converting to WAV (16kHz mono)...")
            run_ffmpeg_extract_to_wav(video_path, wav_path, sample_rate=16000)
        else:
            wav_path = video_path

        # ── Step 2: Transcribe ──────────────────────────────────────────
        start_time = time.time()
        try:
            result = self.transcribe_audio(wav_path)
        finally:
            # Clean up temp files
            if needs_conversion:
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass

        duration = time.time() - start_time
        result["processing_time_seconds"] = round(duration, 2)
        result["model"] = self.model_name
        result["source_file"] = video_path.name

        # ── Step 3: Save results ────────────────────────────────────────
        self._save_results(result, transcript_dir, video_name)

        print(f"  ✓ Transcribed in {duration:.1f}s — {len(result.get('text', ''))} chars")
        return result

    # ── Batch processing ────────────────────────────────────────────────

    def batch_transcribe(
        self,
        folder_path: str = "videos",
        output_dir: str = "processed",
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """
        Transcribe all supported video/audio files in a folder.

        Args:
            folder_path: Directory containing video/audio files
            output_dir: Base output directory
            limit: Max number of files to process (None = all)

        Returns:
            List of result summaries.
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Video folder not found: {folder}")

        extensions = VIDEO_EXTS | AUDIO_EXTS
        files = sorted(
            [f for f in folder.glob("*.*") if f.suffix.lower() in extensions]
        )

        if limit is not None:
            files = files[:limit]

        print(f"\n{'=' * 60}")
        print(f"NVIDIA ASR LOCAL TRANSCRIPTION — {self.model_name}")
        print(f"Device: {self.device}")
        print(f"{'=' * 60}")
        print(f"Found {len(files)} files to transcribe")
        print(f"Output: {Path(output_dir) / 'transcripts' / self.MODEL_DISPLAY_NAME}")
        print(f"{'=' * 60}\n")

        summaries: List[Dict] = []
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
            try:
                result = self.transcribe_video(file_path, output_dir=output_dir)
                summaries.append({
                    "file": file_path.name,
                    "model": self.model_name,
                    "success": True,
                    "transcript_length": len(result.get("text", "")),
                    "processing_time": result.get("processing_time_seconds", 0),
                })
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                import traceback
                traceback.print_exc()
                summaries.append({
                    "file": file_path.name,
                    "model": self.model_name,
                    "success": False,
                    "error": str(e),
                })

        # Print summary
        successful = sum(1 for s in summaries if s["success"])
        print(f"\n{'=' * 60}")
        print("BATCH COMPLETE")
        print(f"{'=' * 60}")
        print(f"Successful: {successful}/{len(files)}")
        print(f"Failed:     {len(files) - successful}/{len(files)}")

        return summaries

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _save_results(result: Dict, transcript_dir: Path, video_name: str) -> None:
        """Save transcription as JSON and plain-text files."""
        # JSON transcript
        json_file = transcript_dir / "transcript.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Plain-text transcript with timestamps
        txt_file = transcript_dir / "transcript.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(f"Transcript: {video_name}\n")
            f.write(f"Model: {result.get('model', 'unknown')}\n")
            f.write(f"{'=' * 50}\n\n")

            segments = result.get("segments", [])
            if segments:
                for seg in segments:
                    start = str(timedelta(seconds=seg.get("start", 0)))
                    end = str(timedelta(seconds=seg.get("end", 0)))
                    f.write(f"[{start} → {end}] {seg.get('text', '').strip()}\n")
            else:
                f.write(result.get("text", ""))
            f.write("\n")


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe videos using NVIDIA Parakeet models locally (NeMo)."
    )
    parser.add_argument(
        "--video-folder",
        default="videos",
        help="Folder containing video/audio files (default: videos)",
    )
    parser.add_argument(
        "--model",
        default="parakeet-ctc-1.1b",
        choices=AVAILABLE_MODELS,
        help="NVIDIA ASR model to use (default: parakeet-ctc-1.1b)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device (default: cuda)",
    )
    parser.add_argument(
        "--output-dir",
        default="processed",
        help="Base output directory (default: processed)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of files to process",
    )
    parser.add_argument(
        "--single",
        default=None,
        help="Process a single video file instead of a folder",
    )

    args = parser.parse_args()

    transcriber = NvidiaTranscriber(
        model=args.model,
        device=args.device,
    )

    if args.single:
        transcriber.transcribe_video(args.single, output_dir=args.output_dir)
    else:
        transcriber.batch_transcribe(
            folder_path=args.video_folder,
            output_dir=args.output_dir,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()