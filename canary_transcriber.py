"""
canary_video_transcriber.py

Transcribe VIDEO files using NVIDIA Canary-1B (NeMo) on Windows/Linux.

What it does:
1) Accepts a video (mp4/mkv/mov/avi/webm) or an audio file
2) Extracts + converts audio to 16kHz mono WAV (Canary-friendly) using ffmpeg
3) Runs Canary-1B transcription via EncDecMultiTaskModel

Requirements:
- ffmpeg installed and available on PATH:
    ffmpeg -version
- Python packages:
    pip install nemo_toolkit[asr] torch torchaudio soundfile huggingface_hub

Run:
    python canary_video_transcriber.py --input videos/my_video.mp4 --device cuda
    python canary_video_transcriber.py --input videos/my_video.mp4 --device cpu
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, List

import torch

# NeMo Canary model
from nemo.collections.asr.models import EncDecMultiTaskModel


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".opus"}


def ensure_ffmpeg_available() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg was not found on PATH.\n"
            "Install ffmpeg and ensure 'ffmpeg -version' works in your terminal.\n"
            "Windows tip: download ffmpeg, add its 'bin' folder to PATH."
        )
    return ffmpeg_path


def run_ffmpeg_extract_to_wav(
    input_path: Path,
    output_wav: Path,
    sample_rate: int = 16000,
) -> None:
    """
    Convert input media to a mono PCM WAV at sample_rate.
    This is a safe format for NeMo ASR.
    """
    ffmpeg = ensure_ffmpeg_available()

    # -vn : ignore video
    # -ac 1 : mono
    # -ar 16000 : 16kHz
    # -c:a pcm_s16le : 16-bit PCM WAV
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
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


class CanaryVideoTranscriber:
    def __init__(
        self,
        device: str = "cuda",
        model_path: str = "nvidia/canary-1b",
    ) -> None:
        """
        device: "cuda" or "cpu"
        model_path: HF repo id (nvidia/canary-1b) or a local .nemo file path
        """
        self.device = self._select_device(device)
        self.model = self._load_canary(model_path, self.device)

    @staticmethod
    def _select_device(device: str) -> str:
        device = device.strip().lower()
        if device not in {"cuda", "cpu"}:
            raise ValueError("device must be 'cuda' or 'cpu'")

        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
            return "cpu"
        return device

    @staticmethod
    def _load_canary(model_path: str, device: str) -> EncDecMultiTaskModel:
        """
        Loads Canary-1B. Works with:
        - HF repo id: "nvidia/canary-1b"
        - Local .nemo file path: "C:/.../canary-1b.nemo"
        """
        print(f"Loading Canary model '{model_path}' on {device}...")

        # If user passed a local path to a .nemo file:
        if model_path.endswith(".nemo") and Path(model_path).exists():
            model = EncDecMultiTaskModel.restore_from(model_path, map_location=device)
        else:
            # HF-style: fetch + restore (NeMo supports hf:// URIs in many installs,
            # but simplest is to rely on built-in restore_from with HF cache path if present).
            #
            # Many setups allow:
            #   EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b")
            #
            # We'll try from_pretrained first, then give a helpful error if not.
            try:
                model = EncDecMultiTaskModel.from_pretrained(model_name=model_path, map_location=device)
            except Exception as e:
                raise RuntimeError(
                    "Failed to load model via from_pretrained().\n"
                    "If this fails in your environment, download the .nemo file and pass it via --model.\n"
                    f"Original error: {e}"
                ) from e

        model.eval()
        return model

    def transcribe_media(
        self,
        input_media: str | Path,
        keep_temp_wav: bool = False,
        batch_size: int = 1,
    ) -> str:
        """
        Accepts video or audio input path. Returns transcription text.
        """
        input_path = Path(input_media).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        ext = input_path.suffix.lower()
        if ext not in VIDEO_EXTS and ext not in AUDIO_EXTS:
            raise ValueError(
                f"Unsupported extension '{ext}'.\n"
                f"Supported video: {sorted(VIDEO_EXTS)}\n"
                f"Supported audio: {sorted(AUDIO_EXTS)}"
            )

        temp_dir = Path(tempfile.mkdtemp(prefix="canary_"))
        wav_path = temp_dir / f"{input_path.stem}_16k_mono.wav"

        print(f"Converting to WAV (16kHz mono): {wav_path}")
        run_ffmpeg_extract_to_wav(input_path, wav_path, sample_rate=16000)

        print("Running Canary transcription...")
        with torch.no_grad():
            # IMPORTANT: Canary's EncDecMultiTaskModel.transcribe expects `audio=...`
            # not `paths2audio_files=...`
            out: List[str] = self.model.transcribe(
                audio=[str(wav_path)],
                batch_size=batch_size,
            )

        text = out[0] if out else ""

        if keep_temp_wav:
            print(f"Keeping temp WAV at: {wav_path}")
        else:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

        return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe video/audio using NVIDIA Canary-1B (NeMo).")
    parser.add_argument("--input", required=True, help="Path to a video or audio file")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument(
        "--model",
        default="nvidia/canary-1b",
        help="HF model id (default: nvidia/canary-1b) OR local .nemo path",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for transcription")
    parser.add_argument("--keep_wav", action="store_true", help="Keep the temporary converted WAV")

    args = parser.parse_args()

    transcriber = CanaryVideoTranscriber(device=args.device, model_path=args.model)
    text = transcriber.transcribe_media(args.input, keep_temp_wav=args.keep_wav, batch_size=args.batch_size)

    print("\n===== TRANSCRIPTION =====")
    print(text.strip())


if __name__ == "__main__":
    main()
