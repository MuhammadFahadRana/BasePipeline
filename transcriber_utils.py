import os
import json
import subprocess
import time
import torch
import numpy as np
import warnings
from pathlib import Path
from datetime import timedelta

warnings.filterwarnings("ignore")

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".ts"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
ALL_MEDIA = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

def extract_audio_to_wav(file_path: Path, target_sr: int = 16000) -> Path:
    """Extract any media file to a 16 kHz mono WAV using ffmpeg."""
    wav_path = file_path.parent / f"_temp_{file_path.stem}.wav"
    if wav_path.exists():
        wav_path.unlink()

    ffmpeg_cmd = "ffmpeg"
    try:
        import imageio_ffmpeg
        ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    # Ensure ffmpeg in path for some libraries
    ffmpeg_dir = os.path.dirname(ffmpeg_cmd)
    if ffmpeg_dir and ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

    cmd = [
        ffmpeg_cmd, "-y", "-i", str(file_path),
        "-ac", "1", "-ar", str(target_sr), "-sample_fmt", "s16", "-vn", str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")
    return wav_path

def load_audio_array(wav_path: Path, target_sr: int = 16000):
    """Load a WAV file as a numpy float32 array."""
    try:
        import soundfile as sf
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio, target_sr
    except ImportError:
        import librosa
        audio, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)
        return audio, sr

def save_results(result: dict, output_dir: Path, video_name: str, model_name: str, elapsed: float, source_file: str):
    """Save transcript results as JSON and TXT."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result.update({
        "model": model_name,
        "source_file": source_file,
        "processing_time_seconds": round(elapsed, 2),
        "num_segments": len(result.get("segments", []))
    })

    with open(output_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    with open(output_dir / "transcript.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\nSource: {source_file}\nTime: {elapsed:.2f}s\n" + "="*60 + "\n\n")
        for seg in result.get("segments", []):
            s, e, t = seg.get("start",0), seg.get("end",0), seg.get("text","")
            spk = f"({seg['speaker']}) " if 'speaker' in seg else ""
            f.write(f"[{str(timedelta(seconds=s)).split('.')[0]} -> {str(timedelta(seconds=e)).split('.')[0]}] {spk}{t}\n")
        f.write("\n" + "="*60 + "\nFULL TEXT:\n" + result.get("text", "") + "\n")

def hf_auth():
    """Load .env and authenticate with Hugging Face Hub."""
    # Check for offline mode override
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        print("Note: Hugging Face Offline Mode is active.")
        return None

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    if token:
        print(f"Applying HF_TOKEN from .env...")
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=True)
            return token
        except Exception as e:
            print(f"Warning: Hugging Face login failed with .env token: {e}")
    else:
        print("Note: HF_TOKEN not found in .env. Using cached credentials from 'hf auth login' if any.")
    
    return token

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device
