"""
Benchmarking Transcriber
========================
Runs multiple ASR models over all videos in the `videos/` folder and saves
structured results to:  benchmarking/<model_name>/<video_name>/transcript.txt + .json

Supported Models:
  1. Vosk             – Offline Kaldi-based ASR
  2. SpeechT5 ASR     – microsoft/speecht5_asr (HuggingFace)
  3. Voxtral Mini 4B  – mistralai/Voxtral-Mini-4B-Realtime-2602 (HuggingFace)
  4. pyannote         – pyannote/speaker-diarization-community-1 (diarization)
  5. MedASR           – google/medasr (medical ASR, HuggingFace)
  6. VibeVoice ASR    – microsoft/VibeVoice-ASR (HuggingFace)

Usage:
  python benchmark_transcriber.py
  python benchmark_transcriber.py --videos-dir videos/ --models vosk speecht5 voxtral
  python benchmark_transcriber.py --models medasr vibevoice --device cuda
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
import warnings
from datetime import timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


import torch
import numpy as np

warnings.filterwarnings("ignore")


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".ts"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
ALL_MEDIA = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

MODEL_REGISTRY = [
    "vosk",
    "speecht5",
    "voxtral",
    "pyannote",
    "medasr",
    "vibevoice",
]


# -------------------------------------------------------------
# Audio helpers
# -------------------------------------------------------------

def extract_audio_to_wav(file_path: Path, target_sr: int = 16000) -> Path:
    """
    Extract/convert any media file to a 16 kHz mono WAV using ffmpeg.
    Returns path to the temporary WAV file.
    """
    wav_path = file_path.parent / f"_benchmark_temp_{file_path.stem}.wav"
    if wav_path.exists():
        wav_path.unlink()

    # Try imageio-ffmpeg first, then system ffmpeg
    ffmpeg_cmd = "ffmpeg"
    try:
        import imageio_ffmpeg
        ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    cmd = [
        ffmpeg_cmd, "-y", "-i", str(file_path),
        "-ac", "1",             # mono
        "-ar", str(target_sr),  # 16 kHz
        "-sample_fmt", "s16",   # 16-bit PCM
        "-vn",                  # no video
        str(wav_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")

    return wav_path


def load_audio_array(wav_path: Path, target_sr: int = 16000):
    """Load a WAV file as a numpy float32 array at the target sample rate."""
    try:
        import librosa
        audio, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)
        return audio, sr
    except ImportError:
        pass

    try:
        import soundfile as sf
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            import torchaudio.transforms as T
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio_t = T.Resample(sr, target_sr)(audio_t)
            audio = audio_t.squeeze(0).numpy()
            sr = target_sr
        return audio, sr
    except ImportError:
        raise ImportError("Please install librosa or soundfile to load audio.")


# -------------------------------------------------------------
# Base class
# -------------------------------------------------------------

class BaseTranscriber:
    """Base class all benchmark transcribers inherit from."""

    name: str = "base"
    requires_gpu: bool = False

    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"  [{self.name}] Using device: {self.device}")

    def transcribe(self, wav_path: Path) -> dict:
        """
        Transcribe a WAV file. Must return a dict with at least:
          {"text": str, "segments": list[dict]}
        Each segment: {"start": float, "end": float, "text": str}
        """
        raise NotImplementedError

    def cleanup(self):
        """Optional cleanup (free GPU memory, etc.)."""
        pass


# -------------------------------------------------------------
# 1. Vosk
# -------------------------------------------------------------

class VoskTranscriber(BaseTranscriber):
    name = "Vosk"
    requires_gpu = False

    def __init__(self, device="auto", model_path=None):
        super().__init__(device)
        from vosk import Model, KaldiRecognizer, SetLogLevel
        SetLogLevel(-1)  # Suppress Vosk logs

        if model_path and os.path.isdir(model_path):
            print(f"  [{self.name}] Loading model from: {model_path}")
            self.model = Model(model_path)
        else:
            print(f"  [{self.name}] Loading default English model (auto-download)...")
            self.model = Model(lang="en-us")

        self.KaldiRecognizer = KaldiRecognizer

    def transcribe(self, wav_path: Path) -> dict:
        import wave

        wf = wave.open(str(wav_path), "rb")
        assert wf.getnchannels() == 1, "Vosk requires mono audio"
        assert wf.getsampwidth() == 2, "Vosk requires 16-bit audio"

        sample_rate = wf.getframerate()

        # Also parse partial results with word timings
        rec = self.KaldiRecognizer(self.model, sample_rate)
        rec.SetWords(True)

        results_list = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                part = json.loads(rec.Result())
                if part.get("text"):
                    results_list.append(part)

        last_part = json.loads(rec.FinalResult())
        if last_part.get("text"):
            results_list.append(last_part)

        wf.close()

        # Build segments from results
        segments = []
        all_text_parts = []
        for r in results_list:
            text = r.get("text", "")
            if text:
                all_text_parts.append(text)
                words = r.get("result", [])
                if words:
                    start = words[0].get("start", 0.0)
                    end = words[-1].get("end", 0.0)
                else:
                    start = 0.0
                    end = 0.0
                segments.append({"start": start, "end": end, "text": text})

        full_text = " ".join(all_text_parts)
        return {"text": full_text, "segments": segments}


# -------------------------------------------------------------
# 2. SpeechT5 ASR
# -------------------------------------------------------------

class SpeechT5Transcriber(BaseTranscriber):
    name = "SpeechT5-ASR"
    requires_gpu = False

    def __init__(self, device="auto"):
        super().__init__(device)
        from transformers import pipeline as hf_pipeline
        print(f"  [{self.name}] Loading microsoft/speecht5_asr ...")
        self.pipe = hf_pipeline(
            "automatic-speech-recognition",
            model="microsoft/speecht5_asr",
            device=0 if self.device == "cuda" else -1,
        )
        print(f"  [{self.name}] Model loaded.")

    def transcribe(self, wav_path: Path) -> dict:
        audio, sr = load_audio_array(wav_path, target_sr=16000)

        result = self.pipe(
            audio,
            chunk_length_s=20,
            stride_length_s=2,
        )

        text = result.get("text", "")
        chunks = result.get("chunks", [])

        segments = []
        for c in chunks:
            ts = c.get("timestamp", (0, 0))
            segments.append({
                "start": ts[0] if ts[0] is not None else 0.0,
                "end": ts[1] if ts[1] is not None else 0.0,
                "text": c.get("text", ""),
            })

        if not segments and text:
            segments = [{"start": 0.0, "end": 0.0, "text": text}]

        return {"text": text, "segments": segments}


# -------------------------------------------------------------
# 3. Voxtral Mini 4B Realtime
# -------------------------------------------------------------

class VoxtralTranscriber(BaseTranscriber):
    name = "Voxtral-Mini-4B"
    requires_gpu = True

    def __init__(self, device="auto"):
        super().__init__(device)
        from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
        from mistral_common.tokens.tokenizers.audio import Audio

        self.Audio = Audio
        repo_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"
        print(f"  [{self.name}] Loading {repo_id} ...")

        self.processor = AutoProcessor.from_pretrained(repo_id)
        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            repo_id,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        print(f"  [{self.name}] Model loaded.")

    def transcribe(self, wav_path: Path) -> dict:
        audio = self.Audio.from_file(str(wav_path), strict=False)
        audio.resample(self.processor.feature_extractor.sampling_rate)

        inputs = self.processor(audio.audio_array, return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=4096)

        text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return {"text": text.strip(), "segments": [{"start": 0.0, "end": 0.0, "text": text.strip()}]}

    def cleanup(self):
        if hasattr(self, "model"):
            del self.model
        torch.cuda.empty_cache()


# -------------------------------------------------------------
# 4. pyannote Speaker Diarization
# -------------------------------------------------------------

class PyannoteTranscriber(BaseTranscriber):
    """
    Speaker diarization (who speaks when) — not ASR.
    Outputs speaker segments with timing, no text transcription.
    """
    name = "Pyannote-Diarization"
    requires_gpu = True

    def __init__(self, device="auto", hf_token=None):
        super().__init__(device)
        from pyannote.audio import Pipeline as PyannotePipeline

        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        print(f"  [{self.name}] Loading pyannote/speaker-diarization-community-1 ...")
        self.pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            use_auth_token=token,
        )

        if self.device == "cuda":
            self.pipeline.to(torch.device("cuda"))

        print(f"  [{self.name}] Pipeline loaded.")

    def transcribe(self, wav_path: Path) -> dict:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(wav_path))
        output = self.pipeline({"waveform": waveform, "sample_rate": sample_rate})

        segments = []
        full_text_parts = []

        for turn, _, speaker in output.itertracks(yield_label=True):
            segment_text = f"[{speaker}]"
            segments.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "text": segment_text,
                "speaker": speaker,
                "duration": round(turn.end - turn.start, 3),
            })
            full_text_parts.append(
                f"[{speaker}: {turn.start:.1f}s - {turn.end:.1f}s]"
            )

        full_text = " ".join(full_text_parts)
        return {
            "text": full_text,
            "segments": segments,
            "type": "diarization",
            "num_speakers": len(set(s["speaker"] for s in segments)),
        }

    def cleanup(self):
        del self.pipeline
        torch.cuda.empty_cache()


# -------------------------------------------------------------
# 5. Google MedASR
# -------------------------------------------------------------

class MedASRTranscriber(BaseTranscriber):
    name = "Google-MedASR"
    requires_gpu = True

    def __init__(self, device="auto"):
        super().__init__(device)
        from transformers import pipeline as hf_pipeline

        model_id = "google/medasr"
        print(f"  [{self.name}] Loading {model_id} ...")
        self.pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=0 if self.device == "cuda" else -1,
        )
        print(f"  [{self.name}] Model loaded.")

    def transcribe(self, wav_path: Path) -> dict:
        result = self.pipe(
            str(wav_path),
            chunk_length_s=20,
            stride_length_s=2,
        )

        text = result.get("text", "")
        chunks = result.get("chunks", [])

        segments = []
        for c in chunks:
            ts = c.get("timestamp", (0, 0))
            segments.append({
                "start": ts[0] if ts[0] is not None else 0.0,
                "end": ts[1] if ts[1] is not None else 0.0,
                "text": c.get("text", ""),
            })

        if not segments and text:
            segments = [{"start": 0.0, "end": 0.0, "text": text}]

        return {"text": text, "segments": segments}

    def cleanup(self):
        del self.pipe
        torch.cuda.empty_cache()


# -------------------------------------------------------------
# 6. Microsoft VibeVoice ASR
# -------------------------------------------------------------

class VibeVoiceTranscriber(BaseTranscriber):
    """
    VibeVoice-ASR: 60-min single-pass ASR + diarization + timestamps.
    Requires: pip install git+https://github.com/microsoft/VibeVoice.git
    """
    name = "VibeVoice-ASR"
    requires_gpu = True

    def __init__(self, device="auto"):
        super().__init__(device)
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_id = "microsoft/VibeVoice-ASR"
        print(f"  [{self.name}] Loading {model_id} ...")

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")

        print(f"  [{self.name}] Model loaded.")

    def transcribe(self, wav_path: Path) -> dict:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(wav_path))

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000

        inputs = self.processor(
            waveform.squeeze(0).numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=8192)

        text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        segments = self._parse_vibevoice_output(text)

        return {
            "text": text.strip(),
            "segments": segments,
        }

    def _parse_vibevoice_output(self, text: str) -> list:
        """
        Parse VibeVoice's structured output into segments.
        VibeVoice outputs lines like: [Speaker_0] <0.0s - 5.2s> Hello world
        """
        import re
        segments = []
        pattern = r'\[([^\]]+)\]\s*<([\d.]+)s?\s*-\s*([\d.]+)s?>\s*(.*?)(?=\[|$)'
        matches = re.finditer(pattern, text, re.DOTALL)

        for m in matches:
            speaker = m.group(1)
            start = float(m.group(2))
            end = float(m.group(3))
            seg_text = m.group(4).strip()
            segments.append({
                "start": start,
                "end": end,
                "text": seg_text,
                "speaker": speaker,
            })

        if not segments:
            segments = [{"start": 0.0, "end": 0.0, "text": text.strip()}]

        return segments

    def cleanup(self):
        del self.model
        del self.processor
        torch.cuda.empty_cache()


# -------------------------------------------------------------
# Model Factory
# -------------------------------------------------------------

def create_transcriber(model_name: str, device: str = "auto", **kwargs):
    """Instantiate a transcriber by name. Returns None if unavailable."""
    constructors = {
        "vosk":      (VoskTranscriber,      ["vosk"]),
        "speecht5":  (SpeechT5Transcriber,  ["transformers"]),
        "voxtral":   (VoxtralTranscriber,   ["transformers", "mistral_common"]),
        "pyannote":  (PyannoteTranscriber,  ["pyannote.audio"]),
        "medasr":    (MedASRTranscriber,    ["transformers"]),
        "vibevoice": (VibeVoiceTranscriber, ["transformers"]),
    }

    if model_name not in constructors:
        print(f"✗ Unknown model: {model_name}")
        return None

    cls, required_pkgs = constructors[model_name]

    # Check dependencies
    for pkg in required_pkgs:
        try:
            __import__(pkg)
        except ImportError:
            print(f"✗ [{model_name}] Missing dependency: {pkg}  (pip install {pkg})")
            return None

    try:
        return cls(device=device, **kwargs)
    except Exception as e:
        print(f"✗ [{model_name}] Failed to initialize: {e}")
        traceback.print_exc()
        return None


# -------------------------------------------------------------
# Result saving
# -------------------------------------------------------------

def save_results(result: dict, output_dir: Path, video_name: str, model_name: str,
                 elapsed: float, source_file: str):
    """Save transcript as JSON and TXT."""
    output_dir.mkdir(parents=True, exist_ok=True)

    result["model"] = model_name
    result["source_file"] = source_file
    result["processing_time_seconds"] = round(elapsed, 2)
    result["num_segments"] = len(result.get("segments", []))

    # JSON
    json_path = output_dir / "transcript.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # TXT
    txt_path = output_dir / "transcript.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Source: {source_file}\n")
        f.write(f"Processing Time: {elapsed:.2f}s\n")
        f.write("=" * 60 + "\n\n")

        if result.get("type") == "diarization":
            f.write("SPEAKER DIARIZATION RESULTS\n")
            f.write(f"Number of speakers detected: {result.get('num_speakers', 'N/A')}\n\n")

        for seg in result.get("segments", []):
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker", "")

            start_str = str(timedelta(seconds=start)).split(".")[0]
            end_str = str(timedelta(seconds=end)).split(".")[0]

            if speaker:
                f.write(f"[{start_str} → {end_str}] ({speaker}) {text}\n")
            else:
                f.write(f"[{start_str} → {end_str}] {text}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("FULL TEXT:\n")
        f.write(result.get("text", "(empty)") + "\n")

    print(f"    ✓ Saved: {output_dir}")
    return json_path


# -------------------------------------------------------------
# Main benchmark runner
# -------------------------------------------------------------

def run_benchmark(
    videos_dir: str = "videos",
    output_base: str = "benchmarking",
    models: list = None,
    device: str = "auto",
    hf_token: str = None,
):
    """
    Run all (or selected) ASR models over every video in the folder.
    """
    videos_dir = Path(videos_dir)
    output_base = Path(output_base)

    if not videos_dir.exists():
        print(f"✗ Videos directory not found: {videos_dir}")
        return

    files = sorted([
        f for f in videos_dir.iterdir()
        if f.is_file() and f.suffix.lower() in ALL_MEDIA
    ])

    if not files:
        print(f"✗ No video/audio files found in {videos_dir}")
        return

    selected_models = models or MODEL_REGISTRY
    selected_models = [m.lower() for m in selected_models]

    print("\n" + "=" * 70)
    print("  BENCHMARKING TRANSCRIBER")
    print("=" * 70)
    print(f"  Videos directory : {videos_dir}")
    print(f"  Output directory : {output_base}")
    print(f"  Files found      : {len(files)}")
    print(f"  Models selected  : {', '.join(selected_models)}")
    print(f"  Device           : {device}")
    print("=" * 70 + "\n")

    summary_rows = []

    for model_key in selected_models:
        print(f"\n{'─' * 60}")
        print(f"  MODEL: {model_key.upper()}")
        print(f"{'─' * 60}")

        kwargs = {}
        if model_key == "pyannote":
            kwargs["hf_token"] = hf_token

        transcriber = create_transcriber(model_key, device=device, **kwargs)

        if transcriber is None:
            print(f"  ⚠ Skipping {model_key} (not available)")
            for f in files:
                summary_rows.append({
                    "model": model_key,
                    "video": f.name,
                    "status": "SKIPPED",
                    "time_seconds": 0,
                    "text_length": 0,
                    "num_segments": 0,
                    "error": "Model not available",
                })
            continue

        for i, file_path in enumerate(files, 1):
            video_name = file_path.stem.replace(" ", "_")
            output_dir = output_base / transcriber.name / video_name

            print(f"\n  [{i}/{len(files)}] {file_path.name}")

            wav_path = None
            start_time = time.time()
            try:
                wav_path = extract_audio_to_wav(file_path, target_sr=16000)
                
                # Transcription
                result = transcriber.transcribe(wav_path)
                elapsed = time.time() - start_time
                
                print(f"    ⏱ {elapsed:.2f}s | {len(result.get('text', ''))} chars | {len(result.get('segments', []))} segments")

                # Save results
                save_results(result, output_dir, video_name, transcriber.name, elapsed, file_path.name)

                summary_rows.append({
                    "model": transcriber.name,
                    "video": file_path.name,
                    "status": "SUCCESS",
                    "time_seconds": round(elapsed, 2),
                    "text_length": len(result.get("text", "")),
                    "num_segments": len(result.get("segments", [])),
                    "error": "",
                })

            except Exception as e:
                elapsed = time.time() - start_time if 'start_time' in locals() else 0
                print(f"    ✗ Failed processing {file_path.name}: {e}")
                traceback.print_exc()

                summary_rows.append({
                    "model": model_key,
                    "video": file_path.name,
                    "status": "FAILED",
                    "time_seconds": round(elapsed, 2),
                    "text_length": 0,
                    "num_segments": 0,
                    "error": str(e)[:200],
                })

            finally:
                if wav_path and wav_path.exists() and "_benchmark_temp_" in wav_path.name:
                    try:
                        wav_path.unlink()
                    except Exception:
                        pass

        # Cleanup model
        try:
            transcriber.cleanup()
        except Exception:
            pass
        del transcriber
        torch.cuda.empty_cache()

    # ── Save summary CSV ──
    summary_path = output_base / "benchmark_summary.csv"
    output_base.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "video", "status", "time_seconds", "text_length", "num_segments", "error"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    # ── Print summary ──
    print(f"\n\n{'=' * 70}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 70}")

    model_stats = {}
    for row in summary_rows:
        model = row["model"]
        if model not in model_stats:
            model_stats[model] = {"success": 0, "failed": 0, "skipped": 0, "total_time": 0}
        if row["status"] == "SUCCESS":
            model_stats[model]["success"] += 1
            model_stats[model]["total_time"] += row["time_seconds"]
        elif row["status"] == "FAILED":
            model_stats[model]["failed"] += 1
        else:
            model_stats[model]["skipped"] += 1

    print(f"\n  {'Model':<25} {'Success':>8} {'Failed':>8} {'Skipped':>8} {'Total Time':>12}")
    print(f"  {'─' * 65}")
    for model, stats in model_stats.items():
        total_time = f"{stats['total_time']:.1f}s"
        print(f"  {model:<25} {stats['success']:>8} {stats['failed']:>8} {stats['skipped']:>8} {total_time:>12}")

    print(f"\n  Summary CSV: {summary_path}")
    print(f"  Results in : {output_base}/")
    print(f"{'=' * 70}\n")


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmarking Transcriber – Run multiple ASR models over videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  vosk        Vosk (Kaldi-based, offline, CPU-friendly)
  speecht5    SpeechT5 ASR (microsoft/speecht5_asr)
  voxtral     Voxtral Mini 4B (mistralai/Voxtral-Mini-4B-Realtime-2602)
  pyannote    Speaker Diarization (pyannote/speaker-diarization-community-1)
  medasr      Google MedASR (google/medasr)
  vibevoice   VibeVoice ASR (microsoft/VibeVoice-ASR)

Examples:
  python benchmark_transcriber.py
  python benchmark_transcriber.py --models vosk speecht5
  python benchmark_transcriber.py --videos-dir videos/ --device cuda
  python benchmark_transcriber.py --models voxtral vibevoice --device cuda
""",
    )

    parser.add_argument(
        "--videos-dir", type=str, default="videos",
        help="Directory containing video/audio files (default: videos/)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarking",
        help="Base output directory (default: benchmarking/)"
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_REGISTRY, default=None,
        help="Specific models to run (default: all)"
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="Device for models (default: auto)"
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace token (needed for pyannote gated models)"
    )

    args = parser.parse_args()

    run_benchmark(
        videos_dir=args.videos_dir,
        output_base=args.output_dir,
        models=args.models,
        device=args.device,
        hf_token=args.hf_token,
    )