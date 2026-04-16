"""
Cross-validation using NVIDIA NeMo models (Canary, Parakeet, etc.)
Validates Whisper/WhisperX/Distil-Whisper transcriptions using a different architecture.

- Supports multiple NeMo variants (Canary, Parakeet CTC/RNNT)
- Caches validator transcripts under: processed/transcripts/Nvidia-{variant}/<video>/transcript.json
- Produces validation results under: processed/validation/<ASRModel>_vs_<NvidiaVariant>/
"""

import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import difflib
import numpy as np

import torch
import torchaudio
import soundfile as sf

# NeMo (Parakeet)
import nemo.collections.asr as nemo_asr


# -------------------------
# Utilities
# -------------------------
def prepare_ffmpeg():
    
    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)

        if os.name == "nt":
            ffmpeg_name = os.path.basename(ffmpeg_exe)
            if ffmpeg_name != "ffmpeg.exe":
                dest = os.path.join(ffmpeg_dir, "ffmpeg.exe")
                if not os.path.exists(dest):
                    try:
                        shutil.copy(ffmpeg_exe, dest)
                    except Exception:
                        pass

        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        return True
    except ImportError:
        return False


def normalize_audio_to_16k_mono_wav(src_path: Path, dst_path: Path) -> Path:
    """
    Convert any readable audio file to a 16kHz mono WAV.
    This makes NeMo transcription more reliable across formats.
    """
    # Try soundfile first (fast, reliable on Windows)
    try:
        data, sr = sf.read(str(src_path))
        if data.ndim == 1:
            waveform = torch.from_numpy(data).float().unsqueeze(0)
        else:
            # [time, channels] -> [channels, time]
            waveform = torch.from_numpy(data.T).float()
    except Exception:
        waveform, sr = torchaudio.load(str(src_path))

    # mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dst_path), waveform, 16000)
    return dst_path


prepare_ffmpeg()
warnings.filterwarnings("ignore")


# -------------------------
# Nvidia Validator
# -------------------------
class NvidiaValidator:
    """
    Validate ASR transcriptions using NVIDIA NeMo models (Canary, Parakeet) as an independent check.

    Example variants:
      - nvidia/canary-1b
      - nvidia/parakeet-ctc-1.1b
    """

    AVAILABLE_VARIANTS = {
        "Canary-1B": "nvidia/canary-1b",
        "Parakeet-CTC-1.1B": "nvidia/parakeet-ctc-1.1b",
        "Parakeet-RNNT-1.1B": "nvidia/parakeet-rnnt-1.1b",
    }

    def __init__(self, variant_key: str = "Canary-1B", device: str = "auto"):
        """
        Args:
            variant_key: One of AVAILABLE_VARIANTS keys (e.g., "Canary-1B")
            device: "cuda", "cpu", or "auto"
        """
        if variant_key not in self.AVAILABLE_VARIANTS:
            raise ValueError(
                f"Unknown variant_key='{variant_key}'. "
                f"Choose from: {list(self.AVAILABLE_VARIANTS.keys())}"
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.variant_key = variant_key
        self.model_name = self.AVAILABLE_VARIANTS[variant_key]

        print(f"Loading Nvidia model: {self.model_name}")
        
        # Use specific model class based on architecture type (per official docs)
        if "canary" in self.model_name.lower():
            print("  Using EncDecMultiTaskModel class for Canary model...")
            # Canary is a MultiTask model
            self.model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name=self.model_name)
        elif "ctc" in self.model_name.lower():
            print("  Using EncDecCTCModelBPE class for CTC model...")
            self.model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=self.model_name)
        else:
            print("  Using generic ASRModel class...")
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)

        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded on {self.device} ({self.variant_key})")

    # -------------------------
    # Discovery / selection (same as your validator)
    # -------------------------
    def discover_transcription_models(self, transcripts_dir: str = "processed/transcripts") -> List[str]:
        transcripts_dir = Path(transcripts_dir)
        if not transcripts_dir.exists():
            print(f"Warning: Transcripts directory not found: {transcripts_dir}")
            return []

        model_dirs = [d for d in transcripts_dir.iterdir() if d.is_dir()]

        model_names = []
        for d in model_dirs:
            has_transcripts = any(
                (subdir / "transcript.json").exists()
                for subdir in d.iterdir()
                if subdir.is_dir()
            )
            if has_transcripts:
                model_names.append(d.name)

        return sorted(model_names)

    def select_model_to_validate(self, transcripts_dir: str = "processed/transcripts") -> Optional[str]:
        models = self.discover_transcription_models(transcripts_dir)
        if not models:
            print("No transcription models found!")
            print(f"Expected structure: {transcripts_dir}/{{ModelName}}/{{VideoName}}/transcript.json")
            return None

        print(f"\n{'=' * 60}")
        print("AVAILABLE TRANSCRIPTION MODELS")
        print(f"{'=' * 60}")

        for i, model in enumerate(models, 1):
            model_dir = Path(transcripts_dir) / model
            video_count = len([d for d in model_dir.iterdir() if d.is_dir()])
            print(f"{i}. {model} ({video_count} videos)")

        print(f"\n{'=' * 60}")

        while True:
            try:
                choice = input(f"\nSelect model to validate (1-{len(models)}, or 'q' to quit): ").strip()
                if choice.lower() == "q":
                    return None
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    selected = models[idx]
                    print(f"Selected: {selected}")
                    return selected
                print(f"Invalid choice. Please enter 1-{len(models)}")
            except (ValueError, KeyboardInterrupt):
                print("\nValidation cancelled")
                return None

    # -------------------------
    # Transcription + caching
    # -------------------------
    def transcribe_audio(self, audio_path: str, video_name: str = None, use_cache: bool = True) -> str:
        audio_path = Path(audio_path)

        if video_name is None:
            video_name = audio_path.stem.replace(" ", "_")

        # Cache:
        # processed/transcripts/Nvidia-<variant_key>/<video_name>/transcript.json
        cache_dir = Path("processed/transcripts") / f"Nvidia-{self.variant_key}" / video_name
        cache_file = cache_dir / "transcript.json"

        if use_cache and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                print("  ✓ Using cached transcription")
                return cached_data.get("text", "")
            except Exception as e:
                print(f"  Warning: Failed to load cache, will re-transcribe: {e}")

        is_temp = False
        temp_dir = Path("processed/temp_audio")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # If it's a video, extract audio first
        if audio_path.suffix.lower() in [".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm"]:
            try:
                try:
                    from moviepy import VideoFileClip
                except ImportError:
                    from moviepy.editor import VideoFileClip

                extracted = temp_dir / f"temp_val_{audio_path.stem}.wav"
                print(f"  Extracting audio -> {extracted.name} ...")
                video = VideoFileClip(str(audio_path))
                video.audio.write_audiofile(str(extracted), logger=None)
                video.close()
                audio_path = extracted
                is_temp = True
            except Exception as e:
                print(f"  Warning: Audio extraction failed, trying direct load: {e}")

        # Ensure 16k mono wav (NeMo likes this)
        normalized_wav = temp_dir / f"temp_norm_{video_name}.wav"
        try:
            normalize_audio_to_16k_mono_wav(audio_path, normalized_wav)
        except Exception as e:
            print(f"  Error normalizing audio: {e}")
            # fall back to raw path
            normalized_wav = audio_path

        # Transcribe with parameters to bypass lhotse sampler
        # num_workers=0 forces simple sequential processing, avoiding dynamic sampler
        with torch.no_grad():
            out = self.model.transcribe(
                [str(normalized_wav)],
                batch_size=1,
                num_workers=0
            )

        # NeMo returns list[str] for most ASR models
        if isinstance(out, (list, tuple)) and len(out) > 0:
            transcription = out[0]
        else:
            transcription = str(out)

        # Cleanup temp files
        for p in [normalized_wav]:
            try:
                if p.exists() and p.parent == temp_dir:
                    p.unlink()
            except Exception:
                pass
        if is_temp:
            try:
                if audio_path.exists() and audio_path.parent == temp_dir:
                    audio_path.unlink()
            except Exception:
                pass

        # Cache transcript
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_data = {
                "text": transcription,
                "model": self.model_name,
                "variant_key": self.variant_key,
                "audio_file": str(Path(audio_path).name),
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Cached transcription to {cache_file}")
        except Exception as e:
            print(f"  Warning: Failed to cache transcription: {e}")

        return transcription

    # -------------------------
    # Validation logic
    # -------------------------
    def validate_single_video(
        self,
        whisper_transcript: str,
        audio_path: str,
        video_name: str,
        asr_model: str,
        output_dir: Path,
    ) -> Dict:
        print("\n  Transcribing with Nvidia Model...")

        nvidia_transcript = self.transcribe_audio(
            audio_path=str(audio_path),
            video_name=video_name,
            use_cache=True,
        )

        whisper_words = whisper_transcript.lower().split()
        nvidia_words = nvidia_transcript.lower().split()

        matcher = difflib.SequenceMatcher(None, whisper_words, nvidia_words)
        matching_blocks = matcher.get_matching_blocks()
        total_matches = sum(block.size for block in matching_blocks)
        max_words = max(len(whisper_words), len(nvidia_words))
        agreement = (total_matches / max_words * 100) if max_words > 0 else 0

        metrics = {
            "video_name": video_name,
            "asr_model": asr_model,
            "audio_file": Path(audio_path).name,
            "agreement": round(agreement, 2),
            "whisper_words": len(whisper_words),
            "nvidia_words": len(nvidia_words),
            "matching_words": total_matches,
            "whisper_transcript": whisper_transcript,
            "nvidia_transcript": nvidia_transcript,
            "validator_model": self.model_name,
            "validator_variant": self.variant_key,
            "device": self.device,
        }

        result_file = output_dir / f"{video_name}_validation.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"  Agreement: {agreement:.1f}%")
        return metrics

    def batch_validate(
        self,
        asr_model: Optional[str] = None,
        transcripts_dir: str = "processed/transcripts",
        videos_dir: str = "videos",
        output_base: str = "processed/validation",
    ) -> List[Dict]:
        transcripts_dir = Path(transcripts_dir)
        videos_dir = Path(videos_dir)
        output_base = Path(output_base)

        if asr_model is None:
            asr_model = self.select_model_to_validate(transcripts_dir)
            if asr_model is None:
                print("Validation cancelled")
                return []

        validation_name = f"{asr_model}_vs_{self.variant_key}"
        output_dir = output_base / validation_name
        output_dir.mkdir(parents=True, exist_ok=True)

        model_dir = transcripts_dir / asr_model
        if not model_dir.exists():
            print(f"Error: Model directory not found: {model_dir}")
            return []

        video_dirs = [
            d for d in model_dir.iterdir()
            if d.is_dir() and (d / "transcript.json").exists()
        ]

        print(f"\n{'=' * 60}")
        print(f"BATCH VALIDATION: {asr_model} vs Nvidia ({self.variant_key})")
        print(f"{'=' * 60}")
        print(f"ASR Model: {asr_model}")
        print(f"Validator: {self.model_name}")
        print(f"Found {len(video_dirs)} transcripts to validate")
        print(f"Output: {output_dir}")
        print(f"{'=' * 60}\n")

        all_metrics = []

        for i, video_dir in enumerate(video_dirs, 1):
            video_name = video_dir.name
            transcript_file = video_dir / "transcript.json"

            print(f"[{i}/{len(video_dirs)}] Validating: {video_name}")

            try:
                with open(transcript_file, "r", encoding="utf-8") as f:
                    transcript_data = json.load(f)

                whisper_text = transcript_data.get("text", "")
                if not whisper_text:
                    print("  Skipping: No transcript text found")
                    continue

                # locate video
                video_path = videos_dir / f"{video_name}.mp4"
                video_name_spaces = video_name.replace("_", " ")

                if not video_path.exists():
                    video_path = videos_dir / f"{video_name_spaces}.mp4"

                if not video_path.exists():
                    for ext in [".mkv", ".avi", ".mov", ".webm"]:
                        test_path = videos_dir / f"{video_name}{ext}"
                        if test_path.exists():
                            video_path = test_path
                            break
                        test_path = videos_dir / f"{video_name_spaces}{ext}"
                        if test_path.exists():
                            video_path = test_path
                            break

                if not video_path.exists():
                    print(f"  Skipping: Video file not found for {video_name}")
                    continue

                metrics = self.validate_single_video(
                    whisper_transcript=whisper_text,
                    audio_path=str(video_path),
                    video_name=video_name,
                    asr_model=asr_model,
                    output_dir=output_dir,
                )
                all_metrics.append(metrics)

            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

        if all_metrics:
            avg_agreement = float(np.mean([m["agreement"] for m in all_metrics]))

            summary = {
                "asr_model": asr_model,
                "validator_model": self.model_name,
                "validator_variant": self.variant_key,
                "total_videos": len(all_metrics),
                "average_agreement": round(avg_agreement, 2),
                "individual_results": all_metrics,
            }

            summary_file = output_dir / "validation_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print(f"\n{'=' * 60}")
            print("VALIDATION SUMMARY")
            print(f"{'=' * 60}")
            print(f"ASR Model: {asr_model}")
            print(f"Validator: {self.model_name} ({self.variant_key})")
            print(f"Total Validated: {len(all_metrics)}")
            print(f"Average Agreement: {avg_agreement:.1f}%")
            print(f"\nResults saved to: {output_dir}")
            print(f"Summary: {summary_file}")
            print(f"{'=' * 60}\n")

        return all_metrics


# -------------------------
# Simple CLI selection
# -------------------------
def get_user_device_selection():
    print("\nWhere do you want to load the model?")
    print("1- CUDA")
    print("2- CPU")

    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            if torch.cuda.is_available():
                return "cuda"
            print("Warning: CUDA not available, falling back to CPU.")
            return "cpu"
        if choice == "2":
            return "cpu"
        print("Invalid choice. Please enter 1 or 2.")


def select_nvidia_variant() -> str:
    keys = list(NvidiaValidator.AVAILABLE_VARIANTS.keys())
    print("\nSelect Nvidia model variant:")
    for i, k in enumerate(keys, 1):
        print(f"{i}. {k}  ->  {NvidiaValidator.AVAILABLE_VARIANTS[k]}")
    while True:
        choice = input(f"Enter choice (1-{len(keys)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
        except ValueError:
            pass
        print("Invalid choice.")


if __name__ == "__main__":
    # Dependencies check
    try:
        import nemo.collections.asr  # noqa
    except ImportError:
        print("Error: Missing NeMo ASR dependencies!")
        print("Install with: pip install nemo_toolkit[asr]")
        raise

    device_choice = get_user_device_selection()
    variant_key = select_nvidia_variant()

    validator = NvidiaValidator(
        variant_key=variant_key,
        device=device_choice,
    )

    # Run batch validation with interactive ASR-model selection
    validator.batch_validate(
        transcripts_dir="processed/transcripts",
        videos_dir="videos",
        output_base="processed/validation",
    )

    # Run single video validation
    validator.validate_single_video(
        whisper_transcript="processed/transcripts/00000000.json",
        audio_path="videos/00000000.mp4",
        video_name="00000000",
        asr_model="whisper",
        output_dir="processed/validation",
    )
    
    print("\nValidation Complete!")