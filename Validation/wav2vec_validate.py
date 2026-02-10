"""
Cross-validation using Wav2Vec2 (Facebook's ASR model)
Validates Whisper/WhisperX/Distil-Whisper transcriptions using a different architecture.
"""

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
import json
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional
import difflib
import numpy as np
import os
import shutil
import warnings


# Ensure ffmpeg is in PATH and named correctly
def prepare_ffmpeg():
    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)

        # On Windows, some tools expect 'ffmpeg.exe'
        if os.name == "nt":
            ffmpeg_name = os.path.basename(ffmpeg_exe)
            if ffmpeg_name != "ffmpeg.exe":
                dest = os.path.join(ffmpeg_dir, "ffmpeg.exe")
                if not os.path.exists(dest):
                    try:
                        shutil.copy(ffmpeg_exe, dest)
                    except Exception:
                        pass

        if ffmpeg_dir not in os.environ["PATH"]:
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
        return True
    except ImportError:
        return False


prepare_ffmpeg()
warnings.filterwarnings("ignore")


class Wav2Vec2Validator:
    """
    Validate ASR transcriptions using Wav2Vec2 as an independent check.

    Supports validation of:
    - Whisper (various sizes)
    - WhisperX
    - Distil-Whisper
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-960h",
        device: str = "auto",
    ):
        """
        Initialize Wav2Vec2 model.

        Args:
            model_name: Hugging Face model identifier
                       "facebook/wav2vec2-base-960h" - English ASR (faster)
                       "facebook/wav2vec2-large-960h" - Larger English model (more accurate)
                       "facebook/mms-1b-all" - Multilingual model (1100+ languages)
            device: Device to load model on ("cuda", "cpu", or "auto")
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"Loading Wav2Vec2 model: {model_name}")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        self.model.to(self.device)
        self.model_name = model_name
        print(f"✓ Wav2Vec2 model loaded on {self.device}")

    def discover_transcription_models(
        self, transcripts_dir: str = "processed/transcripts"
    ) -> List[str]:
        """
        Discover which ASR models have transcripts available.

        Returns:
            List of model names (e.g., ['Whisper-Large', 'WhisperX-Base'])
        """
        transcripts_dir = Path(transcripts_dir)

        if not transcripts_dir.exists():
            print(f"Warning: Transcripts directory not found: {transcripts_dir}")
            return []

        # Find all model directories
        model_dirs = [d for d in transcripts_dir.iterdir() if d.is_dir()]

        # Filter to only model-specific directories (skip old-style video dirs)
        model_names = []
        for d in model_dirs:
            # Model dirs should contain video subdirectories with transcript.json
            has_transcripts = any(
                (subdir / "transcript.json").exists()
                for subdir in d.iterdir()
                if subdir.is_dir()
            )
            if has_transcripts:
                model_names.append(d.name)

        return sorted(model_names)

    def select_model_to_validate(
        self, transcripts_dir: str = "processed/transcripts"
    ) -> Optional[str]:
        """
        Interactively select which model's transcripts to validate.

        Returns:
            Selected model name or None if cancelled
        """
        models = self.discover_transcription_models(transcripts_dir)

        if not models:
            print(" No transcription models found!")
            print(
                f" Expected structure: {transcripts_dir}/{{ModelName}}/{{VideoName}}/transcript.json"
            )
            return None

        print(f"\n{'=' * 60}")
        print("AVAILABLE TRANSCRIPTION MODELS")
        print(f"{'=' * 60}")

        for i, model in enumerate(models, 1):
            # Count videos for this model
            model_dir = Path(transcripts_dir) / model
            video_count = len([d for d in model_dir.iterdir() if d.is_dir()])
            print(f"{i}. {model} ({video_count} videos)")

        print(f"\n{'=' * 60}")

        while True:
            try:
                choice = input(
                    f"\nSelect model to validate (1-{len(models)}, or 'q' to quit): "
                ).strip()

                if choice.lower() == "q":
                    return None

                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    selected = models[idx]
                    print(f" Selected: {selected}")
                    return selected
                else:
                    print(f"Invalid choice. Please enter 1-{len(models)}")
            except (ValueError, KeyboardInterrupt):
                print("\nValidation cancelled")
                return None

    def transcribe_audio(
        self, audio_path: str, video_name: str = None, use_cache: bool = True
    ) -> str:
        """
        Transcribe audio file using Wav2Vec2.

        Args:
            audio_path: Path to audio/video file
            video_name: Name of video (for caching). If None, extracted from path.
            use_cache: If True, load cached transcription if available

        Returns:
            Transcription text
        """
        audio_path = Path(audio_path)

        # Determine video name for caching
        if video_name is None:
            video_name = audio_path.stem.replace(" ", "_")

        # Define cache location: processed/transcripts/Wav2Vec2-{model}/videoname/transcript.json
        model_short_name = self.model_name.split("/")[-1]
        cache_dir = (
            Path("processed/transcripts") / f"Wav2Vec2-{model_short_name}" / video_name
        )
        cache_file = cache_dir / "transcript.json"

        # Check for cached transcription
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                print(f"  ✓ Using cached Wav2Vec2 transcription")
                return cached_data.get("text", "")
            except Exception as e:
                print(f"  Warning: Failed to load cache, will re-transcribe: {e}")

        is_temp = False

        # If it's a video, extract audio first
        if audio_path.suffix.lower() in [
            ".mp4",
            ".mkv",
            ".avi",
            ".mov",
            ".flv",
            ".wmv",
            ".webm",
        ]:
            try:
                try:
                    from moviepy import VideoFileClip
                except ImportError:
                    from moviepy.editor import VideoFileClip

                temp_wav = audio_path.parent / f"temp_val_{audio_path.stem}.wav"
                print(f"  Extracting audio to {temp_wav.name}...")
                video = VideoFileClip(str(audio_path))
                video.audio.write_audiofile(str(temp_wav), logger=None)
                video.close()
                audio_path = temp_wav
                is_temp = True
            except Exception as e:
                print(f"  Warning: Audio extraction failed, trying direct load: {e}")

        # Load audio using soundfile for better reliability on Windows
        try:
            data, sample_rate = sf.read(str(audio_path))
            # Convert to [channels, time] format that torchaudio/torch expect
            if len(data.shape) == 1:
                waveform = torch.from_numpy(data).float().unsqueeze(0)
            else:
                waveform = torch.from_numpy(data.T).float()
        except Exception as e:
            print(f"  Error loading audio with soundfile: {e}")
            # Fallback to torchaudio
            waveform, sample_rate = torchaudio.load(str(audio_path))

        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Prepare input
        input_values = self.processor(
            waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_values.to(self.device)

        # Perform inference
        with torch.no_grad():
            logits = self.model(input_values).logits

        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        # Clean up temp file
        if is_temp and audio_path.exists():
            try:
                audio_path.unlink()
            except:
                pass

        # Cache the transcription
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_data = {
                "text": transcription,
                "model": self.model_name,
                "video_file": str(Path(audio_path).name),
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Cached transcription to {cache_file}")
        except Exception as e:
            print(f"  Warning: Failed to cache transcription: {e}")

        return transcription

    def validate_single_video(
        self,
        whisper_transcript: str,
        audio_path: str,
        video_name: str,
        asr_model: str,
        output_dir: Path,
    ) -> Dict:
        """
        Validate a single video's transcription.

        Args:
            whisper_transcript: Transcript from Whisper/WhisperX/Distil-Whisper
            audio_path: Path to source video/audio file
            video_name: Name of the video
            asr_model: Name of the ASR model being validated (e.g., "Whisper-Large")
            output_dir: Where to save validation results

        Returns:
            Validation metrics dictionary
        """
        print("\n  Transcribing with Wav2Vec2...")

        # Get Wav2Vec2 transcript (with caching)
        wav2vec_transcript = self.transcribe_audio(
            audio_path=str(audio_path), video_name=video_name, use_cache=True
        )

        # Compare transcripts
        whisper_words = whisper_transcript.lower().split()
        wav2vec_words = wav2vec_transcript.lower().split()

        # Calculate agreement
        matcher = difflib.SequenceMatcher(None, whisper_words, wav2vec_words)
        matching_blocks = matcher.get_matching_blocks()
        total_matches = sum(block.size for block in matching_blocks)
        max_words = max(len(whisper_words), len(wav2vec_words))
        agreement = (total_matches / max_words * 100) if max_words > 0 else 0

        metrics = {
            "video_name": video_name,
            "asr_model": asr_model,
            "audio_file": Path(audio_path).name,
            "agreement": round(agreement, 2),
            "whisper_words": len(whisper_words),
            "wav2vec2_words": len(wav2vec_words),
            "matching_words": total_matches,
            "whisper_transcript": whisper_transcript,
            "wav2vec2_transcript": wav2vec_transcript,
            "wav2vec_model": self.model_name,
        }

        # Save individual validation result
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
        """
        Validate all transcripts from a selected ASR model.

        Args:
            asr_model: Model name to validate (e.g., "Whisper-Large"). If None, prompts user.
            transcripts_dir: Base directory containing transcripts
            videos_dir: Directory containing source video files
            output_base: Base directory for validation results

        Returns:
            List of validation metrics for all videos
        """
        transcripts_dir = Path(transcripts_dir)
        videos_dir = Path(videos_dir)
        output_base = Path(output_base)

        # Select model if not specified
        if asr_model is None:
            asr_model = self.select_model_to_validate(transcripts_dir)
            if asr_model is None:
                print("Validation cancelled")
                return []

        # Create model-specific output directory
        # Structure: processed/validation/{ASRModel}_vs_Wav2Vec2/
        validation_name = f"{asr_model}_vs_{self.model_name.split('/')[-1]}"
        output_dir = output_base / validation_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all transcripts for this model
        model_dir = transcripts_dir / asr_model
        if not model_dir.exists():
            print(f"Error: Model directory not found: {model_dir}")
            return []

        # Get all video directories with transcripts
        video_dirs = [
            d
            for d in model_dir.iterdir()
            if d.is_dir() and (d / "transcript.json").exists()
        ]

        print(f"\n{'=' * 60}")
        print(f"BATCH VALIDATION: {asr_model} vs Wav2Vec2")
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
                # Load Whisper transcript
                with open(transcript_file, "r", encoding="utf-8") as f:
                    transcript_data = json.load(f)

                whisper_text = transcript_data.get("text", "")
                if not whisper_text:
                    print("  Skipping: No transcript text found")
                    continue

                # Find corresponding video file
                # Try exact match first (with underscores)
                video_path = videos_dir / f"{video_name}.mp4"

                # Try with spaces instead of underscores
                if not video_path.exists():
                    video_name_spaces = video_name.replace("_", " ")
                    video_path = videos_dir / f"{video_name_spaces}.mp4"

                # Try other common extensions
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

                # Validate
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

        # Save summary report
        if all_metrics:
            avg_agreement = np.mean([m["agreement"] for m in all_metrics])

            summary = {
                "asr_model": asr_model,
                "validator_model": self.model_name,
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
            print(f"Validator: {self.model_name}")
            print(f"Total Validated: {len(all_metrics)}")
            print(f"Average Agreement: {avg_agreement:.1f}%")
            print(f"\nResults saved to: {output_dir}")
            print(f"Summary: {summary_file}")
            print(f"{'=' * 60}\n")

        return all_metrics

def get_user_device_selection():
    print("\nWhere you want to load the model?")
    print("1- CUDA")
    print("2- CPU")

    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            if torch.cuda.is_available():
                return "cuda"
            else:
                print("Warning: CUDA not available, falling back to CPU.")
                return "cpu"
        elif choice == "2":
            return "cpu"
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    # Check dependencies
    try:
        import transformers
        import torchaudio
    except ImportError:
        print("Error: Missing dependencies!")
        print("Please install with: pip install transformers torchaudio")
        exit(1)

    # Initialize validator
    # Available models:
    # - "facebook/wav2vec2-base-960h" (faster, less accurate)
    # - "facebook/wav2vec2-large-960h" (slower, more accurate)
    # - "facebook/mms-1b-all" (multilingual, 1100+ languages)

    # Get device selection from user
    device_choice = get_user_device_selection()

    validator = Wav2Vec2Validator(
        model_name="facebook/wav2vec2-large-960h",
        device=device_choice
    )

    # =====================================================================

    # Directly specify model
    # validator.batch_validate(asr_model="Whisper-Large")
    # validator.batch_validate(asr_model="WhisperX-Large")
    # validator.batch_validate(asr_model="Distil-Whisper-Large-v3")

    # =====================================================================

    # Run batch validation with interactive model selection
    validator.batch_validate(
        transcripts_dir="processed/transcripts",
        videos_dir="videos",
        output_base="processed/validation",
    )

    print("\nValidation Complete!")
