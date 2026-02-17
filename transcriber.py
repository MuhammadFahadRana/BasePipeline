import whisper
import torch
import json
import warnings
import os
import imageio_ffmpeg
import shutil
import numpy as np

from pathlib import Path
from datetime import timedelta

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


# Ensure ffmpeg is in PATH and named correctly for Whisper
def prepare_ffmpeg():
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)

    # On Windows, Whisper expects 'ffmpeg.exe'
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


prepare_ffmpeg()
warnings.filterwarnings("ignore")


class SimpleTranscriber:
    """
    Multi-backend ASR transcriber supporting Whisper, WhisperX, and Distil-Whisper.
    """

    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".ts"}
    AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}

    def __init__(self, backend: str = "whisper", model_variant: dict = None, model_size: str = "large", device: str = "auto"):
        """
        Initialize transcriber with selected backend.

        Args:
            backend: "whisper", "whisperx", or "distil-whisper"
            model_variant: Dict with model details (e.g. {'name': 'base'})
            model_size: fallback if model_variant is None
            device: "auto", "cpu", or "cuda"
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {device}")

        self.device = device
        
        # Handle arguments
        self.backend = backend
        if model_variant and "name" in model_variant:
            self.model_size = model_variant["name"]
        else:
            self.model_size = model_size

        # MODEL SELECTION
        print(f"Initializing {self.backend} with model: {self.model_size}")

        if self.backend == "whisper":
            # OpenAI Whisper
            self.model_name = f"Whisper-{self.model_size.capitalize()}"
            print(f"Loading {self.model_name} on {device}")
            self.model = whisper.load_model(self.model_size, device=device)

        elif self.backend == "whisperx":
            # WhisperX
            self.model_name = f"WhisperX-{self.model_size.capitalize()}"
            print(f"Loading {self.model_name} on {device}")
            try:
                import whisperx
                compute_type = "float16" if device == "cuda" else "int8"
                self.model = whisperx.load_model(self.model_size, device=device, compute_type=compute_type)
                self.whisperx = whisperx
            except ImportError:
                raise ImportError("WhisperX not installed. Run: pip install whisperx")

        elif self.backend == "distil-whisper":
            # Distil-Whisper
            self.model_name = "Distil-Whisper-Large-v3"
            model_id = "distil-whisper/distil-large-v3"
            if model_variant and "model_id" in model_variant:
                model_id = model_variant["model_id"]
                self.model_name = model_id.split("/")[-1]

            print(f"Loading {self.model_name} on {device}")
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True,
            ).to(device)
            print(f"{self.model_name} loaded successfully")
        
        else:
            # Default fallback or error
            print(f"Warning: Unknown backend '{backend}', falling back to Whisper")
            self.backend = "whisper"
            self.model_name = f"Whisper-{self.model_size.capitalize()}"
            self.model = whisper.load_model(self.model_size, device=device)

    def transcribe_video(self, file_path: str, output_dir: str = "processed"):
        """
        Transcribe audio from video/audio file using the selected backend.

        Outputs are saved to: processed/transcripts/{model_name}/{video_name}/
        Example: processed/transcripts/Whisper-Large/AkerBP_1/

        Args:
            file_path: Path to video or audio file
            output_dir: Base output directory (default: "processed")

        Returns:
            Transcription result dictionary
        """
        file_path = Path(file_path)

        # Create model-specific output directory
        # Structure: processed/transcripts/{ModelName}/{VideoName}/
        video_name = file_path.stem.replace(
            " ", "_"
        )  # Replace spaces for cleaner paths
        model_output_dir = (
            Path(output_dir) / "transcripts" / self.model_name / video_name
        )
        model_output_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = model_output_dir / "transcript.json"

        # Check for cached transcript
        if transcript_path.exists():
            # Check modification times
            video_mtime = file_path.stat().st_mtime
            transcript_mtime = transcript_path.stat().st_mtime

            if transcript_mtime > video_mtime:
                print(
                    "Cached transcript found (newer than video). Skipping transcription."
                )
                print(f"  Loaded from: {transcript_path}")
                try:
                    import json

                    with open(transcript_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    print(f"  Warning: Failed to load cache ({e}), re-transcribing...")
            else:
                print("Cached transcript found but video is newer. Re-transcribing...")

        print(f"\nTranscribing: {file_path.name}")
        print(f"Model: {self.model_name}")
        print(f"Output: {model_output_dir}")

        is_audio = file_path.suffix.lower() in self.AUDIO_EXTENSIONS
        audio_path = file_path if is_audio else self.extract_audio(file_path)

        try:
            # Route to appropriate backend
            if self.backend == "whisper":
                result = self._transcribe_whisper(audio_path)
            elif self.backend == "whisperx":
                result = self._transcribe_whisperx(audio_path)
            elif self.backend == "distil-whisper":
                result = self._transcribe_distil_whisper(audio_path)
            else:
                raise ValueError(f"Unknown ASR backend: {self.backend}")

            # Save results with model info
            self.save_transcript(result, transcript_path)

            text_file = model_output_dir / "transcript.txt"
            self.save_text_transcript(result, text_file, file_path.name)

            print(f"✓ Saved to: {model_output_dir}")
            return result

        finally:
            # Clean up temporary audio file
            if not is_audio and audio_path.exists():
                try:
                    audio_path.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete temp audio: {e}")

    # BACKEND IMPLEMENTATIONS

    def _transcribe_whisper(self, audio_path: Path):
        """Standard Whisper transcription with word-level timestamps."""
        return self.model.transcribe(
            str(audio_path),
            word_timestamps=True,
            verbose=False,
            language="en",  # Change to None for auto-detection
        )

    def _transcribe_whisperx(self, audio_path: Path):
        """
        WhisperX transcription with optional word-level alignment.
        Falls back gracefully if alignment models are unavailable.
        """
        result = self.model.transcribe(str(audio_path))

        # Try to perform word-level alignment (WhisperX's best feature)
        try:
            align_model, metadata = self.whisperx.load_align_model(
                language_code=result.get("language", "en"), device=self.device
            )

            aligned = self.whisperx.align(
                result["segments"], align_model, metadata, str(audio_path), self.device
            )

            return {
                "text": result["text"],
                "segments": aligned["segments"],
                "language": result.get("language", "en"),
            }

        except Exception as e:
            print(f"  Warning: Alignment failed ({e}), using unaligned segments")
            return {
                "text": result["text"],
                "segments": result["segments"],
                "language": result.get("language", "en"),
            }

    def _transcribe_distil_whisper(self, audio_path: Path):
        """
        Distil-Whisper transcription (6x faster than standard Whisper).
        Note: Returns only full transcript, no segment-level timestamps.
        """
        import soundfile as sf

        # Load audio using soundfile (avoids torchcodec issues)
        try:
            speech, sr = sf.read(str(audio_path))
            # Convert to torch tensor and ensure shape is [channels, samples]
            speech = torch.from_numpy(speech).float()
            if len(speech.shape) == 1:
                speech = speech.unsqueeze(0)
            else:
                speech = (
                    speech.T
                )  # soundfile gives [samples, channels], we need [channels, samples]
        except Exception as e:
            print(f"  Error loading audio with soundfile: {e}")
            # Fallback to librosa if available
            try:
                import librosa

                speech, sr = librosa.load(str(audio_path), sr=None, mono=False)
                speech = torch.from_numpy(speech).float()
                if len(speech.shape) == 1:
                    speech = speech.unsqueeze(0)
            except ImportError:
                raise RuntimeError(
                    f"Could not load audio file {audio_path}. Please install soundfile or librosa."
                )

        # Convert to mono if stereo
        if speech.shape[0] > 1:
            speech = speech.mean(dim=0, keepdim=True)

        # Resample to 16kHz if necessary (Distil-Whisper expects 16kHz)
        if sr != 16000:
            import torchaudio.transforms as T

            resampler = T.Resample(orig_freq=sr, new_freq=16000)
            speech = resampler(speech)
            sr = 16000

        # Use Voice Activity Detection (VAD) to detect natural speech segments
        # This creates segments at silence boundaries instead of fixed time chunks
        segments_detected = self._detect_speech_segments(speech.squeeze().numpy(), sr)

        print(f"  Detected {len(segments_detected)} speech segments using VAD")

        segments = []

        # Process each detected speech segment
        for i, (start_time, end_time) in enumerate(segments_detected):
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            chunk = speech[:, start_sample:end_sample]

            # Skip very short segments (< 0.3 seconds)
            if (end_sample - start_sample) < 0.3 * sr:
                continue

            inputs = self.processor(
                chunk.squeeze().numpy(),
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )

            # Move inputs to device and create attention mask explicitly
            input_features = inputs.input_features.to(self.device, dtype=self.model.dtype)
            attention_mask = (
                torch.ones_like(input_features[:, 0, :]).long().to(self.device)
            )

            # Generate with higher penalties for better quality in challenging audio
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    num_beams=1,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=4,
                )

            chunk_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            # Hallucination filter: skip segments that are too short or just filler loops
            if not chunk_text or len(chunk_text) < 3:
                continue
            
            # Common hallucinations in music/silence for Whisper-based models
            hallucination_phrases = [
                "I'm going to be", "I'm gonna be", "I'm gonna", "Thank you", 
                "You know", "I think", "I'm", "The", "And"
            ]
            
            # If the segment matches a hallucination phrase exactly (often case for hallucinations)
            is_hallucination = False
            chunk_lower = chunk_text.lower().strip(",. ")
            for phrase in hallucination_phrases:
                if chunk_lower == phrase.lower():
                    # For short segments (< 10s), if it matches perfectly, it's likely a hallucination
                    if (end_time - start_time) < 10.0:
                        is_hallucination = True
                        break

            if is_hallucination:
                continue

            if chunk_text:
                segments.append(
                    {"start": start_time, "end": end_time, "text": chunk_text}
                )

        # Combine all chunk texts for full transcript
        text = " ".join(seg["text"] for seg in segments)

        # Keep output compatible with pipeline
        return {
            "text": text,
            "segments": segments,
            "language": "en",  # Distil-Whisper is English-only
        }

    def _detect_speech_segments(self, audio, sample_rate: int):
        """
        Detect speech segments using silence detection.
        Returns list of (start_time, end_time) tuples for each speech segment.
        """
        try:
            import librosa
            import numpy as np
        except ImportError:
            # Fallback to fixed chunking if librosa not available
            print("  Warning: librosa not found, using fixed chunking")
            duration = len(audio) / sample_rate
            chunk_length = 25
            segments = []
            for start in range(0, int(duration), chunk_length):
                end = min(start + chunk_length, duration)
                segments.append((start, end))
            return segments

        # Parameters for silence detection
        top_db = 15  # Threshold in dB below reference to consider silence (even stricter)
        min_silence_len = 0.3  # Minimum silence duration to split (seconds)
        min_segment_len = 1.0  # Minimum speech segment duration (seconds)
        max_segment_len = 30.0  # Maximum segment length to avoid token limits

        # Detect non-silent intervals (speech segments)
        intervals = librosa.effects.split(
            audio, top_db=top_db, frame_length=2048, hop_length=512
        )

        # Convert sample indices to time
        segments = []
        for start_sample, end_sample in intervals:
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            segment_duration = end_time - start_time

            # Skip very short segments
            if segment_duration < min_segment_len:
                continue

            # Split long segments into smaller chunks
            if segment_duration > max_segment_len:
                num_chunks = int(np.ceil(segment_duration / max_segment_len))
                chunk_duration = segment_duration / num_chunks
                for j in range(num_chunks):
                    chunk_start = start_time + (j * chunk_duration)
                    chunk_end = min(chunk_start + chunk_duration, end_time)
                    segments.append((chunk_start, chunk_end))
            else:
                segments.append((start_time, end_time))

        # If no segments detected, fall back to full audio
        if len(segments) == 0:
            duration = len(audio) / sample_rate
            segments = [(0.0, duration)]

        return segments

    def extract_audio(self, video_path: Path) -> Path:
        """Extract audio from video using moviepy."""
        try:
            try:
                from moviepy import VideoFileClip
            except ImportError:
                from moviepy.editor import VideoFileClip

            # Use a temporary name to avoid overwriting or conflicts
            audio_path = video_path.parent / f"temp_{video_path.stem}.wav"
            video = VideoFileClip(str(video_path))

            # Extract audio
            if video.audio is None:
                video.close()
                raise ValueError(f"Video file {video_path.name} has no audio stream")

            audio = video.audio
            audio.write_audiofile(str(audio_path), logger=None)  # Hide moviepy output

            video.close()
            audio.close()

            return audio_path

        except ImportError:
            # Fallback logic if moviepy is missing (though it should be there)
            print(
                "Warning: moviepy not found, attempting direct transcription (may fail for some video formats)"
            )
            return video_path
        except Exception as e:
            print(f"Error extracting audio: {e}")
            raise

    def save_transcript(self, result: dict, output_file: Path):
        """Save full transcription results as JSON."""
        # Format timestamps for readability
        for segment in result.get("segments", []):
            segment["start_str"] = str(timedelta(seconds=segment["start"]))
            segment["end_str"] = str(timedelta(seconds=segment["end"]))

            if "words" in segment:
                for word in segment["words"]:
                    word["start_str"] = str(timedelta(seconds=word["start"]))
                    word["end_str"] = str(timedelta(seconds=word["end"]))

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def save_text_transcript(
        self, result: dict, output_file: Path, source_name: str = "file"
    ):
        """Save transcript as readable text with timestamps."""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Transcription for {source_name}\n")
            f.write("=" * 50 + "\n\n")

            for segment in result.get("segments", []):
                start_time = str(timedelta(seconds=segment["start"])).split(".")[0]
                text = segment["text"].strip()
                f.write(f"[{start_time}] {text}\n")

    def batch_transcribe(
        self, folder_path: str = "data/videos", output_dir: str = "processed"
    ):
        """
        Transcribe all supported video and audio files in a folder.

        Args:
            folder_path: Directory containing video/audio files
            output_dir: Base output directory

        Returns:
            List of transcription results
        """
        folder_path = Path(folder_path)

        # Find all supported files
        extensions = self.VIDEO_EXTENSIONS | self.AUDIO_EXTENSIONS
        files = [f for f in folder_path.glob("*.*") if f.suffix.lower() in extensions]

        print(f"\n{'=' * 60}")
        print(f"BATCH TRANSCRIPTION - {self.model_name}")
        print(f"{'=' * 60}")
        print(f"Found {len(files)} files to transcribe")
        print(f"Output: {Path(output_dir) / 'transcripts' / self.model_name}")
        print(f"{'=' * 60}\n")

        transcripts = []
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
            try:
                result = self.transcribe_video(file_path, output_dir=output_dir)
                transcripts.append(
                    {
                        "file": file_path.name,
                        "model": self.model_name,
                        "success": True,
                        "transcript_length": len(result.get("text", "")),
                    }
                )
            except Exception as e:
                print(f"✗ Failed: {str(e)}")
                import traceback

                traceback.print_exc()
                transcripts.append(
                    {
                        "file": file_path.name,
                        "model": self.model_name,
                        "error": str(e),
                        "success": False,
                    }
                )

        # Print summary
        successful = sum(1 for t in transcripts if t["success"])
        print(f"\n{'=' * 60}")
        print("BATCH COMPLETE")
        print("=" * 60)
        print(f"Successful: {successful}/{len(files)}")
        print(f"Failed: {len(files) - successful}/{len(files)}")

        return transcripts


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
    # Get user choice for device
    selected_device = get_user_device_selection()

    # Initialize transcriber (comment/uncomment backend in __init__ above)
    transcriber = SimpleTranscriber(
        model_size="large", device=selected_device
    )  # Open AI Whisper model selection tiny, base, small, medium, large

    # transcriber = SimpleTranscriber()

    # Batch process all videos in a folder
    # transcriber.batch_transcribe(
    #     folder_path="videos",
    #     output_dir="processed"
    # )

    # Process single video
    transcriber.transcribe_video("videos\AkerBP 2.mp4", output_dir="processed")
