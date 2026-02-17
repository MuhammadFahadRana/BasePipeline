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


# Model configuration database
MODEL_CONFIGS = {
    "whisper": {
        "name": "OpenAI Whisper",
        "variants": {
            "1": {"name": "tiny", "params": "39M", "description": "Fastest, least accurate"},
            "2": {"name": "base", "params": "74M", "description": "Fast, good for simple audio"},
            "3": {"name": "small", "params": "244M", "description": "Balanced speed/accuracy"},
            "4": {"name": "medium", "params": "769M", "description": "High accuracy, slower"},
            "5": {"name": "large", "params": "1550M", "description": "Best accuracy, slowest"},
            "6": {"name": "large-v2", "params": "1550M", "description": "Improved large model"},
            "7": {"name": "large-v3", "params": "1550M", "description": "Latest version (Nov 2023)"},
        }
    },
    "whisperx": {
        "name": "WhisperX (Enhanced Whisper)",
        "variants": {
            "1": {"name": "tiny", "params": "39M", "description": "Fastest + better timestamps"},
            "2": {"name": "base", "params": "74M", "description": "Fast + word alignment"},
            "3": {"name": "small", "params": "244M", "description": "Balanced + speaker diarization"},
            "4": {"name": "medium", "params": "769M", "description": "High accuracy + alignment"},
            "5": {"name": "large-v2", "params": "1550M", "description": "Best accuracy + features"},
        }
    },
    "distil-whisper": {
        "name": "Distil-Whisper (6x Faster)",
        "variants": {
            "1": {"name": "distil-medium.en", "model_id": "distil-whisper/distil-medium.en", 
                  "description": "English-only, medium size"},
            "2": {"name": "distil-large-v2", "model_id": "distil-whisper/distil-large-v2", 
                  "description": "English-only, large model"},
            "3": {"name": "distil-large-v3", "model_id": "distil-whisper/distil-large-v3", 
                  "description": "Latest version (best)"},
        }
    },
    "wav2vec": {
        "name": "Meta Wav2Vec2",
        "variants": {
            "1": {"name": "wav2vec2-base-960h", "model_id": "facebook/wav2vec2-base-960h", 
                  "params": "95M", "description": "Base model, English"},
            "2": {"name": "wav2vec2-large-960h", "model_id": "facebook/wav2vec2-large-960h", 
                  "params": "315M", "description": "Large model, better accuracy"},
            "3": {"name": "wav2vec2-large-960h-lv60-self", "model_id": "facebook/wav2vec2-large-960h-lv60-self", 
                  "params": "315M", "description": "State-of-the-art English"},
            "4": {"name": "wav2vec2-large-xlsr-53", "model_id": "facebook/wav2vec2-large-xlsr-53", 
                  "params": "315M", "description": "53 languages support"},
        }
    },
    "nemo": {
        "name": "NVIDIA NeMo Canary",
        "variants": {
            "1": {"name": "canary-1b", "model_id": "nvidia/canary-1b", 
                  "params": "1B", "description": "Multilingual, supports 4 languages"},
            "2": {"name": "stt_en_fastconformer_hybrid_large_streaming_80ms", 
                  "model_id": "nvidia/stt_en_fastconformer_hybrid_large_streaming_80ms",
                  "params": "115M", "description": "English streaming ASR"},
            "3": {"name": "parakeet-tdt-1.1b", "model_id": "nvidia/parakeet-tdt-1.1b",
                  "params": "1.1B", "description": "English telephony ASR"},
        }
    }
}


class SimpleTranscriber:
    """
    Multi-backend ASR transcriber supporting Whisper, WhisperX, Distil-Whisper, Wav2Vec2, and NeMo.
    """

    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".ts"}
    AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}

    def __init__(self, backend: str, model_variant: dict, device: str = "auto"):
        """
        Initialize transcriber with selected backend and model variant.

        Args:
            backend: Backend type (whisper/whisperx/distil-whisper/wav2vec/nemo)
            model_variant: Dictionary containing variant details
            device: "auto", "cpu", or "cuda"
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"\n{'='*60}")
        print(f"Initializing ASR Model")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Backend: {backend}")
        print(f"Model: {model_variant['name']}")
        if 'params' in model_variant:
            print(f"Parameters: {model_variant['params']}")
        print(f"{'='*60}\n")

        self.device = device
        self.backend = backend
        self.model_variant = model_variant

        # Load the appropriate model
        if backend == "whisper":
            self._load_whisper(model_variant['name'])
        elif backend == "whisperx":
            self._load_whisperx(model_variant['name'])
        elif backend == "distil-whisper":
            self._load_distil_whisper(model_variant['model_id'])
        elif backend == "wav2vec":
            self._load_wav2vec(model_variant['model_id'])
        elif backend == "nemo":
            self._load_nemo(model_variant['model_id'])
        else:
            raise ValueError(f"Unknown backend: {backend}")

        print(f"✓ {self.model_name} loaded successfully\n")

    def _load_whisper(self, model_size):
        """Load OpenAI Whisper model"""
        self.model_name = f"Whisper-{model_size.capitalize()}"
        print(f"Loading {self.model_name}...")
        self.model = whisper.load_model(model_size, device=self.device)

    def _load_whisperx(self, model_size):
        """Load WhisperX model"""
        self.model_name = f"WhisperX-{model_size.capitalize()}"
        print(f"Loading {self.model_name}...")
        try:
            import whisperx
            compute_type = "float16" if self.device == "cuda" else "int8"
            self.model = whisperx.load_model(model_size, device=self.device, compute_type=compute_type)
            self.whisperx = whisperx  # Store module reference
        except ImportError:
            raise ImportError("WhisperX not installed. Run: pip install whisperx")

    def _load_distil_whisper(self, model_id):
        """Load Distil-Whisper model"""
        self.model_name = model_id.split("/")[-1]
        print(f"Loading {self.model_name} from HuggingFace...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True,
        ).to(self.device)

    def _load_wav2vec(self, model_id):
        """Load Wav2Vec2 model"""
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        
        self.model_name = model_id.split("/")[-1]
        print(f"Loading {self.model_name} from HuggingFace...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)

    def _load_nemo(self, model_id):
        """Load NVIDIA NeMo model"""
        try:
            import nemo.collections.asr as nemo_asr
            self.model_name = model_id.split("/")[-1]
            print(f"Loading {self.model_name} from NVIDIA NeMo...")
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_id)
            if self.device == "cuda":
                self.model = self.model.cuda()
            self.nemo = nemo_asr
        except ImportError:
            raise ImportError("NeMo not installed. Run: pip install nemo_toolkit[asr]")

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
            elif self.backend == "wav2vec":
                result = self._transcribe_wav2vec(audio_path)
            elif self.backend == "nemo":
                result = self._transcribe_nemo(audio_path)
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
            # Ensure input_features matches model's dtype (float16 on CUDA)
            input_features = inputs.input_features.to(self.device, dtype=self.model.dtype)
            attention_mask = (
                torch.ones_like(input_features[:, 0, :]).long().to(self.device)
            )

            # Generate with proper parameters for Distil-Whisper
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    max_new_tokens=256,  # Reduced to stay within combined limit
                    num_beams=1,  # Faster greedy search
                )

            chunk_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            if chunk_text.strip():  # Only add non-empty chunks
                segments.append(
                    {"start": start_time, "end": end_time, "text": chunk_text.strip()}
                )

        # Combine all chunk texts for full transcript
        text = " ".join(seg["text"] for seg in segments)

        # Keep output compatible with pipeline
        return {
            "text": text,
            "segments": segments,
            "language": "en",  # Distil-Whisper is English-only
        }

    def _transcribe_wav2vec(self, audio_path: Path):
        """Wav2Vec2 transcription"""
        import soundfile as sf
        
        # Load audio
        speech, sr = sf.read(str(audio_path))
        
        # Resample to 16kHz if needed
        if sr != 16000:
            import torchaudio.transforms as T
            speech = torch.from_numpy(speech).float().unsqueeze(0)
            resampler = T.Resample(orig_freq=sr, new_freq=16000)
            speech = resampler(speech).squeeze().numpy()
            sr = 16000
        
        # Tokenize
        inputs = self.processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return {
            "text": transcription,
            "segments": [{"start": 0, "end": len(speech) / sr, "text": transcription}],
            "language": "en"
        }

    def _transcribe_nemo(self, audio_path: Path):
        """
        NeMo transcription with workaround for lhotse compatibility issues.
        
        Note: NeMo has known compatibility issues with certain versions of lhotse library.
        This implementation uses a simplified config to avoid DynamicCutSampler errors.
        """
        try:
            # Create a custom transcribe config to avoid lhotse DynamicCutSampler issues
            # This bypasses the problematic dataloader initialization
            print("  Transcribing with NeMo Canary (this may take a moment)...")
            
            # NeMo Canary (multitask model) - simple transcription without override_config
            # The model will use its default configuration
            transcriptions = self.model.transcribe(audio=[str(audio_path)])
            
            # NeMo returns a list of transcriptions
            text = transcriptions[0] if transcriptions else ""
            
            return {
                "text": text,
                "segments": [{"start": 0, "end": 0, "text": text}],
                "language": "en"
            }
            
        except Exception as e:
            error_msg = str(e)
            
            # Check for specific lhotse/NeMo compatibility error
            if "object.__init__()" in error_msg or "DynamicCutSampler" in error_msg:
                print("\n" + "=" * 60)
                print("NEMO COMPATIBILITY ERROR")
                print("=" * 60)
                print("\nNeMo has a known compatibility issue with the lhotse library.")
                print("This is an internal dependency conflict in NeMo toolkit.")
                print("\nPossible solutions:")
                print("  1. Use a different model (Whisper, WhisperX, Distil-Whisper, Wav2Vec2)")
                print("  2. Try downgrading lhotse: pip install lhotse==1.16.0")
                print("  3. Update NeMo: pip install --upgrade nemo_toolkit[asr]")
                print("\nFor now, please select a different model.")
                print("=" * 60)
                raise RuntimeError(
                    "NeMo transcription failed due to lhotse compatibility issue. "
                    "Please use a different model or fix the library versions."
                ) from e
            else:
                # Re-raise other errors
                print(f"\nNeMo transcription error: {error_msg}")
                raise


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
        top_db = 30  # Threshold in dB below reference to consider silence
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


def display_model_menu():
    """Display main model selection menu"""
    print("\n" + "=" * 60)
    print("ASR MODEL SELECTION")
    print("=" * 60)
    print("\nChoose the model for transcription:\n")
    print("1. OpenAI Whisper          - Most popular, great accuracy")
    print("2. WhisperX                - Enhanced Whisper with better timestamps")
    print("3. Distil-Whisper          - 6x faster than Whisper")
    print("4. Wav2Vec2                - Meta's open-source model")
    print("5. NVIDIA NeMo Canary      - Enterprise-grade ASR")
    print("\n" + "=" * 60)
    
    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        if choice in ["1", "2", "3", "4", "5"]:
            return choice
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")


def display_variant_menu(backend_name):
    """Display variant selection menu for chosen model"""
    backend_map = {
        "1": "whisper",
        "2": "whisperx",
        "3": "distil-whisper",
        "4": "wav2vec",
        "5": "nemo"
    }
    
    backend = backend_map[backend_name]
    config = MODEL_CONFIGS[backend]
    
    print("\n" + "=" * 60)
    print(f"SELECT {config['name'].upper()} VARIANT")
    print("=" * 60)
    
    for key, variant in config['variants'].items():
        desc = f"{key}. {variant['name']}"
        if 'params' in variant:
            desc += f" ({variant['params']})"
        desc += f" - {variant['description']}"
        print(desc)
    
    print("=" * 60)
    
    max_choice = str(len(config['variants']))
    while True:
        choice = input(f"\nEnter your choice (1-{max_choice}): ").strip()
        if choice in config['variants']:
            return backend, config['variants'][choice]
        else:
            print(f"Invalid choice. Please enter a number between 1 and {max_choice}.")


def get_user_device_selection():
    """Get user's device preference"""
    print("\n" + "=" * 60)
    print("DEVICE SELECTION")
    print("=" * 60)
    print("\nWhere do you want to load the model?\n")
    print("1. CUDA (GPU)")
    print("2. CPU")
    print("=" * 60)

    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice == "1":
            if torch.cuda.is_available():
                print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
                return "cuda"
            else:
                print("⚠ Warning: CUDA not available, falling back to CPU.")
                return "cpu"
        elif choice == "2":
            return "cpu"
        else:
            print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    # Step 1: Select main model
    model_choice = display_model_menu()
    
    # Step 2: Select variant
    backend, variant = display_variant_menu(model_choice)
    
    # Step 3: Select device
    device = get_user_device_selection()
    
    # Step 4: Initialize transcriber
    transcriber = SimpleTranscriber(
        backend=backend,
        model_variant=variant,
        device=device
    )
    
    # Step 5: Transcribe
    # Single file transcription
    transcriber.transcribe_video("videos\\17Min.mp4", output_dir="processed")
    
    # Batch transcription (uncomment to use)
    # transcriber.batch_transcribe(
    #     folder_path="videos",
    #     output_dir="processed"
    # )
