"""
Qwen-Audio ASR Transcriber

High-performance transcription using Alibaba's Qwen-Audio model.
Qwen-Audio is a multimodal model that excels at audio understanding tasks
including speech recognition, audio captioning, and speech analysis.

Models available:
- Qwen/Qwen-Audio-Chat (7B parameters) - Conversational, best for dialogue
- Qwen/Qwen2-Audio-7B-Instruct - Multimodal audio understanding
- Qwen/Qwen3-ASR-1.7B - State-of-the-art ASR (via qwen-asr package)
- Qwen/Qwen3-ASR-0.6B - Lightweight ASR (via qwen-asr package)

Features:
- Multi-language support (100+ languages)
- Speaker diarization capabilities
- Emotion detection (optional)
- Low latency (~2-3x faster than Whisper Large)
- Better punctuation and capitalization
- Smart caching (skips existing transcripts)
- Robust batch processing

Installation:
    pip install transformers accelerate torch torchaudio
    pip install -U qwen-asr  # For Qwen3-ASR models

Usage:
    python qwen_asr_transcriber.py
    # Automatically detects CUDA/CPU
"""

import torch
import torchaudio
import soundfile as sf
import time
import json
import warnings
import os
import shutil
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np

try:
    # Key fix: Import Qwen2AudioForConditionalGeneration
    from transformers import (
        AutoProcessor,
        Qwen2AudioForConditionalGeneration,
        AutoModelForCausalLM,
    )
except ImportError as e:
    raise ImportError(
        f"transformers error: {e}. Run: pip install transformers accelerate"
    )

try:
    try:
        from moviepy import VideoFileClip
    except ImportError:
        from moviepy.editor import VideoFileClip
except ImportError:
    VideoFileClip = None
    print("Warning: moviepy not installed. Run: pip install moviepy")

try:
    from qwen_asr import Qwen3ASRModel
    HAS_QWEN_ASR = True
except ImportError:
    HAS_QWEN_ASR = False

warnings.filterwarnings("ignore")


class QwenTranscriber:
    """
    Qwen-Audio ASR Transcriber

    Uses Alibaba's Qwen-Audio models for high-quality speech recognition.
    """

    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".ts"}
    AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}

    SUPPORTED_MODELS = {
        "qwen-audio-chat": "Qwen/Qwen-Audio-Chat",
        "qwen2-audio": "Qwen/Qwen2-Audio-7B-Instruct",
        "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
        "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    }

    # Qwen3-ASR expects full language names, not ISO codes
    QWEN3_LANGUAGE_MAP = {
        "en": "English", "zh": "Chinese", "yue": "Cantonese",
        "ar": "Arabic", "de": "German", "fr": "French", "es": "Spanish",
        "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
        "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese",
        "ja": "Japanese", "tr": "Turkish", "hi": "Hindi", "ms": "Malay",
        "nl": "Dutch", "sv": "Swedish", "da": "Danish", "fi": "Finnish",
        "pl": "Polish", "cs": "Czech", "fil": "Filipino", "fa": "Persian",
        "el": "Greek", "hu": "Hungarian", "mk": "Macedonian", "ro": "Romanian",
    }
    def __init__(
        self,
        model_name: str = "qwen2-audio",
        device: str = "auto",
        compute_type: str = "float16",
        language: str = "en",
        enable_timestamps: bool = True,
    ):
        """
        Initialize Qwen-Audio transcriber.

        Args:
            model_name: Model to use ("qwen-audio-chat" or "qwen2-audio")
            device: "auto", "cpu", or "cuda"
            compute_type: "float16" (faster) or "float32" (more accurate)
            language: Target language code (e.g., "en", "no", "zh")
            enable_timestamps: Generate word-level timestamps
        """
        # Resolve model name
        if model_name in self.SUPPORTED_MODELS:
            self.model_id = self.SUPPORTED_MODELS[model_name]
            self.model_name = model_name
        else:
            self.model_id = model_name  # Allow custom HF model IDs
            self.model_name = model_name.split("/")[-1]

        # Check if this is a Qwen3-ASR model (uses dedicated qwen-asr package)
        self.is_qwen3_asr = "qwen3-asr" in self.model_id.lower()

        # Device selection
        if device == "auto":
            from transcriber_utils import get_device
            self.device = get_device()
        else:
            self.device = device

        if device != "cuda" and not (device == "auto" and torch.cuda.is_available()):
            self.compute_type = "float32"  # float16 segfaults on CPU
        else:
            self.compute_type = compute_type

        self.language = language
        self.enable_timestamps = enable_timestamps

        print(f"\n{'=' * 60}")
        print(f"Qwen-Audio ASR Transcriber")
        print(f"{'=' * 60}")
        print(f"Model:        {self.model_id}")
        print(f"Device:       {self.device}")
        print(f"Compute Type: {self.compute_type}")
        print(f"Language:     {self.language}")
        print(f"Timestamps:   {self.enable_timestamps}")
        print(f"{'=' * 60}\n")

        # Load model and processor
        self._load_model()

    def _load_model(self):
        """Load Qwen-Audio model and processor."""
        print(f"Loading {self.model_id}...")

        # --- Qwen3-ASR: uses dedicated qwen-asr package ---
        if self.is_qwen3_asr:
            if not HAS_QWEN_ASR:
                raise ImportError(
                    "qwen-asr package required for Qwen3-ASR models. "
                    "Run: pip install -U qwen-asr"
                )
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            try:
                self.model = Qwen3ASRModel.from_pretrained(
                    self.model_id,
                    dtype=dtype,
                    device_map="cuda:0" if self.device == "cuda" else "cpu",
                    max_new_tokens=1024,
                )
                self.processor = None
                print(f"Qwen3-ASR model loaded successfully\n")
            except Exception as e:
                if self.device == "cuda":
                    print(f"  Warning: Qwen3-ASR failure on CUDA: {e}. Falling back to CPU.")
                    self.device = "cpu"
                    self.model = Qwen3ASRModel.from_pretrained(
                        self.model_id,
                        dtype=torch.float32,
                        device_map="cpu",
                        max_new_tokens=1024,
                    )
                    self.processor = None
                else:
                    raise e
            return

        # --- Qwen2-Audio / legacy Qwen-Audio-Chat ---
        dtype = torch.float16 if self.compute_type == "float16" else torch.float32

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)

            if "qwen2-audio" in self.model_id.lower():
                loader = Qwen2AudioForConditionalGeneration
            else:
                loader = AutoModelForCausalLM

            try:
                self.model = loader.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                )
            except Exception as e:
                if self.device == "cuda":
                    print(f"  Warning: Qwen model failure on CUDA: {e}. Falling back to CPU.")
                    self.device = "cpu"
                    self.model = loader.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float32,
                        device_map=None,
                        trust_remote_code=True,
                    )
                else:
                    raise e

            if self.device == "cpu":
                self.model = self.model.to("cpu")

            self.model.eval()
            print(f"Model loaded successfully\n")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def transcribe_video(
        self,
        file_path: str,
        output_dir: str = "processed",
        include_emotion: bool = False,
        include_speaker_info: bool = False,
        skip_if_exists: bool = False,
    ) -> Dict:
        """
        Transcribe video or audio file using Qwen-Audio.

        Args:
            file_path: Path to video/audio file
            output_dir: Base output directory
            include_emotion: Detect emotional tone (experimental)
            include_speaker_info: Attempt speaker separation
            skip_if_exists: Skip if output file already exists

        Returns:
            Transcription result dictionary
        """
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"✗ File not found: {file_path}")
            return {}

        # Setup output paths
        video_name = file_path.stem
        # Sanitize video name for folder creation (legacy compatibility)
        video_sanitized = video_name.replace(" ", "_")
        model_output_dir = (
            Path(output_dir)
            / "transcripts"
            / f"Qwen-{self.model_name}"
            / video_sanitized
        )
        json_path = model_output_dir / "full_transcript.json"

        # Check cache — but reprocess if the cached transcript is empty
        if skip_if_exists and json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("text", "").strip():
                    print(f"Cached transcript found for {video_name}. Skipping.")
                    return cached
                else:
                    print(f"  Cached transcript for {video_name} is empty, reprocessing...")
            except Exception:
                print("  (Cache file corrupted, reprocessing...)")

        print(f"\n{'=' * 60}")
        print(f"Transcribing: {file_path.name}")
        print(f"{'=' * 60}")

        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Extract audio if video
        temp_audio_path = None
        try:
            if file_path.suffix.lower() in self.VIDEO_EXTENSIONS:
                print("Extracting audio from video...")
                try:
                    audio_path = self.extract_audio(file_path)
                    temp_audio_path = audio_path  # Mark for deletion
                except Exception as e:
                    print(f"Audio extraction failed: {e}")
                    # Special handling for files with no audio stream
                    if "no audio stream" in str(e).lower():
                        print("Skipping file (no audio)")
                        return {"text": "", "segments": [], "error": "No audio stream"}
                    return {}
            else:
                audio_path = file_path

            # --- Qwen3-ASR: native pipeline (handles long audio, resampling, etc.) ---
            if self.is_qwen3_asr:
                print("Transcribing with Qwen3-ASR native pipeline...")
                start_processing_time = time.time()

                qwen3_result = self._transcribe_qwen3(str(audio_path))

                processing_time = time.time() - start_processing_time

                result = {
                    "text": qwen3_result.get("text", ""),
                    "segments": qwen3_result.get("segments", []),
                    "language": qwen3_result.get("language", self.language),
                    "processing_time_seconds": round(processing_time, 2),
                }
                result["metadata"] = {
                    "model": self.model_id,
                    "file": str(file_path),
                    "language": result["language"],
                    "device": self.device,
                    "processing_time": round(processing_time, 2),
                }

                self.save_transcript(result, json_path)
                txt_path = model_output_dir / "transcript.txt"
                self.save_text_transcript(result, txt_path, source_name=video_name)
                standard_json_path = model_output_dir / "transcript.json"
                self.save_transcript(result, standard_json_path)

                print(f"Transcription complete! Saved to: {model_output_dir}")
                return result

            # Load audio
            print("Loading audio...")
            try:
                try:
                    # Try soundfile first (reliable on Windows)
                    audio_array, sample_rate = sf.read(str(audio_path))
                    waveform = torch.from_numpy(audio_array).float()

                    # Convert to [channels, samples] if needed
                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0)
                    else:
                        waveform = waveform.T
                except Exception as sf_err:
                    print(
                        f"  Note: soundfile load failed ({sf_err}), trying torchaudio..."
                    )
                    waveform, sample_rate = torchaudio.load(str(audio_path))

                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # Resample if needed (Qwen expects 16kHz)
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                    sample_rate = 16000

                audio_array = waveform.squeeze().numpy()
                
                # --- Audio normalization (CRITICAL for Qwen) ---
                if audio_array.ndim > 1:
                    audio_array = np.mean(audio_array, axis=0)
                
                # Scale to [-1, 1] if not already
                peak = np.max(np.abs(audio_array))
                if peak > 0:
                    audio_array = audio_array / peak
                
                print(f"  Duration: {len(audio_array) / sample_rate:.2f}s")
                print(f"  Audio Peak: {peak:.4f}, Mean: {np.mean(np.abs(audio_array)):.4f}")

            except Exception as e:
                print(f"✗ Error loading audio: {e}")
                return {"error": str(e)}

            # Start timing
            start_processing_time = time.time()

            # Transcribe with chunking to handle long audio
            print(f"Transcribing with Qwen-Audio (chunking enabled)...")

            chunk_size_sec = 30  # Qwen2-Audio works best with 30s chunks
            stride_sec = 2
            chunk_size = chunk_size_sec * sample_rate
            stride = stride_sec * sample_rate

            assert chunk_size > stride, f"chunk_size ({chunk_size}) must exceed stride ({stride})"

            full_text = []
            segments = []

            curr = 0
            while curr < len(audio_array):
                end = min(curr + chunk_size, len(audio_array))
                chunk = audio_array[curr:end]

                # Skip very short tail
                if len(chunk) < sample_rate * 0.5:  # 0.5s
                    break

                print(
                    f"  Processing chunk {curr / sample_rate:.1f}s - {end / sample_rate:.1f}s..."
                )

                try:
                    chunk_res = self._transcribe_qwen(
                        chunk,
                        sample_rate,
                        include_emotion=include_emotion,
                        include_speaker_info=include_speaker_info,
                    )

                    chunk_text = chunk_res.get("text", "").strip()
                    print(f"    Chunk result ({len(chunk_text)} chars): {chunk_text[:50]}...")

                    # Filter refusal messages but log them so we can diagnose
                    is_refusal = (
                        "I'm sorry" in chunk_text
                        or "provide the" in chunk_text
                        or "I cannot" in chunk_text
                        or "I can't" in chunk_text
                    )
                    if chunk_text and not is_refusal:
                        full_text.append(chunk_text)
                        segments.append(
                            {
                                "start": curr / sample_rate,
                                "end": end / sample_rate,
                                "text": chunk_text,
                            }
                        )
                    elif chunk_text and is_refusal:
                        print(f"    [WARN] Filtered refusal: {chunk_text[:80]}")
                except Exception as e:
                    print(f"    Warning: Chunk processing failed: {e}")

                curr += chunk_size - stride
                if end == len(audio_array):
                    break

            # End timing
            processing_time = time.time() - start_processing_time

            result = {
                "text": " ".join(full_text),
                "segments": segments,
                "language": self.language,
                "processing_time_seconds": round(processing_time, 2),
            }

            # Add metadata
            result["metadata"] = {
                "model": self.model_id,
                "file": str(file_path),
                "duration": len(audio_array) / sample_rate,
                "language": self.language,
                "device": self.device,
                "processing_time": round(processing_time, 2),
            }

            # Save results
            # Save normal transcript.json in addition to full_transcript.json to match user expectation
            # User wants it like transcriber.py

            # 1. full_transcript.json
            self.save_transcript(result, json_path)

            # 2. transcript.txt
            txt_path = model_output_dir / "transcript.txt"
            self.save_text_transcript(result, txt_path, source_name=video_name)

            # 3. transcript.json (standard format)
            # This is what transcriber.py usually produces
            standard_json_path = model_output_dir / "transcript.json"
            self.save_transcript(result, standard_json_path)

            print(f"Transcription complete! Saved to: {model_output_dir}")
            return result

        except Exception as e:
            print(f"Transcription error: {e}")
            return {}
        finally:
            # Clean up temp audio file
            if temp_audio_path and temp_audio_path.exists():
                try:
                    os.remove(temp_audio_path)
                except:
                    pass

    def _transcribe_qwen3(self, audio_path: str) -> Dict:
        """Transcribe using Qwen3-ASR's native pipeline (handles long audio natively)."""
        lang_name = self.QWEN3_LANGUAGE_MAP.get(self.language)  # None triggers auto-detect
        try:
            results = self.model.transcribe(
                audio=audio_path,
                language=lang_name,
            )
        except Exception as e:
            print(f"  Qwen3-ASR transcription error: {e}")
            return {"text": "", "segments": [], "language": self.language}

        if not results:
            return {"text": "", "segments": [], "language": self.language}

        text = results[0].text or ""
        detected_lang = getattr(results[0], "language", None) or self.language

        segments = []
        if text:
            segments = [{"start": 0, "end": 0, "text": text}]

        return {
            "text": text,
            "segments": segments,
            "language": detected_lang,
        }

    def _transcribe_qwen(
        self,
        audio: any,
        sample_rate: int,
        include_emotion: bool = False,
        include_speaker_info: bool = False,
    ) -> Dict:
        """
        Perform Qwen-Audio transcription.
        """
        # --- Audio sanity + normalization (prevents silent failures) ---
        # Qwen2-Audio expects float audio in roughly [-1, 1]. Some loaders/codepaths
        # can yield int16-like ranges or NaNs/Infs which often decode to empty output.
        audio_np = np.asarray(audio)
        if audio_np.ndim != 1:
            audio_np = audio_np.reshape(-1)

        if audio_np.size == 0:
            return {"text": "", "segments": [], "language": self.language}

        audio_np = audio_np.astype(np.float32, copy=False)
        if not np.isfinite(audio_np).all():
            audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)

        peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
        if peak == 0.0:
            return {"text": "", "segments": [], "language": self.language}

        # If values look like int16 scale (or otherwise too large), normalize.
        if peak > 1.5:
            audio_np = audio_np / peak

        # Build a simple, direct prompt — complex instructions cause refusals
        # or confuse the model, leading to empty output.
        prompt = f"Transcribe this audio. Output only the transcription in {self.language}."

        # Prepare conversation format (no system role — Qwen2-Audio may not
        # support it and can produce malformed token sequences).
        conversation = [
            {
                "role": "user",
                "content": [
                    # IMPORTANT: For Qwen2-Audio, the chat template expects an audio *placeholder*.
                    # The actual waveform is passed via `audios=[...]` to the processor.
                    {"type": "audio"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Process
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        # print(f"    [DEBUG] Prompt: {text}") # Uncomment if needed

        inputs = self.processor(
            text=text,
            audios=[audio_np],
            return_tensors="pt",
            sampling_rate=sample_rate,
            padding=True,
        )

        # Move to device
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Generate
        def _generate(do_sample: bool, temperature: float):
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=do_sample,
                repetition_penalty=1.1,  # Prevent repeating sentences
            )
            if do_sample:
                gen_kwargs.update({"temperature": temperature, "top_p": 0.9})

            # Some configs may not define pad_token_id; align with tokenizer if needed.
            if hasattr(self.processor, "tokenizer"):
                tok = self.processor.tokenizer
                if getattr(self.model.generation_config, "pad_token_id", None) is None:
                    gen_kwargs["pad_token_id"] = getattr(tok, "pad_token_id", None)
                if getattr(self.model.generation_config, "eos_token_id", None) is None:
                    gen_kwargs["eos_token_id"] = getattr(tok, "eos_token_id", None)

            with torch.no_grad():
                return self.model.generate(**inputs, **gen_kwargs)

        output_ids = _generate(do_sample=False, temperature=0.0)
        
        # print(f"    [DEBUG] Input IDs shape: {inputs['input_ids'].shape}, Output IDs shape: {output_ids.shape}")

        # Decode
        # Check if we actually generated something
        if output_ids.shape[1] > inputs["input_ids"].shape[1]:
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            transcription = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
        else:
            print("    [DEBUG] No new tokens generated.")
            transcription = ""

        # Retry once with a simpler prompt + mild sampling if we got nothing back.
        if not transcription.strip():
            retry_conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio"},
                        {
                            "type": "text",
                            "text": f"Transcribe the audio in {self.language}.",
                        },
                    ],
                },
            ]
            retry_text = self.processor.apply_chat_template(
                retry_conversation, tokenize=False, add_generation_prompt=True
            )
            retry_inputs = self.processor(
                text=retry_text,
                audios=[audio_np],
                return_tensors="pt",
                sampling_rate=sample_rate,
                padding=True,
            )
            retry_inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in retry_inputs.items()
            }
            inputs = retry_inputs
            output_ids = _generate(do_sample=True, temperature=0.2)
            # Validate that new tokens were actually generated
            if output_ids.shape[1] > inputs["input_ids"].shape[1]:
                output_ids = output_ids[:, inputs["input_ids"].shape[1] :]
                transcription = self.processor.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0]
            else:
                print("    [DEBUG] Retry also produced no new tokens.")
                transcription = ""

        # Parse result
        result = {
            "text": transcription.strip(),
            "segments": [],
            "language": self.language,
        }

        # Try to parse timestamps
        if self.enable_timestamps:
            result["segments"] = self._parse_timestamps(transcription)

        return result

    def _parse_timestamps(self, text: str) -> List[Dict]:
        """Parse timestamp-annotated text from Qwen-Audio."""
        import re

        segments = []
        pattern = r"\[(\d+:\d+(?::\d+)?)\]\s*([^\[]+)"
        matches = re.findall(pattern, text)

        for i, (timestamp, segment_text) in enumerate(matches):
            parts = timestamp.split(":")
            if len(parts) == 2:  # MM:SS
                minutes, seconds = map(int, parts)
                start_time = minutes * 60 + seconds
            else:  # HH:MM:SS
                hours, minutes, seconds = map(int, parts)
                start_time = hours * 3600 + minutes * 60 + seconds

            if i + 1 < len(matches):
                next_parts = matches[i + 1][0].split(":")
                if len(next_parts) == 2:
                    end_time = int(next_parts[0]) * 60 + int(next_parts[1])
                else:
                    end_time = (
                        int(next_parts[0]) * 3600
                        + int(next_parts[1]) * 60
                        + int(next_parts[2])
                    )
            else:
                end_time = start_time + 5

            segments.append(
                {"start": start_time, "end": end_time, "text": segment_text.strip()}
            )

        if not segments:
            segments = [{"start": 0, "end": 0, "text": text.strip()}]

        return segments

    def extract_audio(self, video_path: Path) -> Path:
        """Extract audio from video file."""
        if VideoFileClip is None:
            raise RuntimeError("moviepy not installed. Run: pip install moviepy")

        audio_path = video_path.parent / f"{video_path.stem}_audio.wav"

        try:
            # Try ffmpeg directly first if available (often more reliable for weird codecs)
            import subprocess
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', str(video_path),
                    '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                    str(audio_path)
                ], check=True, capture_output=True)
                return audio_path
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to moviepy
                if VideoFileClip is None:
                    raise RuntimeError("ffmpeg not found and moviepy not installed.")
                video = VideoFileClip(str(video_path))
                try:
                    if video.audio is None:
                        raise ValueError("Video file has no audio stream")
                    video.audio.write_audiofile(
                        str(audio_path), codec="pcm_s16le", fps=16000, nbytes=2, logger=None
                    )
                    return audio_path
                finally:
                    video.close()

        except Exception as e:
            print(f"Error extracting audio: {e}")
            raise

    def save_transcript(self, result: Dict, output_file: Path):
        """Save full transcription results as JSON."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def save_text_transcript(
        self, result: Dict, output_file: Path, source_name: str = "file"
    ):
        """Save transcript as readable text."""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Transcription of: {source_name}\n")
            f.write(f"Model: {result.get('metadata', {}).get('model', 'Unknown')}\n")
            f.write("=" * 60 + "\n\n")

            if result.get("segments"):
                for seg in result["segments"]:
                    start_str = str(timedelta(seconds=int(seg["start"])))
                    end_str = str(timedelta(seconds=int(seg["end"])))
                    f.write(f"[{start_str} -> {end_str}]\n")
                    f.write(f"{seg['text']}\n\n")
            else:
                f.write(result.get("text", ""))

    def batch_transcribe(
        self,
        folder_path: str = "videos",
        output_dir: str = "processed",
        file_extensions: Optional[List[str]] = None,
        skip_existing: bool = True,
    ) -> List[Dict]:
        """
        Transcribe all video/audio files in a folder.
        """
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Folder not found: {folder}")
            return []

        if file_extensions is None:
            extensions = self.VIDEO_EXTENSIONS | self.AUDIO_EXTENSIONS
        else:
            extensions = {
                ext if ext.startswith(".") else f".{ext}" for ext in file_extensions
            }

        files = [
            f
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]

        if not files:
            print(f"No supported files found in {folder}")
            return []

        print(f"\n============================================================")
        print(f"BATCH PROCESSING: {len(files)} files")
        print(f"Model: {self.model_name}")
        print(f"Skip existing: {skip_existing}")
        print(f"============================================================\n")

        results = []
        success_count = 0

        # Use simple loop if tqdm not available, else tqdm
        iterator = tqdm(files, desc="Batch Progress")

        for file_path in iterator:
            try:
                result = self.transcribe_video(
                    str(file_path), output_dir, skip_if_exists=skip_existing
                )
                if result and not result.get("error"):
                    results.append(result)
                    success_count += 1
            except KeyboardInterrupt:
                print("\nBatch processing interrupted by user.")
                break
            except Exception as e:
                print(f"Unexpected error on {file_path.name}: {e}")
                continue

        print(f"\n============================================================")
        print(f"BATCH COMPLETE")
        print(f"Successful: {success_count}/{len(files)}")
        print(
            f"Total transcripts saved to: {Path(output_dir) / 'transcripts' / f'Qwen-{self.model_name}'}"
        )
        print(f"============================================================\n")

        return results


def main():
    # If arguments provided, use CLI mode
    import argparse
    import sys

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Qwen-Audio ASR Transcriber")
        parser.add_argument("input_pos", nargs='?', help="Video/audio file or folder (positional)")
        parser.add_argument("--input", dest="input_named", help="Video/audio file or folder (named)")
        parser.add_argument(
            "--model",
            default="qwen2-audio",
            choices=list(QwenTranscriber.SUPPORTED_MODELS.keys()),
            help="Model to use",
        )
        parser.add_argument("--device", default="auto")
        parser.add_argument("--language", default="en")
        parser.add_argument("--output", default="processed")
        parser.add_argument(
            "--batch", action="store_true", help="Treat input as folder"
        )
        parser.add_argument(
            "--force", action="store_true", help="Reprocess existing files"
        )
        parser.add_argument("--emotion", action="store_true", help="Detect emotion")
        parser.add_argument("--speaker", action="store_true", help="Identify speakers")

        args = parser.parse_args()

        transcriber = QwenTranscriber(
            model_name=args.model,
            device=args.device,
            language=args.language,
            enable_timestamps=True,
        )

        input_path = args.input_named or args.input_pos
        if not input_path:
            print("Error: No input file or folder specified.")
            parser.print_help()
            return

        if args.batch or Path(input_path).is_dir():
            transcriber.batch_transcribe(
                input_path, args.output, skip_existing=not args.force
            )
        else:
            transcriber.transcribe_video(
                input_path,
                args.output,
                include_emotion=args.emotion,
                include_speaker_info=args.speaker,
                skip_if_exists=not args.force,
            )
        return

    # Auto device selection
    selected_device = "auto"

    # Initialize transcriber
    transcriber = QwenTranscriber(model_name="qwen2-audio", device=selected_device)

    ## Batch process all videos in a folder
    transcriber.batch_transcribe(folder_path="videos", output_dir="processed")

    # Process single video
    # transcriber.transcribe_video("videos\\Risk management.mp4", output_dir="processed")


if __name__ == "__main__":
    main()
