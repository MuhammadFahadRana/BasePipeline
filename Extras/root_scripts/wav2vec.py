import os
import json
import time
import torch
import torchaudio
import warnings
import argparse
import traceback
import soundfile as sf
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Optional: Load environment variables if needed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings("ignore")

class Wav2VecTranscriber:
    """
    Standalone Wav2Vec2 transcriber based on facebook/wav2vec2-large-960h.
    Saves results to project-specific paths:
    processed/transcripts/Wav2Vec/<video_name>/transcripts/
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-960h",
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Select precision: float16 for GPU, float32 for CPU
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"Loading Wav2Vec2 model: {model_name} on {self.device} ({self.dtype})")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()
        self.model_name = model_name
        print(f"✓ Wav2Vec2 model loaded")

    def extract_audio(self, video_path: Path) -> Path:
        """Extract audio to a temporary WAV file using moviepy."""
        from moviepy import VideoFileClip
        
        temp_wav = video_path.parent / f"temp_wv_{video_path.stem}_{int(time.time())}.wav"
        print(f"  Extracting audio...")
        video = VideoFileClip(str(video_path))
        if video.audio is None:
            video.close()
            raise ValueError(f"Video file has no audio track: {video_path.name}")
        video.audio.write_audiofile(str(temp_wav), logger=None, fps=16000, nbytes=2, codec='pcm_s16le')
        video.close()
        return temp_wav

    def transcribe(self, audio_path: Path, chunk_size_s: int = 30) -> dict:
        """Process audio file using chunking and return text + timestamps."""
        # Load audio
        try:
            data, sample_rate = sf.read(str(audio_path), dtype='float32')
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            waveform = torch.from_numpy(data)
        except Exception as e:
            print(f"  Warning: soundfile load failed, trying torchaudio: {e}")
            waveform, sample_rate = torchaudio.load(str(audio_path))
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0)
            waveform = waveform.squeeze()

        # Resample to 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Chunked processing
        total_samples = len(waveform)
        chunk_len = chunk_size_s * 16000
        overlap = 2 * 16000
        
        full_text = []
        all_segments = []
        
        print(f"  Inference ({len(waveform)/16000:.1f}s audio with timestamps)...")
        
        start = 0
        while start < total_samples:
            end = min(start + chunk_len + overlap, total_samples)
            chunk = waveform[start:end]
            
            # Run inference for this chunk
            chunk_result = self._run_inference(chunk, offset_s=start/16000)
            
            if chunk_result["text"]:
                full_text.append(chunk_result["text"])
                all_segments.extend(chunk_result["segments"])
            
            start += chunk_len
            if start >= total_samples:
                break
                
        return {
            "text": " ".join(full_text).replace("  ", " ").strip(),
            "segments": all_segments
        }

    def _run_inference(self, waveform_chunk: torch.Tensor, offset_s: float = 0.0) -> dict:
        """Helper to run model inference and extract timestamps."""
        input_values = self.processor(
            waveform_chunk.numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_values.to(self.device, dtype=self.dtype)

        with torch.no_grad():
            logits = self.model(input_values).logits

        # Get predicted IDs and word offsets
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # CTC models have a frame rate of 50 per second (20ms per frame)
        # 16000 Hz / 320 (downsampling factor of wav2vec2) = 50 Hz
        time_per_frame = self.model.config.inputs_to_logits_ratio / 16000
        
        # Decode with offsets (pass sequences/ids, not logits)
        outputs = self.processor.batch_decode(predicted_ids, output_word_offsets=True)
        text = outputs.text[0]
        word_offsets = outputs.word_offsets[0]
        
        segments = []
        for word_info in word_offsets:
            word = word_info["word"]
            # Start and End are indices in the logits
            start_time = round(word_info["start_offset"] * time_per_frame + offset_s, 3)
            end_time = round(word_info["end_offset"] * time_per_frame + offset_s, 3)
            
            segments.append({
                "word": word,
                "start": start_time,
                "end": end_time
            })
            
        return {"text": text, "segments": segments}

    def process_file(self, file_path: str, output_base: str = "processed/transcripts/Wav2Vec"):
        """Process a single video/audio file and save results."""
        file_path = Path(file_path)
        video_name = file_path.stem.replace(" ", "_")
        
        # Target: processed/transcripts/Wav2Vec/<video_name>/transcripts/
        output_dir = Path(output_base) / video_name / "transcripts"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = output_dir / "transcript.json"
        txt_path = output_dir / "transcript.txt"

        print(f"\nProcessing: {file_path.name}")
        
        is_audio = file_path.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a"]
        audio_path = None
        
        try:
            audio_path = file_path if is_audio else self.extract_audio(file_path)
            
            start_time = time.time()
            result = self.transcribe(audio_path)
            elapsed = time.time() - start_time
            
            # Save JSON (Include segments for Atlas search)
            result_data = {
                "text": result["text"],
                "segments": result["segments"],
                "model": self.model_name,
                "video_file": file_path.name,
                "processing_time": round(elapsed, 2),
                "device": self.device,
                "precision": str(self.dtype)
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
                
            # Save TXT (Clean transcription only)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
            
            # Force OS to flush or ensure it exists
            if not txt_path.exists():
                print(f"  Warning: TXT file not detected immediately after write.")
                
            print(f"✓ Transcribed in {elapsed:.2f}s")
            print(f"✓ Saved JSON: {json_path.name}")
            print(f"✓ Saved TXT:  {txt_path.name}")
            print(f"✓ Output Dir: {output_dir}")
            
        except Exception as e:
            print(f"  ✗ Error processing {file_path.name}: {e}")
            traceback.print_exc()
        finally:
            # Cleanup temp
            if audio_path and audio_path != file_path and audio_path.exists():
                try:
                    audio_path.unlink()
                except:
                    pass
            # Clear GPU cache
            if self.device == "cuda":
                torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Dedicated Wav2Vec2 Transcriber")
    parser.add_argument("--videos-dir", type=str, default="videos", help="Directory with video/audio files")
    parser.add_argument("--output-dir", type=str, default="processed/transcripts/Wav2Vec", help="Base output directory")
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu, or auto")
    
    args = parser.parse_args()
    
    transcriber = Wav2VecTranscriber(device=args.device)
    
    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists():
        print(f"Error: Directory {videos_dir} not found")
        return
        
    supported_exts = [".mp4", ".mkv", ".avi", ".mov", ".ts", ".wav", ".mp3", ".flac"]
    files = [f for f in videos_dir.iterdir() if f.suffix.lower() in supported_exts]
    
    if not files:
        print(f"No supported media files found in {videos_dir}")
        return
        
    print(f"Found {len(files)} files to process.")
    
    for f in files:
        try:
            transcriber.process_file(str(f), output_base=args.output_dir)
        except Exception as e:
            print(f"Error processing {f.name}: {e}")

if __name__ == "__main__":
    main()
