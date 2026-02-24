import os
import time
import torch
import warnings
from pathlib import Path

# Fix for broken torchcodec on Windows
# We try to prevent it from being imported or used by transformers
try:
    import sys
    sys.modules['torchcodec'] = None
except Exception:
    pass

# Suppress "You seem to be using the pipelines sequentially on GPU" warning
import logging
logging.getLogger("transformers.pipelines.base").setLevel(logging.ERROR)

from transcriber_utils import (
    extract_audio_to_wav, load_audio_array, save_results, get_device, ALL_MEDIA
)

class SpeechT5Transcriber:
    def __init__(self, model_size="large", device="auto"):
        if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        from transformers import pipeline
        print(f"Loading microsoft/speecht5_asr on {device}...")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="microsoft/speecht5_asr",
            device=0 if device == "cuda" else -1,
        )
        self.model_name = "SpeechT5-ASR"

    def transcribe_video(self, file_path, output_dir="processed"):
        file_path = Path(file_path)
        video_name = file_path.stem.replace(" ", "_")
        model_output_dir = Path(output_dir) / "transcripts" / self.model_name / video_name
        
        print(f"\nTranscribing (SpeechT5): {file_path.name}")
        wav_path = extract_audio_to_wav(file_path)
        start_time = time.time()
        
        try:
            audio, sr = load_audio_array(wav_path, target_sr=16000)
            
            # Manual chunking to bypass broken pipeline chunking in transformers dev
            chunk_size = 15 * 16000  # 15 seconds
            stride = 2 * 16000     # 2 seconds overlap
            
            full_text = []
            segments = []
            
            curr = 0
            while curr < len(audio):
                end = min(curr + chunk_size, len(audio))
                chunk = audio[curr:end]
                
                # Skip very short tail
                if len(chunk) < 1600: # 0.1s
                    break
                    
                res = self.pipe(chunk)
                chunk_text = res.get("text", "").strip()
                
                if chunk_text:
                    full_text.append(chunk_text)
                    segments.append({
                        "start": curr / 16000,
                        "end": end / 16000,
                        "text": chunk_text
                    })
                
                curr += (chunk_size - stride)
                if end == len(audio):
                    break
                
            result = {"text": " ".join(full_text), "segments": segments}
            elapsed = time.time() - start_time
            save_results(result, model_output_dir, video_name, self.model_name, elapsed, file_path.name)
            return result
        finally:
            if wav_path.exists(): wav_path.unlink()

    def batch_transcribe(self, folder_path="videos", output_dir="processed"):
        folder_path = Path(folder_path)
        files = [f for f in folder_path.glob("*.*") if f.suffix.lower() in ALL_MEDIA]
        for f in files:
            try: self.transcribe_video(f, output_dir)
            except Exception as e: print(f"Failed {f.name}: {e}")

if __name__ == "__main__":
    device = get_device()
    transcriber = SpeechT5Transcriber(model_size="large", device=device)
    
    # Batch process
    transcriber.batch_transcribe(folder_path="videos", output_dir="processed")

    # Single video
    # transcriber.transcribe_video(r"videos_test\AkerBP 1.mp4", output_dir="processed")
