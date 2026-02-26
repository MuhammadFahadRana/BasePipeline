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
    extract_audio_to_wav, load_audio_array, save_results, get_device, hf_auth, ALL_MEDIA
)

class SpeechT5Transcriber:
    def __init__(self, model_size="large", device="auto"):
        if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Authenticate
        token = hf_auth()
        
        from transformers import pipeline, SpeechT5Processor, SpeechT5ForConditionalGeneration
        print(f"Loading microsoft/speecht5_asr on {device}...")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr", token=token)
        self.model = SpeechT5ForConditionalGeneration.from_pretrained("microsoft/speecht5_asr", token=token).to(device)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=0 if device == "cuda" else -1,
            token=token
        )
        self.model_name = "SpeechT5-ASR"

    def _create_batch_summary(self, results, output_dir, batch_total_time):
        import csv
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        times = [r["time"] for r in successful]
        
        min_time = min(times) if times else 0
        max_time = max(times) if times else 0
        avg_time = sum(times) / len(times) if times else 0
        
        summary_dir = Path(output_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)
        csv_file = summary_dir / f"{self.model_name}_batch_timing.csv"
        
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "success", "time_s", "error"])
            writer.writeheader()
            for r in results:
                writer.writerow({"file": r["file"], "success": "Yes" if r["success"] else "No", "time_s": round(r["time"], 2), "error": r.get("error", "")})
        
        print(f"\n{'='*60}")
        print(f"BATCH SUMMARY ({self.model_name})")
        print(f"{'='*60}")
        print(f"Successful: {len(successful)}/{len(results)}")
        print(f"Failed: {len(failed)}/{len(results)}")
        print(f"\nTiming Statistics:")
        print(f"  Batch Total Time: {batch_total_time:.2f}s")
        print(f"  Min Time: {min_time:.2f}s")
        print(f"  Max Time: {max_time:.2f}s")
        print(f"  Avg Time: {avg_time:.2f}s")
        print(f"  Total Processing Time: {sum(times):.2f}s")
        print(f"\nTiming saved to: {csv_file}")

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
        
        print(f"\n{'='*60}")
        print(f"Starting batch transcription of {len(files)} files")
        print(f"{'='*60}")
        
        results = []
        batch_start_time = time.time()
        for i, f in enumerate(files, 1):
            print(f"\nFile {i}/{len(files)}: {f.name}")
            video_start_time = time.time()
            try:
                self.transcribe_video(f, output_dir)
                video_elapsed = time.time() - video_start_time
                results.append({"file": f.name, "success": True, "time": video_elapsed})
            except Exception as e:
                video_elapsed = time.time() - video_start_time
                print(f"Failed {f.name}: {e}")
                results.append({"file": f.name, "success": False, "error": str(e), "time": video_elapsed})
        
        batch_total_time = time.time() - batch_start_time
        self._create_batch_summary(results, output_dir, batch_total_time)

if __name__ == "__main__":
    device = get_device()
    transcriber = SpeechT5Transcriber(model_size="large", device=device)
    
    # Batch process
    transcriber.batch_transcribe(folder_path="videos", output_dir="processed")

    # Single video
    # transcriber.transcribe_video(r"videos_test\AkerBP 1.mp4", output_dir="processed")
