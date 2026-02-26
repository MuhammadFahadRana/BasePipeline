import os
import time
import torch
from pathlib import Path
from transcriber_utils import (
    extract_audio_to_wav, save_results, get_device, hf_auth, ALL_MEDIA
)

class MedASRTranscriber:
    def __init__(self, model_size="large", device="auto"):
        if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Authenticate
        token = hf_auth()
        
        from transformers import pipeline
        from huggingface_hub.utils import HfHubHTTPError
        
        print(f"Loading google/medasr on {device}...")
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model="google/medasr",
                device=0 if device == "cuda" else -1,
                token=token
            )
        except (HfHubHTTPError, Exception) as e:
            if "403" in str(e):
                print("\n" + "!"*60)
                print("ACCESS DENIED: google/medasr is a gated model.")
                print("1. Visit https://huggingface.co/google/medasr and Accept Terms.")
                print("2. If using a FINE-GRAINED token, ensure it has the following permission:")
                print("   'Read access to contents and metadata of public gated repositories'")
                print("!"*60 + "\n")
            raise e
            
        self.model_name = "Google-MedASR"

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
        
        print(f"\nTranscribing (Medical): {file_path.name}")
        wav_path = extract_audio_to_wav(file_path)
        start_time = time.time()
        
        try:
            result_raw = self.pipe(str(wav_path), chunk_length_s=20, stride_length_s=2)
            text = result_raw.get("text", "")
            chunks = result_raw.get("chunks", [])
            segments = []
            for c in chunks:
                ts = c.get("timestamp", (0, 0))
                segments.append({
                    "start": ts[0] or 0.0, "end": ts[1] or 0.0, "text": c.get("text", "")
                })
            
            if not segments and text:
                segments = [{"start": 0.0, "end": 0.0, "text": text}]
                
            result = {"text": text, "segments": segments}
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
    transcriber = MedASRTranscriber(model_size="large", device=device)
    # transcriber.batch_transcribe(folder_path="videos", output_dir="processed")
    transcriber.transcribe_video(r"videos_test\AkerBP 1.mp4", output_dir="processed")
