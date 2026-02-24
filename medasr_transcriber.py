import os
import time
import torch
from pathlib import Path
from transcriber_utils import (
    extract_audio_to_wav, save_results, get_device, ALL_MEDIA
)

class MedASRTranscriber:
    def __init__(self, model_size="large", device="auto"):
        if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        from transformers import pipeline
        print(f"Loading google/medasr on {device}...")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="google/medasr",
            device=0 if device == "cuda" else -1,
        )
        self.model_name = "Google-MedASR"

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
        for f in files:
            try: self.transcribe_video(f, output_dir)
            except Exception as e: print(f"Failed {f.name}: {e}")

if __name__ == "__main__":
    device = get_device()
    transcriber = MedASRTranscriber(model_size="large", device=device)
    # transcriber.batch_transcribe(folder_path="videos_test", output_dir="processed")
    transcriber.transcribe_video(r"videos_test\AkerBP 1.mp4", output_dir="processed")
