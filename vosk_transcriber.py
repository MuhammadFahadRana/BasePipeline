import os
import time
import torch
import json
from pathlib import Path
from transcriber_utils import (
    extract_audio_to_wav, save_results, get_device, ALL_MEDIA
)

class VoskTranscriber:
    def __init__(self, model_size="large", device="cpu"):
        from vosk import Model, KaldiRecognizer, SetLogLevel
        SetLogLevel(-1)
        print(f"Loading Vosk model (en-us)...")
        self.model = Model(lang="en-us")
        self.KaldiRecognizer = KaldiRecognizer
        self.model_name = "Vosk-En"

    def transcribe_video(self, file_path, output_dir="processed"):
        file_path = Path(file_path)
        video_name = file_path.stem.replace(" ", "_")
        model_output_dir = Path(output_dir) / "transcripts" / self.model_name / video_name
        
        print(f"\nTranscribing: {file_path.name}")
        wav_path = extract_audio_to_wav(file_path)
        start_time = time.time()
        
        try:
            import wave
            wf = wave.open(str(wav_path), "rb")
            rec = self.KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)
            
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0: break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    if res.get("text"): results.append(res)
            
            final = json.loads(rec.FinalResult())
            if final.get("text"): results.append(final)
            wf.close()
            
            segments = []
            for r in results:
                words = r.get("result", [])
                segments.append({
                    "start": words[0]["start"] if words else 0,
                    "end": words[-1]["end"] if words else 0,
                    "text": r["text"]
                })
            
            result = {"text": " ".join([r["text"] for r in results]), "segments": segments}
            elapsed = time.time() - start_time
            save_results(result, model_output_dir, video_name, self.model_name, elapsed, file_path.name)
            return result
        finally:
            if wav_path.exists(): wav_path.unlink()

    def batch_transcribe(self, folder_path="videos", output_dir="processed"):
        folder_path = Path(folder_path)
        files = [f for f in folder_path.glob("*.*") if f.suffix.lower() in ALL_MEDIA]
        print(f"Batch processing {len(files)} files...")
        for f in files:
            try: self.transcribe_video(f, output_dir)
            except Exception as e: print(f"Failed {f.name}: {e}")

if __name__ == "__main__":
    device = "cpu" # Vosk is CPU-based
    transcriber = VoskTranscriber(model_size="large", device=device)
    
    transcriber.batch_transcribe(folder_path="videos", output_dir="processed")
    # transcriber.transcribe_video(r"videos_test\AkerBP 1.mp4", output_dir="processed")
