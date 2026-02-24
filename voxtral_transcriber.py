import os
import time
import torch
from pathlib import Path
from transcriber_utils import (
    extract_audio_to_wav, save_results, get_device, ALL_MEDIA
)

class VoxtralTranscriber:
    def __init__(self, model_size="large", device="auto"):
        if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
        from mistral_common.tokens.tokenizers.audio import Audio
        
        self.Audio = Audio
        repo_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"
        print(f"Loading {repo_id}...")
        self.processor = AutoProcessor.from_pretrained(repo_id)
        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            repo_id, device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        self.model_name = "Voxtral-Mini-4B"

    def transcribe_video(self, file_path, output_dir="processed"):
        file_path = Path(file_path)
        video_name = file_path.stem.replace(" ", "_")
        model_output_dir = Path(output_dir) / "transcripts" / self.model_name / video_name
        
        print(f"\nTranscribing: {file_path.name}")
        wav_path = extract_audio_to_wav(file_path)
        start_time = time.time()
        
        try:
            audio_obj = self.Audio.from_file(str(wav_path), strict=False)
            audio_obj.resample(self.processor.feature_extractor.sampling_rate)
            inputs = self.processor(audio_obj.audio_array, return_tensors="pt")
            inputs = inputs.to(self.model.device, dtype=self.model.dtype)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=4096)
            
            text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            result = {"text": text.strip(), "segments": [{"start":0, "end":0, "text":text.strip()}]}
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
    transcriber = VoxtralTranscriber(model_size="large", device=device)
    # transcriber.batch_transcribe(folder_path="videos", output_dir="processed")
    transcriber.transcribe_video(r"videos_test\AkerBP 1.mp4", output_dir="processed")
