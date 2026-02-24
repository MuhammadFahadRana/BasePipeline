import os
import time
import torch
import re
from pathlib import Path
from transcriber_utils import (
    extract_audio_to_wav, save_results, get_device, ALL_MEDIA
)

class VibeVoiceTranscriber:
    def __init__(self, model_size="large", device="auto"):
        if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        from transformers import AutoModelForCausalLM, AutoProcessor
        model_id = "microsoft/VibeVoice-ASR"
        print(f"Loading {model_id} on {device}...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        if device == "cpu": self.model = self.model.to("cpu")
        self.model_name = "VibeVoice-ASR"

    def transcribe_video(self, file_path, output_dir="processed"):
        file_path = Path(file_path)
        video_name = file_path.stem.replace(" ", "_")
        model_output_dir = Path(output_dir) / "transcripts" / self.model_name / video_name
        
        print(f"\nTranscribing (VibeVoice): {file_path.name}")
        wav_path = extract_audio_to_wav(file_path)
        start_time = time.time()
        
        try:
            import torchaudio
            waveform, sr = torchaudio.load(str(wav_path))
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
                sr = 16000
            
            inputs = self.processor(waveform.squeeze(0).numpy(), sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=8192)
            
            text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            segments = self._parse_output(text)
            
            result = {"text": text.strip(), "segments": segments}
            elapsed = time.time() - start_time
            save_results(result, model_output_dir, video_name, self.model_name, elapsed, file_path.name)
            return result
        finally:
            if wav_path.exists(): wav_path.unlink()

    def _parse_output(self, text):
        segments = []
        pattern = r'\[([^\]]+)\]\s*<([\d.]+)s?\s*-\s*([\d.]+)s?>\s*(.*?)(?=\[|$)'
        matches = re.finditer(pattern, text, re.DOTALL)
        for m in matches:
            segments.append({
                "speaker": m.group(1), "start": float(m.group(2)),
                "end": float(m.group(3)), "text": m.group(4).strip()
            })
        return segments if segments else [{"start":0, "end":0, "text":text.strip()}]

    def batch_transcribe(self, folder_path="videos", output_dir="processed"):
        folder_path = Path(folder_path)
        files = [f for f in folder_path.glob("*.*") if f.suffix.lower() in ALL_MEDIA]
        for f in files:
            try: self.transcribe_video(f, output_dir)
            except Exception as e: print(f"Failed {f.name}: {e}")

if __name__ == "__main__":
    device = get_device()
    transcriber = VibeVoiceTranscriber(model_size="large", device=device)
    # transcriber.batch_transcribe(folder_path="videos_test", output_dir="processed")
    transcriber.transcribe_video(r"videos_test\AkerBP 1.mp4", output_dir="processed")
