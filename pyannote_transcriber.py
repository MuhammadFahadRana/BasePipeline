import os
import time
import torch
from pathlib import Path
from transcriber_utils import (
    extract_audio_to_wav, save_results, get_device, hf_auth, ALL_MEDIA
)

class PyannoteTranscriber:
    def __init__(self, model_size="large", device="auto", hf_token=None):
        if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Authenticate
        token = hf_token or hf_auth()
        
        from pyannote.audio import Pipeline
        from huggingface_hub.utils import HfHubHTTPError
        
        print(f"Loading pyannote/speaker-diarization-community-1 on {device}...")
        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1", use_auth_token=token
            )
        except (HfHubHTTPError, Exception) as e:
            if "403" in str(e):
                print("\n" + "!"*60)
                print("ACCESS DENIED: pyannote/speaker-diarization-community-1 is a gated model.")
                print("1. Visit https://hf.co/pyannote/speaker-diarization-community-1 and Accept Terms.")
                print("2. If using a FINE-GRAINED token, ensure it has the following permission:")
                print("   'Read access to contents and metadata of public gated repositories'")
                print("!"*60 + "\n")
            self.pipeline = None
            
        if self.pipeline is None:
            print("Warning: Pyannote pipeline failed to load. Diarization will not work.")
        elif device == "cuda":
            self.pipeline.to(torch.device("cuda"))
            
        self.model_name = "Pyannote-Diarization"

    def transcribe_video(self, file_path, output_dir="processed"):
        file_path = Path(file_path)
        video_name = file_path.stem.replace(" ", "_")
        model_output_dir = Path(output_dir) / "transcripts" / self.model_name / video_name
        
        print(f"\nProcessing Diarization: {file_path.name}")
        wav_path = extract_audio_to_wav(file_path)
        start_time = time.time()
        
        try:
            import torchaudio
            waveform, sr = torchaudio.load(str(wav_path))
            output = self.pipeline({"waveform": waveform, "sample_rate": sr})
            
            segments = []
            full_text = []
            for turn, _, speaker in output.itertracks(yield_label=True):
                segments.append({
                    "start": round(turn.start, 3), "end": round(turn.end, 3),
                    "text": f"[{speaker}]", "speaker": speaker
                })
                full_text.append(f"[{speaker}: {turn.start:.1f}s - {turn.end:.1f}s]")
                
            result = {"text": " ".join(full_text), "segments": segments, "type": "diarization"}
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
    transcriber = PyannoteTranscriber(model_size="large", device=device)
    transcriber.batch_transcribe(folder_path="videos", output_dir="processed")
    # transcriber.transcribe_video(r"videos_test\AkerBP 1.mp4", output_dir="processed")
