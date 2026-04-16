from pathlib import Path
import re
import time
import torch

from transcriber_utils import (
    extract_audio_to_wav, save_results, get_device, hf_auth, ALL_MEDIA
)

class VibeVoiceTranscriber:
    """
    VibeVoice-ASR transcriber for long-form speech recognition.
    
    VibeVoice-ASR is designed for 60-minute single-pass processing with:
    - Speaker diarization (Who)
    - Timestamping (When)  
    - Transcription (What)
    
    Installation:
    1. Basic installation (requires transformers with trust_remote_code):
       pip install transformers torch torchaudio
    
    2. Recommended (with flash-attention for GPU speedup):
       Option A: Pre-built wheels (easiest)
         pip install flash-attn
       
       Option B: From source (requires CUDA toolkit)
         pip install git+https://github.com/microsoft/VibeVoice.git
         pip install flash-attn --no-build-isolation
         
       If CUDA_HOME is not set, set it first:
         SET CUDA_HOME=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1
    
    Reference: https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr.md
    """
    def __init__(self, model_size="large", device="auto"):
        if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # Authenticate
        token = hf_auth()
        
        model_id = "microsoft/VibeVoice-ASR"
        print(f"Loading {model_id} on {device}...")
        
        try:
            # Try using the official VibeVoice library (requires installation from GitHub)
            try:
                from vibevoice.models.vibevoice_asr import VibeVoiceASRPipeline
                print("[OK] Using official VibeVoice library...")
                self.pipeline = VibeVoiceASRPipeline(model_id=model_id, device=device)
                self.use_official_lib = True
            except ImportError:
                print("[INFO] Official VibeVoice library not found. Using transformers fallback...")
                self.use_official_lib = False
                
                # Fallback to transformers with trust_remote_code
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                
                if device == "cpu" and self.model: 
                    self.model = self.model.to("cpu")
            
            print("[OK] VibeVoice-ASR loaded successfully")
        except Exception as e:
            print("\n" + "!"*70)
            print("Failed to load VibeVoice-ASR model.")
            
            error_msg = str(e).lower()
            if "vibevoice" in error_msg:
                print("\nThe 'vibevoice' architecture is not recognized in your transformers version.")
                print("\nSolution: Install the official VibeVoice library from GitHub:")
                print("  pip install git+https://github.com/microsoft/VibeVoice.git")
            elif "cuda_home" in error_msg or "nvcc" in error_msg or "flash" in error_msg:
                print("\nFlash-attention setup issue detected.")
                print("\nFlash-attention is OPTIONAL for VibeVoice-ASR.")
                print("\nTry one of these approaches:")
                print("\n  Approach 1: Skip flash-attention (uses standard attention)")
                print("    pip install transformers torch torchaudio")
                print("\n  Approach 2: Install flash-attention from pre-built wheels")
                print("    pip install flash-attn --prefer-binary")
                print("\n  Approach 3: Set CUDA_HOME and build from source")
                print("    SET CUDA_HOME=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.X")
                print("    pip install flash-attn --no-build-isolation")
            else:
                print(f"Error: {e}")
                print("\nMake sure you have:")
                print("  - transformers library installed")
                print("  - torch and torchaudio for audio processing")
            print("!"*70 + "\n")
            raise
        
        self.model_name = "VibeVoice-ASR"

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
        
        print(f"\nTranscribing (VibeVoice): {file_path.name}")
        wav_path = extract_audio_to_wav(file_path)
        start_time = time.time()
        
        try:
            if self.use_official_lib:
                # Use official VibeVoice library
                result = self.pipeline(str(wav_path))
                # Extract text and segments from pipeline output
                text = result.get("text", "")
                segments = result.get("segments", [])
            else:
                # Use transformers fallback
                import torchaudio
                waveform, sr = torchaudio.load(str(wav_path))
                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)
                    sr = 16000
                
                # Prepare input - VibeVoice expects raw waveform
                inputs = {"input_features": waveform.unsqueeze(0).to(self.model.device)}
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=512)
                
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
    try:
        device = get_device()
        transcriber = VibeVoiceTranscriber(model_size="large", device=device)
        transcriber.batch_transcribe(folder_path="videos", output_dir="processed")
        # transcriber.transcribe_video(r"videos_test\AkerBP 1.mp4", output_dir="processed")
    except Exception as e:
        error_msg = str(e).lower()
        if "vibevoice" in error_msg or "not found" in error_msg:
            print("\n" + "="*70)
            print("VibeVoice-ASR Setup Required")
            print("="*70)
            print("\n[ERROR] VibeVoice-ASR model architecture not recognized.")
            print("\n[SOLUTION] Install VibeVoice library from GitHub:")
            print("\n  pip install git+https://github.com/microsoft/VibeVoice.git")
            print("\nOptional (for GPU acceleration):")
            print("  pip install flash-attn --prefer-binary")
            print("\nDocumentation: https://github.com/microsoft/VibeVoice")
            print("="*70 + "\n")
        elif "cuda_home" in error_msg or "nvcc" in error_msg:
            print("\n" + "="*70)
            print("CUDA Setup Issue with flash-attn")
            print("="*70)
            print("\n[WARNING] flash-attn installation failed due to missing CUDA compiler.")
            print("\n[NOTE] Flash-attention is OPTIONAL for VibeVoice.")
            print("\nTry this approach (uses standard attention, no flash-attn needed):")
            print("\n  pip install transformers torch torchaudio")
            print("\nAlternatively, if you want flash-attn:")
            print("  1. Install pre-built wheels (easiest):")
            print("     pip install flash-attn --prefer-binary")
            print("  2. Or set CUDA_HOME and build from source:")
            print("     SET CUDA_HOME=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.X")
            print("     pip install flash-attn --no-build-isolation")
            print("="*70 + "\n")
        else:
            print(f"\n[ERROR] Unexpected error: {e}")
        
        import sys
        sys.exit(1)
