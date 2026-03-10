import os
import time
import torch
import numpy as np
from pathlib import Path
from transcriber_utils import (
    extract_audio_to_wav,
    save_results,
    get_device,
    hf_auth,
    ALL_MEDIA,
)


class VoxtralTranscriber:
    def __init__(self, model_size="large", device="auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Authenticate
        token = hf_auth()

        from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
        from mistral_common.audio import Audio

        self.Audio = Audio
        repo_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"
        print(f"Loading {repo_id}...")
        self.processor = AutoProcessor.from_pretrained(repo_id, token=token)
        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            repo_id,
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            token=token,
        )
        self.model_name = "Voxtral-Mini-4B"

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
            writer = csv.DictWriter(
                f, fieldnames=["file", "success", "time_s", "error"]
            )
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "file": r["file"],
                        "success": "Yes" if r["success"] else "No",
                        "time_s": round(r["time"], 2),
                        "error": r.get("error", ""),
                    }
                )

        print(f"\n{'=' * 60}")
        print(f"BATCH SUMMARY ({self.model_name})")
        print(f"{'=' * 60}")
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
        model_output_dir = (
            Path(output_dir) / "transcripts" / self.model_name / video_name
        )

        print(f"\nTranscribing: {file_path.name}")
        wav_path = extract_audio_to_wav(file_path)
        start_time = time.time()

        try:
            audio_obj = self.Audio.from_file(str(wav_path), strict=False)

            # Voxtral requires resampling to its specific sample rate
            sample_rate = self.processor.feature_extractor.sampling_rate
            audio_obj.resample(sample_rate)

            # Chunking logic for long videos
            chunk_length_s = 30.0
            chunk_size = int(chunk_length_s * sample_rate)

            # The audio_array is typically shape (1, samples) or (samples,)
            # Mistral Audio.from_file normalizes to a flat list/array, need inner length
            audio_array = audio_obj.audio_array

            # Handle potential dimensionality (squeeze to 1D if needed)
            if isinstance(audio_array, np.ndarray) and len(audio_array.shape) > 1:
                audio_array = audio_array.flatten()
            elif isinstance(audio_array, torch.Tensor) and len(audio_array.shape) > 1:
                audio_array = audio_array.squeeze()

            # We must recreate the batch dim when passing to processor [1, samples]
            if isinstance(audio_array, torch.Tensor):
                audio_array_np = audio_array.numpy()
            else:
                audio_array_np = np.array(audio_array)

            total_samples = len(audio_array_np)
            all_segments = []
            full_text = []

            for start_idx in range(0, total_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, total_samples)
                chunk_samples = audio_array_np[start_idx:end_idx]

                # Expand dims back for the processor if it expects batches
                chunk_input = chunk_samples[np.newaxis, :]

                # Print progress
                chunk_start_s = start_idx / sample_rate
                chunk_end_s = end_idx / sample_rate
                print(
                    f"  Processing chunk {chunk_start_s:.1f}s - {chunk_end_s:.1f}s..."
                )

                inputs = self.processor(chunk_input, return_tensors="pt")
                inputs = inputs.to(self.model.device, dtype=self.model.dtype)

                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=4096)

                chunk_text = self.processor.batch_decode(
                    outputs, skip_special_tokens=True
                )[0].strip()

                if chunk_text:
                    full_text.append(chunk_text)
                    all_segments.append(
                        {
                            "start": round(chunk_start_s, 2),
                            "end": round(chunk_end_s, 2),
                            "text": chunk_text,
                        }
                    )

            final_text = " ".join(full_text)

            result = {
                "text": final_text,
                "segments": all_segments,
            }

            elapsed = time.time() - start_time
            save_results(
                result,
                model_output_dir,
                video_name,
                self.model_name,
                elapsed,
                file_path.name,
            )
            return result
        finally:
            if wav_path.exists():
                wav_path.unlink()

    def batch_transcribe(self, folder_path="videos", output_dir="processed"):
        folder_path = Path(folder_path)
        files = [f for f in folder_path.glob("*.*") if f.suffix.lower() in ALL_MEDIA]

        print(f"\n{'=' * 60}")
        print(f"Starting batch transcription of {len(files)} files")
        print(f"{'=' * 60}")

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
                results.append(
                    {
                        "file": f.name,
                        "success": False,
                        "error": str(e),
                        "time": video_elapsed,
                    }
                )

        batch_total_time = time.time() - batch_start_time
        self._create_batch_summary(results, output_dir, batch_total_time)


if __name__ == "__main__":
    device = get_device()
    transcriber = VoxtralTranscriber(model_size="large", device=device)
    # transcriber.batch_transcribe(folder_path="videos", output_dir="processed")
    transcriber.transcribe_video(r"videos\30Min.mp4", output_dir="processed")
