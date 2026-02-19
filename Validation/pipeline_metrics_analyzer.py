import json
import csv
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "processed" / "results"
EVAL_DIR = BASE_DIR / "processed" / "ground_truth_evaluation"
OUTPUT_CSV = BASE_DIR / "pipeline_metrics.csv"

def get_video_duration(results_data: Dict[str, Any]) -> float:
    """Extracts video duration from the last segment's end time."""
    try:
        # Check for whisper segments
        whisper_results = results_data.get("transcription", {})
        segments = whisper_results.get("segments", [])
        
        if segments:
            return segments[-1].get("end", 0.0)
        
        # Fallback to scenes if no transcription segments
        scenes = results_data.get("scenes", [])
        if scenes:
             return scenes[-1].get("end_time", 0.0)

        return 0.0
    except Exception as e:
        print(f"Error extracting duration: {e}")
        return 0.0

def get_processing_metrics(video_path: Path) -> Dict[str, Any]:
    """Reads results.json to extract processing time and duration."""
    results_file = video_path / "results.json"
    metrics = {
        "processing_time": 0.0,
        "video_duration": 0.0,
        "real_time_factor": 0.0
    }

    if not results_file.exists():
        return metrics

    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Get processing duration from processing_info
        processing_info = data.get("processing_info", {})
        metrics["processing_time"] = processing_info.get("processing_duration", 0.0)
        
        # Get video duration
        metrics["video_duration"] = get_video_duration(data)
        
        # Calculate Real Time Factor (RTF)
        if metrics["video_duration"] > 0:
            metrics["real_time_factor"] = metrics["processing_time"] / metrics["video_duration"]
            
    except Exception as e:
        print(f"Error reading {results_file.name}: {e}")

    return metrics

def get_quality_metrics(video_name: str) -> Dict[str, Any]:
    """Reads evaluation file to get WER, CER, and Accuracy."""
    # The evaluation file might be named slightly differently, let's try strict matching first
    # Based on previous `ground_truth_eval.py`, typical name is `{video_name}_evaluation.json`
    eval_file = EVAL_DIR / f"{video_name}_evaluation.json"
    
    metrics = {
        "WER": "N/A",
        "CER": "N/A",
        "Accuracy": "N/A"
    }

    if not eval_file.exists():
        return metrics

    try:
        with open(eval_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Previous eval script output format:
        # { "video": "...", "evaluations": [ { "model": "...", "wer": 0.1, ... } ] }
        evals = data.get("evaluations", [])
        if evals:
            # Assuming we want the first model's evaluation (usually Whisper-Large-v3)
            first_eval = evals[0]
            metrics["WER"] = f"{first_eval.get('wer', 0):.2f}%"
            metrics["CER"] = f"{first_eval.get('cer', 0):.2f}%"
            metrics["Accuracy"] = f"{first_eval.get('accuracy', 0):.2f}%"
            
    except Exception as e:
        print(f"Error reading evaluation {eval_file.name}: {e}")

    return metrics

def main():
    if not RESULTS_DIR.exists():
        print(f"Error: Results directory not found at {RESULTS_DIR}")
        return

    print(f"Scanning metrics from: {RESULTS_DIR}")
    print(f"{'Video Name':<50} | {'Dur (s)':<10} | {'Proc (s)':<10} | {'RTF':<6} | {'WER':<8} | {'CER':<8} | {'Acc':<8}")
    print("-" * 120)

    all_metrics = []

    # Iterate over directories in processed/results
    video_dirs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()])
    
    for video_dir in video_dirs:
        video_name = video_dir.name
        
        # 1. Get Processing Metrics
        proc_metrics = get_processing_metrics(video_dir)
        
        # 2. Get Quality Metrics
        qual_metrics = get_quality_metrics(video_name)
        
        # Combine
        combined = {
            "Video": video_name,
            "Duration_sec": round(proc_metrics["video_duration"], 2),
            "Processing_sec": round(proc_metrics["processing_time"], 2),
            "RTF": round(proc_metrics["real_time_factor"], 2),
            **qual_metrics
        }
        all_metrics.append(combined)

        print(f"{video_name:<50} | {combined['Duration_sec']:<10} | {combined['Processing_sec']:<10} | {combined['RTF']:<6} | {combined['WER']:<8} | {combined['CER']:<8} | {combined['Accuracy']:<8}")

    # Write to CSV
    if all_metrics:
        headers = ["Video", "Duration_sec", "Processing_sec", "RTF", "WER", "CER", "Accuracy"]
        try:
            with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(all_metrics)
            print(f"\nMetrics saved to: {OUTPUT_CSV}")
        except Exception as e:
            print(f"\nError saving CSV: {e}")
    else:
        print("\nNo metrics found.")

if __name__ == "__main__":
    main()
