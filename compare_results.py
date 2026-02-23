import json
import os
import difflib
from pathlib import Path
from typing import Dict, List, Optional
import argparse

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two strings using SequenceMatcher."""
    if not text1 or not text2:
        return 0.0 if text1 != text2 else 100.0
    
    # Normalize: lowercase and strip extra whitespace
    s1 = " ".join(text1.lower().split())
    s2 = " ".join(text2.lower().split())
    
    matcher = difflib.SequenceMatcher(None, s1, s2)
    return round(matcher.ratio() * 100, 2)

def get_word_count(text: str) -> int:
    return len(text.split())

def find_json_result(base_dir: Path, video_name: str) -> Optional[Path]:
    """
    Search for transcript.json in either:
    - base_dir / video_name / transcript.json
    - base_dir / video_name / transcripts / transcript.json
    """
    options = [
        base_dir / video_name / "transcript.json",
        base_dir / video_name / "transcripts" / "transcript.json"
    ]
    for opt in options:
        if opt.exists():
            return opt
    return None

def main():
    parser = argparse.ArgumentParser(description="Compare ASR results against Whisper-Large-v3")
    parser.add_argument("--ref-dir", default="processed/transcripts/Whisper-Large-v3", help="Reference transcripts directory")
    parser.add_argument("--bench-dir", default="benchmarking", help="Benchmark transcripts directory")
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    bench_dir = Path(args.bench_dir)

    if not ref_dir.exists():
        print(f"Error: Reference directory not found: {ref_dir}")
        return

    # 1. Discover videos in reference
    ref_videos = [d.name for d in ref_dir.iterdir() if d.is_dir()]
    if not ref_videos:
        print(f"No video transcriptions found in {ref_dir}")
        return

    # 2. Discover models
    # Look at both the benchmarking folder and specific requested models in processed/transcripts
    models = {} # name -> path
    
    if bench_dir.exists():
        for d in bench_dir.iterdir():
            if d.is_dir():
                models[d.name] = d
                
    # Add specific models from processed/transcripts
    extra_models = [
        "Qwen-qwen2-audio",
        "Wav2Vec"
    ]
    transcripts_base = ref_dir.parent # This is "processed/transcripts"
    
    for model_name in extra_models:
        model_path = transcripts_base / model_name
        if model_path.exists():
            models[model_name] = model_path
        else:
            print(f"Warning: Could not find extra model output: {model_path}")

    if not models:
        print(f"No models found to compare in {bench_dir} or extra paths.")
        return

    model_list = sorted(list(models.keys()))

    print("\n" + "=" * 90)
    print("  ASR BENCHMARK COMPARISON")
    print("=" * 90)
    print(f"  REFERENCE: {ref_dir.name}")
    print(f"  MODELS   : {', '.join(model_list)}")
    print("=" * 90 + "\n")

    overall_results = []
    
    # Header for the detailed table
    header = f"{'Model':<20} | {'Video':<30} | {'Sim %':>7} | {'WordsDiff':>9} | {'Time (s)':>10}"
    print(header)
    print("-" * len(header))

    model_aggregates = {m: {"total_sim": 0, "count": 0, "total_time": 0} for m in models.keys()}

    for model_name, model_dir in models.items():
        current_model_dir = model_dir
        
        for video in ref_videos:
            # Get reference data
            ref_path = find_json_result(ref_dir, video)
            # Get benchmark data
            bench_path = find_json_result(current_model_dir, video)

            if not ref_path or not bench_path:
                continue

            try:
                with open(ref_path, "r", encoding="utf-8") as f:
                    ref_data = json.load(f)
                with open(bench_path, "r", encoding="utf-8") as f:
                    bench_data = json.load(f)

                ref_text = ref_data.get("text", "")
                bench_text = bench_data.get("text", "")
                
                sim = calculate_similarity(ref_text, bench_text)
                ref_words = get_word_count(ref_text)
                bench_words = get_word_count(bench_text)
                word_diff = bench_words - ref_words
                
                # Processing time (if available)
                bench_time = bench_data.get("processing_time_seconds") or bench_data.get("processing_time", 0)
                
                print(f"{model_name:<20} | {video[:30]:<30} | {sim:>7.2f}% | {word_diff:>+9} | {bench_time:>10.2f}")
                
                model_aggregates[model_name]["total_sim"] += sim
                model_aggregates[model_name]["total_time"] += bench_time
                model_aggregates[model_name]["count"] += 1
                
                overall_results.append({
                    "model": model_name,
                    "video": video,
                    "similarity": sim,
                    "word_diff": word_diff,
                    "time": bench_time
                })

            except Exception as e:
                print(f"Error processing {model}/{video}: {e}")

    # 3. Final Summary Table
    print("\n\n" + "=" * 60)
    print("  SUMMARY STATISTICS (Averages)")
    print("=" * 60)
    summary_header = f"{'Model':<20} | {'Avg Sim %':>10} | {'Sum Time (s)':>12} | {'Videos':>6}"
    print(summary_header)
    print("-" * len(summary_header))

    for model, stats in model_aggregates.items():
        if stats["count"] > 0:
            avg_sim = stats["total_sim"] / stats["count"]
            print(f"{model:<20} | {avg_sim:>9.2f}% | {stats['total_time']:>12.2f} | {stats['count']:>6}")
    
    print("=" * 60 + "\n")
    print(f"Similarity scores compare models to Whisper-Large-v3 (100% means identical text).")

if __name__ == "__main__":
    main()
