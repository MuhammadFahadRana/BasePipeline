import json
import os
from pathlib import Path
from datetime import datetime

def recover_timings():
    base_dir = Path("processed")
    results_root = base_dir / "results"
    
    if not results_root.exists():
        print("No results found.")
        return

    print(f"Scanning for results in: {results_root.absolute()}")
    
    for video_dir in results_root.iterdir():
        if not video_dir.is_dir():
            continue
            
        results_file = video_dir / "results.json"
        if not results_file.exists():
            continue
            
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception as e:
            print(f"Error reading {results_file}: {e}")
            continue
            
        # Check if processing_duration already exists
        if "processing_info" in results and "processing_duration" in results["processing_info"]:
            if results["processing_info"]["processing_duration"] > 0:
                print(f"✓ Skipping {video_dir.name}: Duration already exists ({results['processing_info']['processing_duration']}s)")
                continue

        # Estimate duration from timestamps
        # We look for files related to this video name
        video_name = video_dir.name
        related_files = []
        
        # 1. results.json itself
        related_files.append(results_file)
        
        # 2. Transcripts
        transcripts_root = base_dir / "transcripts"
        if transcripts_root.exists():
            for model_dir in transcripts_root.iterdir():
                if model_dir.is_dir():
                    v_trans_dir = model_dir / video_name
                    if v_trans_dir.exists():
                        related_files.extend(v_trans_dir.glob("*"))
        
        # 3. Scenes
        scenes_v_dir = base_dir / "scenes" / video_name
        if scenes_v_dir.exists():
            related_files.extend(scenes_v_dir.glob("*"))
            
        if not related_files:
            print(f"? No related files found for {video_name}")
            continue
            
        # Get mtimes
        mtimes = [os.path.getmtime(f) for f in related_files if f.is_file()]
        if not mtimes:
            continue
            
        start_ts = min(mtimes)
        end_ts = max(mtimes)
        duration = end_ts - start_ts
        
        # If duration is 0, it means all files were created at the same time (unlikely but possible for tiny videos)
        # Or if it's too large, it might be from different runs.
        # But for recovery, we take what we get or set a minimum.
        if duration < 1.0:
            duration = 1.0 # Minimal placeholder
            
        print(f"Found {video_name}: Estimated duration {duration:.2f}s (from {len(related_files)} files)")
        
        # Update results
        if "processing_info" not in results:
            results["processing_info"] = {}
            
        results["processing_info"]["processing_duration"] = round(duration, 2)
        results["processing_info"]["duration_is_estimated"] = True
        
        try:
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Updated {results_file}")
        except Exception as e:
            print(f"  ! Failed to update {results_file}: {e}")

if __name__ == "__main__":
    recover_timings()
