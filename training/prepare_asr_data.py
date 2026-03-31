"""
prepare_asr_data.py

Prepare audio-transcript training pairs for Whisper fine-tuning.

Problem: Ground truth files have full transcripts WITHOUT timestamps.
Solution: Use existing Whisper segment timestamps as alignment anchors,
           then pair each audio chunk with the ground-truth text.

Two modes:
  1. "aligned"  — Aligns GT text to Whisper segments using fuzzy matching.
                    Each Whisper segment's timestamps define the audio chunk,
                    and the GT text is used as the target transcript.
  2. "full"     — Uses the full audio file + full GT transcript as one
                    long training example (for models that handle long audio).

Usage:
    python -m training.prepare_asr_data                             # default aligned mode
    python -m training.prepare_asr_data --mode full                 # full audio mode
    python -m training.prepare_asr_data --max-chunk-sec 30          # limit chunk size
    python -m training.prepare_asr_data --dry-run                   # preview without writing
"""

import json
import argparse
import sys
import subprocess
import os
import difflib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class ASRTrainingSample:
    """One audio-transcript training pair."""
    audio_path: str          # path to .wav chunk
    transcript: str          # ground-truth text for that chunk
    video: str               # source video name
    start_time: float        # chunk start in original video (seconds)
    end_time: float          # chunk end in original video (seconds)
    duration: float          # chunk duration (seconds)
    language: str            # e.g. "en", "no"
    split: str               # "train" or "eval"
    alignment_score: float = 1.0  # 0..1 similarity score for aligned mode


# ──────────────────────────────────────────────
# Audio slicing
# ──────────────────────────────────────────────

def slice_audio(
    audio_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    target_sr: int = 16000,
) -> bool:
    """
    Extract an audio chunk from start_sec to end_sec using ffmpeg.
    Outputs 16kHz mono WAV suitable for Whisper.
    """
    ffmpeg_cmd = "ffmpeg"
    try:
        import imageio_ffmpeg
        ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_cmd, "-y",
        "-i", str(audio_path),
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-ac", "1",
        "-ar", str(target_sr),
        "-sample_fmt", "s16",
        "-vn",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def extract_full_audio(video_path: Path, output_path: Path, target_sr: int = 16000) -> bool:
    """Extract full audio from a video file to WAV."""
    ffmpeg_cmd = "ffmpeg"
    try:
        import imageio_ffmpeg
        ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_cmd, "-y",
        "-i", str(video_path),
        "-ac", "1",
        "-ar", str(target_sr),
        "-sample_fmt", "s16",
        "-vn",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


# ──────────────────────────────────────────────
# Text alignment (fuzzy matching GT to Whisper segments)
# ──────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)      # collapse whitespace
    return text


def text_similarity(a: str, b: str) -> float:
    """Compute robust similarity between two transcript snippets (0..1)."""
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    if not a_norm or not b_norm:
        return 0.0

    seq_ratio = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()

    a_words = set(a_norm.split())
    b_words = set(b_norm.split())
    overlap = len(a_words & b_words) / max(1, len(a_words | b_words))

    # Sequence similarity catches ordering; overlap catches lexical agreement.
    return 0.6 * seq_ratio + 0.4 * overlap


def align_gt_to_segments(
    gt_text: str,
    whisper_segments: List[Dict],
    min_overlap_ratio: float = 0.3,
) -> List[Tuple[Dict, str, float]]:
    """
    Align ground-truth transcript text to Whisper segment timestamps.

    Strategy: Walk through the GT text and Whisper segments in parallel.
    For each Whisper segment, find the best-matching portion of GT text.
    Uses a sliding window approach on the normalized GT text.

    Returns:
        List of (whisper_segment, gt_text_for_that_segment, similarity_score) tuples
    """
    if not whisper_segments:
        return []

    gt_words = gt_text.split()
    total_gt_words = len(gt_words)

    if total_gt_words == 0:
        return []

    aligned_pairs = []
    gt_word_cursor = 0  # tracks where we are in the GT text

    for seg in whisper_segments:
        seg_text = seg.get("text", "").strip()
        if not seg_text:
            continue

        seg_word_count = len(seg_text.split())

        # Estimate how many GT words this segment should cover
        # Use proportional allocation based on segment duration vs total duration
        total_duration = whisper_segments[-1].get("end", 1) - whisper_segments[0].get("start", 0)
        seg_duration = seg.get("end", 0) - seg.get("start", 0)

        if total_duration > 0:
            expected_words = int((seg_duration / total_duration) * total_gt_words)
        else:
            expected_words = seg_word_count

        # Use a search window around the expected size.
        window_size = max(seg_word_count, expected_words, 3)
        start_idx = gt_word_cursor
        end_idx = min(gt_word_cursor + window_size + 8, total_gt_words)

        if start_idx >= total_gt_words:
            # We've consumed all GT text
            break

        best_text = ""
        best_advance = 0
        best_score = -1.0

        min_advance = max(1, min(seg_word_count // 2, total_gt_words - start_idx))
        max_advance = max(min_advance, end_idx - start_idx)

        for advance in range(min_advance, max_advance + 1):
            candidate = " ".join(gt_words[start_idx : start_idx + advance])
            score = text_similarity(seg_text, candidate)
            if score > best_score:
                best_score = score
                best_text = candidate
                best_advance = advance

        if best_score >= min_overlap_ratio and best_text.strip():
            aligned_pairs.append((seg, best_text, round(best_score, 4)))
            gt_word_cursor += best_advance
        else:
            # Skip low-confidence alignment and move cursor conservatively.
            gt_word_cursor += max(1, min(seg_word_count, total_gt_words - gt_word_cursor))

    return aligned_pairs


def merge_short_segments(
    aligned_pairs: List[Tuple[Dict, str, float]],
    min_duration: float = 1.0,
    max_duration: float = 30.0,
) -> List[Tuple[float, float, str, float]]:
    """
    Merge very short aligned segments into longer chunks and split very long ones.

    Returns:
        List of (start_time, end_time, merged_gt_text, merged_alignment_score) tuples
    """
    if not aligned_pairs:
        return []

    merged = []
    current_start = aligned_pairs[0][0].get("start", 0)
    current_end = aligned_pairs[0][0].get("end", 0)
    current_text_parts = [aligned_pairs[0][1]]
    current_scores = [aligned_pairs[0][2]]

    for seg, gt_text, score in aligned_pairs[1:]:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        current_duration = current_end - current_start

        # If adding this segment would exceed max_duration, flush current
        if current_duration + (seg_end - seg_start) > max_duration:
            merged.append((
                current_start,
                current_end,
                " ".join(current_text_parts),
                sum(current_scores) / max(1, len(current_scores)),
            ))
            current_start = seg_start
            current_end = seg_end
            current_text_parts = [gt_text]
            current_scores = [score]
        else:
            current_end = seg_end
            current_text_parts.append(gt_text)
            current_scores.append(score)

    # Flush remaining
    if current_text_parts:
        merged.append((
            current_start,
            current_end,
            " ".join(current_text_parts),
            sum(current_scores) / max(1, len(current_scores)),
        ))

    # Post-merge: split any remaining oversized chunks
    final = []
    for start, end, text, score in merged:
        duration = end - start
        if duration <= max_duration:
            final.append((start, end, text, score))
        else:
            # Split into roughly equal parts
            n_parts = int(duration / max_duration) + 1
            words = text.split()
            words_per_part = max(1, len(words) // n_parts)
            time_per_part = duration / n_parts

            for i in range(n_parts):
                part_start = start + i * time_per_part
                part_end = start + (i + 1) * time_per_part
                part_words = words[i * words_per_part : (i + 1) * words_per_part]
                if part_words:
                    final.append((part_start, part_end, " ".join(part_words), score))

    return final


# ──────────────────────────────────────────────
# Dataset building
# ──────────────────────────────────────────────

def find_video_file(video_name: str, search_dirs: List[str]) -> Optional[Path]:
    """Find the original video file by name across search directories."""
    from transcriber_utils import VIDEO_EXTENSIONS, AUDIO_EXTENSIONS

    all_exts = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            continue

        # Try exact name with each extension (fast, no directory listing)
        for ext in all_exts:
            candidate = search_path / f"{video_name}{ext}"
            if candidate.exists():
                return candidate

    return None


def build_asr_dataset(
    gt_dir: str = "ground_truth",
    processed_dir: str = "processed",
    video_dirs: Optional[List[str]] = None,
    output_dir: str = "training/asr_data",
    mode: str = "aligned",
    max_chunk_sec: float = 30.0,
    min_chunk_sec: float = 1.0,
    eval_ratio: float = 0.15,
    min_alignment_similarity: float = 0.35,
    min_words_per_second: float = 0.8,
    max_words_per_second: float = 6.0,
    dry_run: bool = False,
) -> Dict:
    """
    Build the ASR training dataset.

    Args:
        gt_dir: Path to ground_truth/ folder
        processed_dir: Path to processed/ folder (for Whisper segment timestamps)
        video_dirs: Directories to search for original video/audio files
        output_dir: Where to save audio chunks and manifest
        mode: "aligned" (chunk by Whisper segments) or "full" (whole audio)
        max_chunk_sec: Maximum chunk duration in seconds
        min_chunk_sec: Minimum chunk duration in seconds
        eval_ratio: Fraction of samples held out for evaluation
        dry_run: Preview without writing files

    Returns:
        Statistics dict
    """
    import random
    random.seed(42)

    gt_dir = Path(gt_dir)
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)

    if video_dirs is None:
        video_dirs = [
            "videos",                       # primary video folder
            "videos_test",                  # test videos
            ".",                            # project root
            "test_videos",                  # common test folder
            str(processed_dir / "temp_audio"),  # pre-extracted audio
        ]

    # Find all ground truth files
    gt_files = sorted(gt_dir.glob("*_gt.json"))
    if not gt_files:
        print(f"No ground truth files found in {gt_dir}")
        return {"error": "no_gt_files"}

    print(f"\n{'=' * 60}")
    print(f"ATLAS ASR Training Data Preparation")
    print(f"{'=' * 60}")
    print(f"Ground truth files: {len(gt_files)}")
    print(f"Mode: {mode}")
    print(f"Max chunk: {max_chunk_sec}s | Min chunk: {min_chunk_sec}s")
    print(f"{'=' * 60}\n")

    all_samples: List[ASRTrainingSample] = []
    stats = {
        "total_gt_files": len(gt_files),
        "processed": 0,
        "skipped_no_video": 0,
        "skipped_no_segments": 0,
        "skipped_low_alignment": 0,
        "skipped_bad_speaking_rate": 0,
        "total_samples": 0,
        "total_duration_sec": 0,
    }

    for gt_file in gt_files:
        try:
            with open(gt_file, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  [{gt_file.stem}] Invalid JSON, skipping: {e}")
            continue

        video_name = gt_data.get("video", gt_file.stem.replace("_gt", ""))
        gt_transcript_parts = gt_data.get("ground_truth_transcript", [])
        # Collapse all newlines and multiple spaces into a single space
        gt_text = " ".join(gt_transcript_parts)
        gt_text = re.sub(r"\s+", " ", gt_text).strip()

        if not gt_text:
            print(f"  [{video_name}] No transcript text, skipping")
            continue

        print(f"  [{video_name}] GT: {len(gt_text.split())} words")

        # Find video/audio file
        video_path = find_video_file(video_name, video_dirs)
        if video_path is None:
            print(f"    -> Video not found, skipping")
            stats["skipped_no_video"] += 1
            continue

        if mode == "full":
            # Full audio mode: one sample per video
            audio_out = output_dir / "audio" / f"{video_name}.wav"

            if not dry_run:
                success = extract_full_audio(video_path, audio_out)
                if not success:
                    print(f"    -> Audio extraction failed, skipping")
                    continue

            sample = ASRTrainingSample(
                audio_path=str(audio_out),
                transcript=gt_text,
                video=video_name,
                start_time=0,
                end_time=0,  # unknown without probing
                duration=0,
                language="en",  # default, can be detected
                split="train",
            )
            all_samples.append(sample)
            stats["processed"] += 1

        elif mode == "aligned":
            # Find Whisper segments for this video
            whisper_segments = _find_whisper_segments(video_name, processed_dir)

            if not whisper_segments:
                print(f"    -> No Whisper segments found, skipping")
                stats["skipped_no_segments"] += 1
                continue

            # Detect language from processed results
            language = _detect_language(video_name, processed_dir)

            # Align GT text to Whisper segment timestamps
            aligned = align_gt_to_segments(
                gt_text,
                whisper_segments,
                min_overlap_ratio=min_alignment_similarity,
            )
            print(f"    -> Aligned {len(aligned)} segments")

            # Merge short segments and split long ones
            chunks = merge_short_segments(
                aligned,
                min_duration=min_chunk_sec,
                max_duration=max_chunk_sec,
            )
            print(f"    -> {len(chunks)} chunks after merging "
                  f"({min_chunk_sec}–{max_chunk_sec}s)")

            for i, (start, end, chunk_text, alignment_score) in enumerate(chunks):
                if not chunk_text.strip():
                    continue

                duration = end - start
                if duration < min_chunk_sec:
                    continue

                if alignment_score < min_alignment_similarity:
                    stats["skipped_low_alignment"] += 1
                    continue

                words = len(chunk_text.split())
                wps = words / max(duration, 1e-6)
                if wps < min_words_per_second or wps > max_words_per_second:
                    stats["skipped_bad_speaking_rate"] += 1
                    continue

                chunk_audio = output_dir / "audio" / f"{video_name}_chunk_{i:04d}.wav"

                if not dry_run:
                    success = slice_audio(video_path, chunk_audio, start, end)
                    if not success:
                        continue

                sample = ASRTrainingSample(
                    audio_path=str(chunk_audio),
                    transcript=chunk_text,
                    video=video_name,
                    start_time=round(start, 3),
                    end_time=round(end, 3),
                    duration=round(duration, 3),
                    language=language,
                    split="train",
                    alignment_score=round(alignment_score, 4),
                )
                all_samples.append(sample)
                stats["total_duration_sec"] += duration

            stats["processed"] += 1

    if not all_samples:
        print("\nNo samples generated!")
        return stats

    # Assign train/eval splits
    random.shuffle(all_samples)
    eval_count = max(1, int(len(all_samples) * eval_ratio))
    for i, sample in enumerate(all_samples):
        sample.split = "eval" if i < eval_count else "train"

    # Sort back by video + time for readability
    all_samples.sort(key=lambda s: (s.video, s.start_time))

    stats["total_samples"] = len(all_samples)
    stats["train_samples"] = sum(1 for s in all_samples if s.split == "train")
    stats["eval_samples"] = sum(1 for s in all_samples if s.split == "eval")
    stats["total_duration_min"] = round(stats["total_duration_sec"] / 60, 1)

    # Save manifest
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "created_at": datetime.now().isoformat(),
            "mode": mode,
            "stats": stats,
            "samples": [asdict(s) for s in all_samples],
        }
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"\nManifest saved to: {manifest_path}")

        # Also save HuggingFace-compatible JSONL files
        for split in ["train", "eval"]:
            jsonl_path = output_dir / f"{split}.jsonl"
            split_samples = [s for s in all_samples if s.split == split]
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for s in split_samples:
                    f.write(json.dumps({
                        "audio": s.audio_path,
                        "sentence": s.transcript,
                        "language": s.language,
                    }, ensure_ascii=False) + "\n")
            print(f"  {split}.jsonl: {len(split_samples)} samples")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"ASR DATA PREPARATION {'(DRY RUN) ' if dry_run else ''}COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Videos processed: {stats['processed']}")
    print(f"  Videos skipped (no file):     {stats['skipped_no_video']}")
    print(f"  Videos skipped (no segments): {stats['skipped_no_segments']}")
    print(f"  Chunks skipped (low align):   {stats['skipped_low_alignment']}")
    print(f"  Chunks skipped (bad WPS):     {stats['skipped_bad_speaking_rate']}")
    print(f"  Total samples:    {stats['total_samples']}")
    print(f"  Train samples:    {stats.get('train_samples', 0)}")
    print(f"  Eval samples:     {stats.get('eval_samples', 0)}")
    print(f"  Total duration:   {stats.get('total_duration_min', 0)} minutes")
    print(f"{'=' * 60}")

    return stats


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _find_whisper_segments(video_name: str, processed_dir: Path) -> List[Dict]:
    """Find Whisper transcript segments for a video from processed results."""
    # Try multiple layouts
    patterns = [
        processed_dir / "*/*/results.json",       # processed/<Model>/<Video>/results.json
        processed_dir / "results/*/results.json",  # processed/results/<Video>/results.json
    ]

    for pattern in patterns:
        for results_file in processed_dir.glob(str(pattern.relative_to(processed_dir))):
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    results = json.load(f)

                result_video = results.get("video", {}).get("filename", "")
                if Path(result_video).stem == video_name or results_file.parent.name == video_name:
                    return results.get("transcription", {}).get("segments", [])
            except Exception:
                continue

    # Try transcript files directly
    for transcript_file in (processed_dir / "transcripts").glob(f"{video_name}/transcript.json"):
        try:
            with open(transcript_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("segments", [])
        except Exception:
            continue

    return []


def _detect_language(video_name: str, processed_dir: Path) -> str:
    """Detect language from processed results."""
    patterns = [
        processed_dir / "*/*/results.json",
        processed_dir / "results/*/results.json",
    ]

    for pattern in patterns:
        for results_file in processed_dir.glob(str(pattern.relative_to(processed_dir))):
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                result_video = results.get("video", {}).get("filename", "")
                if Path(result_video).stem == video_name or results_file.parent.name == video_name:
                    return results.get("transcription", {}).get("language", "en")
            except Exception:
                continue

    return "en"


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare audio-transcript pairs for Whisper ASR fine-tuning"
    )
    parser.add_argument(
        "--gt-dir", default="ground_truth",
        help="Path to ground_truth/ directory"
    )
    parser.add_argument(
        "--processed-dir", default="processed",
        help="Path to processed/ directory (for Whisper segment timestamps)"
    )
    parser.add_argument(
        "--video-dirs", nargs="*", default=None,
        help="Directories to search for original video/audio files"
    )
    parser.add_argument(
        "--output", default="training/asr_data",
        help="Output directory for audio chunks and manifest"
    )
    parser.add_argument(
        "--mode", choices=["aligned", "full"], default="aligned",
        help="'aligned' = chunk by Whisper segments, 'full' = whole audio"
    )
    parser.add_argument(
        "--max-chunk-sec", type=float, default=30.0,
        help="Maximum chunk duration in seconds"
    )
    parser.add_argument(
        "--min-chunk-sec", type=float, default=1.0,
        help="Minimum chunk duration in seconds"
    )
    parser.add_argument(
        "--eval-ratio", type=float, default=0.15,
        help="Fraction of samples for evaluation"
    )
    parser.add_argument(
        "--min-alignment-similarity", type=float, default=0.35,
        help="Minimum alignment similarity (0..1) to keep a chunk"
    )
    parser.add_argument(
        "--min-wps", type=float, default=0.8,
        help="Minimum words-per-second threshold for chunk quality"
    )
    parser.add_argument(
        "--max-wps", type=float, default=6.0,
        help="Maximum words-per-second threshold for chunk quality"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview statistics without writing audio files"
    )
    args = parser.parse_args()

    build_asr_dataset(
        gt_dir=args.gt_dir,
        processed_dir=args.processed_dir,
        video_dirs=args.video_dirs,
        output_dir=args.output,
        mode=args.mode,
        max_chunk_sec=args.max_chunk_sec,
        min_chunk_sec=args.min_chunk_sec,
        eval_ratio=args.eval_ratio,
        min_alignment_similarity=args.min_alignment_similarity,
        min_words_per_second=args.min_wps,
        max_words_per_second=args.max_wps,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
