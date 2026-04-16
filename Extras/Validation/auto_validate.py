"""
Cross-Validation of Transcriber Results
Compares transcription outputs from different models (Whisper Large vs Distil-Whisper).
"""

import json
from pathlib import Path
from typing import Dict, List
import difflib
import numpy as np


class TranscriberCrossValidator:
    """
    Cross-validates transcription results from different models in transcriber.py.
    Compares Whisper Large vs Distil-Whisper outputs.
    """

    def __init__(
        self,
        primary_model: str = "Whisper-Large",
        validator_model: str = "Distil-Whisper-Large-v3",
        transcripts_dir: str = "processed/transcripts",
    ):
        """
        Initialize cross-validator for existing transcription results.

        Args:
            primary_model: Primary model name (e.g., "Whisper-Large")
            validator_model: Validation model name (e.g., "Distil-Whisper-Large-v3")
            transcripts_dir: Base directory containing transcripts
        """
        self.primary_name = primary_model
        self.validator_name = validator_model
        self.transcripts_dir = Path(transcripts_dir)

        print("Cross-Validator initialized:")
        print(f"  Primary model: {self.primary_name}")
        print(f"  Validator model: {self.validator_name}")
        print(f"  Transcripts directory: {self.transcripts_dir}")

    def validate_video(
        self, video_name: str, output_dir: str = "processed/validation"
    ) -> Dict:
        """
        Compare existing transcription results for a video.

        Args:
            video_name: Name of the video (with or without extension)
            output_dir: Where to save validation results

        Returns:
            Validation metrics including agreement scores and flagged segments
        """
        # Clean video name (remove extension if present)
        video_stem = Path(video_name).stem.replace(" ", "_")

        # Load primary model results
        primary_path = (
            self.transcripts_dir / self.primary_name / video_stem / "transcript.json"
        )
        if not primary_path.exists():
            print(f"❌ Primary transcript not found: {primary_path}")
            print(f"   Run transcriber.py with {self.primary_name} first")
            return {}

        # Load validator model results
        validator_path = (
            self.transcripts_dir / self.validator_name / video_stem / "transcript.json"
        )
        if not validator_path.exists():
            print(f"❌ Validator transcript not found: {validator_path}")
            print(f"   Run transcriber.py with {self.validator_name} first")
            return {}

        print(f"\n{'=' * 60}")
        print(f"CROSS-VALIDATING: {video_name}")
        print(f"{'=' * 60}\n")

        # Load transcripts
        with open(primary_path, "r", encoding="utf-8") as f:
            primary_result = json.load(f)

        with open(validator_path, "r", encoding="utf-8") as f:
            validator_result = json.load(f)

        print(
            f"✓ Loaded {self.primary_name} transcript ({len(primary_result.get('segments', []))} segments)"
        )
        print(
            f"✓ Loaded {self.validator_name} transcript ({len(validator_result.get('segments', []))} segments)"
        )

        # Compare results
        print("\nAnalyzing agreement...")
        metrics = self.compare_transcriptions(primary_result, validator_result)
        metrics["video"] = video_name
        metrics["primary_model"] = self.primary_name
        metrics["validator_model"] = self.validator_name

        # Save detailed results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / f"{video_stem}_cross_validation.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "video": video_name,
                    "metrics": metrics,
                    "primary_model": self.primary_name,
                    "validator_model": self.validator_name,
                    "primary_transcript": primary_result.get("text", ""),
                    "validator_transcript": validator_result.get("text", ""),
                    "flagged_segments": metrics.get("flagged_segments", []),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Print summary
        self.print_validation_summary(metrics)

        print(f"\n✓ Validation report saved to: {results_file}")

        return metrics

    def compare_transcriptions(self, primary: Dict, validator: Dict) -> Dict:
        """
        Compare two transcription results and calculate agreement metrics.
        """
        # Overall text comparison
        primary_text = primary.get("text", "").lower().strip()
        validator_text = validator.get("text", "").lower().strip()

        # Calculate word-level agreement
        primary_words = primary_text.split()
        validator_words = validator_text.split()

        # Use SequenceMatcher to find agreement
        matcher = difflib.SequenceMatcher(None, primary_words, validator_words)
        matching_blocks = matcher.get_matching_blocks()

        # Count matching words
        total_matches = sum(block.size for block in matching_blocks)
        max_words = max(len(primary_words), len(validator_words))
        agreement_rate = (total_matches / max_words * 100) if max_words > 0 else 0

        # Calculate WER between the two models
        wer = self.calculate_wer(primary_words, validator_words)

        # Segment-level analysis
        flagged_segments = self.analyze_segments(
            primary.get("segments", []), validator.get("segments", [])
        )

        # Confidence analysis (if available)
        primary_confidences = self.extract_confidences(primary.get("segments", []))
        validator_confidences = self.extract_confidences(validator.get("segments", []))

        return {
            "overall_agreement": round(agreement_rate, 2),
            "word_error_rate": round(wer, 2),
            "total_words_primary": len(primary_words),
            "total_words_validator": len(validator_words),
            "matching_words": total_matches,
            "total_segments_primary": len(primary.get("segments", [])),
            "total_segments_validator": len(validator.get("segments", [])),
            "avg_confidence_primary": round(np.mean(primary_confidences), 3)
            if primary_confidences
            else None,
            "avg_confidence_validator": round(np.mean(validator_confidences), 3)
            if validator_confidences
            else None,
            "flagged_segments": flagged_segments,
            "validation_status": self.determine_status(agreement_rate, wer),
        }

    def calculate_wer(self, reference: List[str], hypothesis: List[str]) -> float:
        """Calculate Word Error Rate between two transcriptions."""
        matcher = difflib.SequenceMatcher(None, reference, hypothesis)

        errors = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                errors += max(i2 - i1, j2 - j1)
            elif tag == "delete":
                errors += i2 - i1
            elif tag == "insert":
                errors += j2 - j1

        total_words = len(reference)
        wer = (errors / total_words * 100) if total_words > 0 else 0

        return wer

    def analyze_segments(
        self, primary_segments: List, validator_segments: List
    ) -> List[Dict]:
        """
        Analyze segment-level disagreements between models.
        Flag segments where models strongly disagree.
        """
        flagged = []

        # Match segments by timestamp overlap
        for i, p_seg in enumerate(primary_segments):
            p_start, p_end = p_seg["start"], p_seg["end"]
            p_text = p_seg["text"].strip().lower()

            # Skip empty segments
            if not p_text:
                continue

            # Find overlapping validator segment
            best_match = None
            best_overlap = 0

            for v_seg in validator_segments:
                v_start, v_end = v_seg["start"], v_seg["end"]

                # Calculate overlap
                overlap_start = max(p_start, v_start)
                overlap_end = min(p_end, v_end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = v_seg

            if best_match:
                v_text = best_match["text"].strip().lower()

                # Calculate similarity
                similarity = difflib.SequenceMatcher(None, p_text, v_text).ratio()

                # Flag if disagreement is significant
                if similarity < 0.7:  # Less than 70% similar
                    flagged.append(
                        {
                            "segment_index": i,
                            "timestamp": f"{p_start:.1f}s - {p_end:.1f}s",
                            "primary_text": p_seg["text"].strip(),
                            "validator_text": best_match["text"].strip(),
                            "similarity": round(similarity * 100, 1),
                            "reason": "Low model agreement",
                        }
                    )

        return flagged

    def extract_confidences(self, segments: List[Dict]) -> List[float]:
        """Extract confidence scores from segments (if available)."""
        confidences = []
        for seg in segments:
            if "avg_logprob" in seg:
                # Convert log probability to approximate confidence
                confidence = np.exp(seg["avg_logprob"])
                confidences.append(confidence)
        return confidences

    def determine_status(self, agreement: float, wer: float) -> str:
        """Determine overall validation status."""
        if agreement >= 90 and wer < 10:
            return "EXCELLENT - High confidence"
        elif agreement >= 80 and wer < 20:
            return "GOOD - Reliable transcription"
        elif agreement >= 70 and wer < 30:
            return "ACCEPTABLE - Minor discrepancies"
        else:
            return "NEEDS_REVIEW - Significant disagreement"

    def print_validation_summary(self, metrics: Dict):
        """Print validation summary."""
        print(f"\n{'=' * 60}")
        print("VALIDATION RESULTS")
        print(f"{'=' * 60}")
        print(f"\nOverall Agreement: {metrics['overall_agreement']:.1f}%")
        print(f"Word Error Rate: {metrics['word_error_rate']:.1f}%")
        print(f"Status: {metrics['validation_status']}")

        print(f"\nModel Comparison:")
        print(
            f"  Primary ({metrics.get('primary_model', 'N/A')}): "
            f"{metrics['total_words_primary']} words, "
            f"{metrics['total_segments_primary']} segments"
        )
        print(
            f"  Validator ({metrics.get('validator_model', 'N/A')}): "
            f"{metrics['total_words_validator']} words, "
            f"{metrics['total_segments_validator']} segments"
        )
        print(f"  Matching words: {metrics['matching_words']}")

        if metrics.get("avg_confidence_primary"):
            print(f"\nConfidence Scores:")
            print(f"  Primary: {metrics['avg_confidence_primary']:.3f}")
            if metrics.get("avg_confidence_validator"):
                print(f"  Validator: {metrics['avg_confidence_validator']:.3f}")

        flagged = metrics.get("flagged_segments", [])
        if flagged:
            print(f"\nFlagged Segments: {len(flagged)}")
            print("(Segments where models significantly disagree)")
            for seg in flagged[:3]:  # Show first 3
                print(f"\n  [{seg['timestamp']}]")
                print(f"    Primary: '{seg['primary_text']}'")
                print(f"    Validator: '{seg['validator_text']}'")
                print(f"    Similarity: {seg['similarity']}%")

            if len(flagged) > 3:
                print(f"\n  ... and {len(flagged) - 3} more (see JSON report)")
        else:
            print("\n✓ No significant disagreements found!")

    def batch_validate(self) -> List[Dict]:
        """
        Validate all videos that have been transcribed by both models.
        """
        # Find all videos transcribed by primary model
        primary_dir = self.transcripts_dir / self.primary_name
        if not primary_dir.exists():
            print(f"❌ Primary model directory not found: {primary_dir}")
            return []

        video_dirs = [d for d in primary_dir.iterdir() if d.is_dir()]

        if not video_dirs:
            print(f"No transcripts found for {self.primary_name}")
            return []

        print(f"\n{'=' * 60}")
        print(f"BATCH CROSS-VALIDATION")
        print(f"{'=' * 60}")
        print(f"Found {len(video_dirs)} videos from {self.primary_name}\n")

        all_metrics = []

        for i, video_dir in enumerate(video_dirs, 1):
            video_name = video_dir.name.replace("_", " ")
            print(f"\n[{i}/{len(video_dirs)}] Processing: {video_name}")

            try:
                metrics = self.validate_video(video_name)
                if metrics:
                    all_metrics.append(metrics)
            except Exception as e:
                print(f"❌ Error validating {video_name}: {e}")
                import traceback

                traceback.print_exc()

        # Aggregate summary
        if all_metrics:
            self.print_batch_summary(all_metrics)

        return all_metrics

    def print_batch_summary(self, all_metrics: List[Dict]):
        """Print aggregate validation statistics."""
        print(f"\n{'=' * 60}")
        print("BATCH VALIDATION SUMMARY")
        print(f"{'=' * 60}")

        agreements = [m["overall_agreement"] for m in all_metrics]
        wers = [m["word_error_rate"] for m in all_metrics]

        print(
            f"\nAverage Agreement: {np.mean(agreements):.1f}% (±{np.std(agreements):.1f}%)"
        )
        print(f"Average WER: {np.mean(wers):.1f}% (±{np.std(wers):.1f}%)")

        # Count by status
        statuses = {}
        for m in all_metrics:
            status = m["validation_status"].split(" - ")[0]
            statuses[status] = statuses.get(status, 0) + 1

        print(f"\nValidation Status Distribution:")
        for status, count in sorted(statuses.items()):
            print(f"  {status}: {count} videos ({count / len(all_metrics) * 100:.1f}%)")

        # Videos needing review
        needs_review = [
            m for m in all_metrics if "NEEDS_REVIEW" in m["validation_status"]
        ]
        if needs_review:
            print(f"\nVideos needing manual review:")
            for m in needs_review:
                print(f"  - {m['video']} (Agreement: {m['overall_agreement']:.1f}%)")

    def list_available_videos(self):
        """List which videos have been transcribed by which models."""
        print(f"\n{'=' * 60}")
        print("AVAILABLE TRANSCRIPTIONS")
        print(f"{'=' * 60}\n")

        primary_dir = self.transcripts_dir / self.primary_name
        validator_dir = self.transcripts_dir / self.validator_name

        primary_videos = set()
        validator_videos = set()

        if primary_dir.exists():
            primary_videos = {d.name for d in primary_dir.iterdir() if d.is_dir()}

        if validator_dir.exists():
            validator_videos = {d.name for d in validator_dir.iterdir() if d.is_dir()}

        both = primary_videos & validator_videos
        only_primary = primary_videos - validator_videos
        only_validator = validator_videos - primary_videos

        print(f"Videos with BOTH models ({len(both)}):")
        for v in sorted(both):
            print(f"  ✓ {v.replace('_', ' ')}")

        if only_primary:
            print(f"\nVideos with ONLY {self.primary_name} ({len(only_primary)}):")
            for v in sorted(only_primary):
                print(f"  - {v.replace('_', ' ')}")

        if only_validator:
            print(f"\nVideos with ONLY {self.validator_name} ({len(only_validator)}):")
            for v in sorted(only_validator):
                print(f"  - {v.replace('_', ' ')}")

        if not both:
            print("\n⚠ No videos have been transcribed by both models yet!")
            print(
                f"   Run transcriber.py with both {self.primary_name} and {self.validator_name}"
            )


def discover_available_models(transcripts_dir: str = "processed/transcripts") -> List[str]:
    """
    Discover all available model directories in the transcripts folder.
    A valid model directory contains video subdirectories with transcript.json files.
    
    Returns:
        List of model names that have transcripts
    """
    transcripts_path = Path(transcripts_dir)
    
    if not transcripts_path.exists():
        print(f"❌ Transcripts directory not found: {transcripts_path}")
        print("   Please run transcribe_all.py first to create transcripts.")
        return []
    
    model_dirs = []
    
    # Check each directory to see if it's a model folder or a video folder
    for potential_model in transcripts_path.iterdir():
        if not potential_model.is_dir():
            continue
        
        # A model directory contains subdirectories (videos) with transcript.json files
        # A video directory contains transcript.json directly
        has_video_subdirs = False
        has_direct_transcript = (potential_model / "transcript.json").exists()
        
        # Check if this directory contains video subdirectories
        for item in potential_model.iterdir():
            if item.is_dir():
                # Check if this subdirectory has a transcript.json
                if (item / "transcript.json").exists():
                    has_video_subdirs = True
                    break
        
        # It's a model directory if it has video subdirectories, not direct transcripts
        if has_video_subdirs and not has_direct_transcript:
            model_dirs.append(potential_model.name)
    
    return sorted(model_dirs)


def display_model_selection_menu(models: List[str], prompt_text: str) -> str:
    """
    Display interactive model selection menu.
    
    Args:
        models: List of available model names
        prompt_text: Text to display as prompt (e.g., "Choose Model 1")
    
    Returns:
        Selected model name
    """
    print(f"\n{'=' * 60}")
    print(prompt_text.upper())
    print(f"{'=' * 60}\n")
    
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    print(f"\n{'=' * 60}")
    
    while True:
        try:
            choice = input(f"\nEnter your choice (1-{len(models)}): ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(models):
                return models[choice_idx]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(models)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            exit(0)


def list_videos_for_models(model1: str, model2: str, transcripts_dir: str = "processed/transcripts"):
    """
    List videos that have transcripts from both models and allow selection.
    
    Args:
        model1: First model name
        model2: Second model name
        transcripts_dir: Base transcripts directory
    
    Returns:
        Tuple of (videos_with_both, videos_only_model1, videos_only_model2)
    """
    transcripts_path = Path(transcripts_dir)
    
    model1_dir = transcripts_path / model1
    model2_dir = transcripts_path / model2
    
    model1_videos = set()
    model2_videos = set()
    
    if model1_dir.exists():
        model1_videos = {d.name for d in model1_dir.iterdir() if d.is_dir()}
    
    if model2_dir.exists():
        model2_videos = {d.name for d in model2_dir.iterdir() if d.is_dir()}
    
    both = model1_videos & model2_videos
    only_model1 = model1_videos - model2_videos
    only_model2 = model2_videos - model1_videos
    
    print(f"\n{'=' * 60}")
    print("AVAILABLE VIDEOS FOR COMPARISON")
    print(f"{'=' * 60}")
    print(f"Model 1: {model1}")
    print(f"Model 2: {model2}")
    print(f"{'=' * 60}\n")
    
    if both:
        print(f"✓ Videos with BOTH models ({len(both)}):")
        for i, v in enumerate(sorted(both), 1):
            print(f"  {i}. {v.replace('_', ' ')}")
    else:
        print("❌ No videos have transcripts from both models!")
    
    if only_model1:
        print(f"\n⚠ Videos with ONLY {model1} ({len(only_model1)}):")
        for v in sorted(only_model1):
            print(f"  - {v.replace('_', ' ')}")
    
    if only_model2:
        print(f"\n⚠ Videos with ONLY {model2} ({len(only_model2)}):")
        for v in sorted(only_model2):
            print(f"  - {v.replace('_', ' ')}")
    
    return both, only_model1, only_model2


def select_video_to_compare(videos_with_both: set) -> str:
    """
    Allow user to select a video from the list of videos with both transcripts.
    
    Args:
        videos_with_both: Set of video names that have both transcripts
    
    Returns:
        Selected video name
    """
    if not videos_with_both:
        return None
    
    videos_list = sorted(list(videos_with_both))
    
    print(f"\n{'=' * 60}")
    print("SELECT VIDEO TO COMPARE")
    print(f"{'=' * 60}\n")
    
    for i, video in enumerate(videos_list, 1):
        print(f"{i}. {video.replace('_', ' ')}")
    
    print(f"\n0. Compare ALL videos")
    print(f"{'=' * 60}")
    
    while True:
        try:
            choice = input(f"\nEnter your choice (0-{len(videos_list)}): ").strip()
            choice_idx = int(choice)
            
            if choice_idx == 0:
                return "ALL"
            elif 1 <= choice_idx <= len(videos_list):
                return videos_list[choice_idx - 1]
            else:
                print(f"Invalid choice. Please enter a number between 0 and {len(videos_list)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            exit(0)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TRANSCRIPT COMPARISON TOOL")
    print("=" * 60)
    print("Compare transcription results from different ASR models")
    print("=" * 60)
    
    # Step 1: Discover available models
    available_models = discover_available_models()
    
    if len(available_models) < 2:
        print(f"\n❌ Error: Found only {len(available_models)} model(s) with transcripts.")
        print("   You need at least 2 models to compare.")
        print("\n💡 Tip: Run transcribe_all.py with different models to create transcripts.")
        exit(1)
    
    print(f"\n✓ Found {len(available_models)} models with transcripts")
    
    # Step 2: Select Model 1
    model1 = display_model_selection_menu(available_models, "Select Model 1 (Primary)")
    print(f"\n✓ Selected Model 1: {model1}")
    
    # Step 3: Select Model 2 (exclude Model 1 from options)
    remaining_models = [m for m in available_models if m != model1]
    model2 = display_model_selection_menu(remaining_models, "Select Model 2 (Comparison)")
    print(f"\n✓ Selected Model 2: {model2}")
    
    # Step 4: List available videos for both models
    videos_with_both, only_model1, only_model2 = list_videos_for_models(model1, model2)
    
    # Step 5: Check if there are videos to compare
    if not videos_with_both:
        print(f"\n{'=' * 60}")
        print("❌ NO VIDEOS TO COMPARE")
        print(f"{'=' * 60}")
        print(f"\nNo videos have been transcribed by BOTH models.")
        print(f"\nTo fix this:")
        print(f"  1. Run transcribe_all.py")
        print(f"  2. Select model: {model1}")
        print(f"  3. Transcribe the videos")
        print(f"  4. Run transcribe_all.py again")
        print(f"  5. Select model: {model2}")
        print(f"  6. Transcribe the same videos")
        print(f"\nThen run this comparison tool again.")
        exit(1)
    
    # Step 6: Select which video(s) to compare
    selected_video = select_video_to_compare(videos_with_both)
    
    if selected_video is None:
        print("\n❌ No video selected.")
        exit(1)
    
    # Step 7: Initialize validator and compare
    validator = TranscriberCrossValidator(
        primary_model=model1,
        validator_model=model2,
    )
    
    if selected_video == "ALL":
        # Compare all videos
        print(f"\n✓ Comparing ALL {len(videos_with_both)} videos...\n")
        validator.batch_validate()
    else:
        # Compare single video
        video_name = selected_video.replace('_', ' ') + ".mp4"
        print(f"\n✓ Comparing: {video_name}\n")
        validator.validate_video(video_name)
