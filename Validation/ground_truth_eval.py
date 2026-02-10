"""
Ground Truth Evaluation Script
Evaluates transcriptions from all models (Whisper, Distil-Whisper, Wav2Vec2, etc.)
against manually created ground truth transcripts.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import difflib
import numpy as np


def calculate_wer(reference: str, hypothesis: str) -> Tuple[float, dict]:
    """
    Calculate Word Error Rate (WER).

    WER = (Substitutions + Deletions + Insertions) / Total Words in Reference

    Args:
        reference: Ground truth transcript
        hypothesis: Generated transcript

    Returns:
        WER score and detailed metrics
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Use difflib to find edit operations
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)

    substitutions = 0
    deletions = 0
    insertions = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            substitutions += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            deletions += i2 - i1
        elif tag == "insert":
            insertions += j2 - j1

    total_words = len(ref_words)
    wer = (
        (substitutions + deletions + insertions) / total_words if total_words > 0 else 0
    )

    return wer, {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "total_words": total_words,
        "correct_words": total_words - substitutions - deletions,
    }


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER)."""
    ref_chars = list(reference.lower().replace(" ", ""))
    hyp_chars = list(hypothesis.lower().replace(" ", ""))

    matcher = difflib.SequenceMatcher(None, ref_chars, hyp_chars)

    errors = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            errors += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            errors += i2 - i1
        elif tag == "insert":
            errors += j2 - j1

    total_chars = len(ref_chars)
    cer = errors / total_chars if total_chars > 0 else 0

    return cer


class GroundTruthEvaluator:
    """
    Evaluates transcription accuracy against ground truth for all models.
    """

    def __init__(
        self,
        transcripts_dir: str = "processed/transcripts",
        ground_truth_dir: str = "ground_truth",
        output_dir: str = "processed/ground_truth_evaluation",
    ):
        self.transcripts_dir = Path(transcripts_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def discover_models(self) -> List[str]:
        """Discover available transcription models."""
        if not self.transcripts_dir.exists():
            return []

        model_dirs = [d for d in self.transcripts_dir.iterdir() if d.is_dir()]
        model_names = []

        for d in model_dirs:
            # Check if this is a model directory (has video subdirs with transcript.json)
            has_transcripts = any(
                (subdir / "transcript.json").exists()
                for subdir in d.iterdir()
                if subdir.is_dir()
            )
            if has_transcripts:
                model_names.append(d.name)

        return sorted(model_names)

    def discover_ground_truth_files(self) -> List[str]:
        """Discover available ground truth files."""
        if not self.ground_truth_dir.exists():
            return []

        gt_files = list(self.ground_truth_dir.glob("*_gt.json"))
        return [f.stem.replace("_gt", "") for f in gt_files]

    def load_ground_truth(self, video_name: str) -> Optional[str]:
        """Load ground truth transcript for a video."""
        gt_file = self.ground_truth_dir / f"{video_name}_gt.json"

        if not gt_file.exists():
            return None

        try:
            with open(gt_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            transcript = data.get("ground_truth_transcript", "")
            if isinstance(transcript, list):
                return " ".join(transcript)
            return transcript
        except Exception as e:
            print(f"  Error loading ground truth: {e}")
            return None

    def load_model_transcript(self, model_name: str, video_name: str) -> Optional[str]:
        """Load transcript from a specific model."""
        # Try with underscores (normalized name)
        video_name_normalized = video_name.replace(" ", "_")
        transcript_file = (
            self.transcripts_dir
            / model_name
            / video_name_normalized
            / "transcript.json"
        )

        if not transcript_file.exists():
            # Try with original name
            transcript_file = (
                self.transcripts_dir / model_name / video_name / "transcript.json"
            )

        if not transcript_file.exists():
            return None

        try:
            with open(transcript_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = data.get("text", "")
            if isinstance(text, list):
                return " ".join(text)
            return text
        except Exception as e:
            print(f"  Error loading {model_name} transcript: {e}")
            return None

    def evaluate_single(
        self, model_name: str, video_name: str, ground_truth: str
    ) -> Optional[Dict]:
        """Evaluate a single model transcription against ground truth."""
        hypothesis = self.load_model_transcript(model_name, video_name)

        if not hypothesis:
            return None

        # Calculate metrics
        wer, wer_details = calculate_wer(ground_truth, hypothesis)
        cer = calculate_cer(ground_truth, hypothesis)

        return {
            "model": model_name,
            "video": video_name,
            "wer": round(wer * 100, 2),
            "cer": round(cer * 100, 2),
            "accuracy": round((1 - wer) * 100, 2),
            "wer_details": wer_details,
            "hypothesis_text": hypothesis,
        }

    def evaluate_video_all_models(self, video_name: str) -> Dict:
        """
        Evaluate all models' transcriptions for a single video.

        Returns:
            Comparison of all models for this video
        """
        print(f"\n{'=' * 60}")
        print(f"EVALUATING: {video_name}")
        print(f"{'=' * 60}")

        # Load ground truth
        ground_truth = self.load_ground_truth(video_name)
        if not ground_truth or ground_truth.strip() == "":
            print(f"No ground truth found for {video_name}")
            print(
                f"   Create one at: {self.ground_truth_dir / f'{video_name}_gt.json'}"
            )
            return {}

        print(f"Ground truth loaded ({len(ground_truth.split())} words)\n")

        # Discover available models
        models = self.discover_models()
        if not models:
            print("No transcription models found!")
            return {}

        # Evaluate each model
        results = []
        for model in models:
            print(f"Evaluating {model}... ", end="")
            metrics = self.evaluate_single(model, video_name, ground_truth)

            if metrics:
                results.append(metrics)
                print(
                    f"WER: {metrics['wer']:.1f}%, Accuracy: {metrics['accuracy']:.1f}%"
                )
            else:
                print("Transcript not found")

        if not results:
            print("\nNo transcripts found for this video!")
            return {}

        # Sort by accuracy (best first)
        results.sort(key=lambda x: x["wer"])

        # Print comparison table
        self.print_comparison_table(results, video_name)

        # Save results
        output_file = self.output_dir / f"{video_name}_evaluation.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "video": video_name,
                    "ground_truth_words": len(ground_truth.split()),
                    "models_evaluated": len(results),
                    "evaluations": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"\nResults saved to: {output_file}")

        return results

    def print_comparison_table(self, results: List[Dict], video_name: str):
        """Print comparison table of all models."""
        print(f"\n{'=' * 60}")
        print(f"COMPARISON TABLE: {video_name}")
        print(f"{'=' * 60}")
        print(f"{'Rank':<6} {'Model':<30} {'WER':<8} {'CER':<8} {'Accuracy':<10}")
        print(f"{'-' * 60}")

        for i, r in enumerate(results, 1):
            rank_symbol = "" if i == 1 else "" if i == 2 else "" if i == 3 else f"{i}."
            print(
                f"{rank_symbol:<6} {r['model']:<30} {r['wer']:.1f}%   {r['cer']:.1f}%   {r['accuracy']:.1f}%"
            )

        print(f"{'-' * 60}")

    def batch_evaluate_all_videos(self) -> List[Dict]:
        """Evaluate all videos with ground truth across all models."""
        videos = self.discover_ground_truth_files()

        if not videos:
            print("No ground truth files found!")
            print(f"   Create ground truth files in: {self.ground_truth_dir}")
            return []

        print(f"\n{'=' * 60}")
        print("BATCH GROUND TRUTH EVALUATION")
        print(f"{'=' * 60}")
        print(f"Found {len(videos)} videos with ground truth")
        print(f"Output: {self.output_dir}\n")

        all_results = []

        for i, video in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] ", end="")
            video_results = self.evaluate_video_all_models(video)
            if video_results:
                all_results.extend(video_results)

        if all_results:
            self.print_aggregate_summary(all_results)

            # Save aggregate
            summary_file = self.output_dir / "aggregate_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "total_videos": len(videos),
                        "total_evaluations": len(all_results),
                        "models": list(set(r["model"] for r in all_results)),
                        "results": all_results,
                        "aggregate_by_model": self.calculate_model_averages(
                            all_results
                        ),
                    },
                    f,
                    indent=2,
                )

            print(f"\n✓ Aggregate summary saved to: {summary_file}")

        return all_results

    def calculate_model_averages(self, all_results: List[Dict]) -> Dict:
        """Calculate average metrics for each model across all videos."""
        models = {}

        for result in all_results:
            model = result["model"]
            if model not in models:
                models[model] = {"wer": [], "cer": [], "accuracy": []}

            models[model]["wer"].append(result["wer"])
            models[model]["cer"].append(result["cer"])
            models[model]["accuracy"].append(result["accuracy"])

        # Calculate averages
        averages = {}
        for model, metrics in models.items():
            averages[model] = {
                "avg_wer": round(np.mean(metrics["wer"]), 2),
                "avg_cer": round(np.mean(metrics["cer"]), 2),
                "avg_accuracy": round(np.mean(metrics["accuracy"]), 2),
                "std_wer": round(np.std(metrics["wer"]), 2),
                "evaluations": len(metrics["wer"]),
            }

        return averages

    def print_aggregate_summary(self, all_results: List[Dict]):
        """Print aggregate statistics across all evaluations."""
        model_averages = self.calculate_model_averages(all_results)

        print(f"\n{'=' * 60}")
        print("AGGREGATE SUMMARY (All Videos)")
        print(f"{'=' * 60}")

        # Sort models by average accuracy
        sorted_models = sorted(model_averages.items(), key=lambda x: x[1]["avg_wer"])

        print(f"{'Model':<30} {'Avg WER':<10} {'Avg Acc':<10} {'Videos':<8}")
        print(f"{'-' * 60}")

        for model, metrics in sorted_models:
            print(
                f"{model:<30} {metrics['avg_wer']:.1f}%    {metrics['avg_accuracy']:.1f}%    {metrics['evaluations']}"
            )

        print(f"{'-' * 60}")

    def create_ground_truth_template(self, video_name: str):
        """Create a template for manual ground truth annotation."""
        template = {
            "video": video_name,
            "ground_truth_transcript": "",
            "instructions": "Manually transcribe the video audio exactly as spoken. This is the reference transcript.",
            "notes": "",
        }

        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.ground_truth_dir / f"{video_name}_gt.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2)

        print(f"✓ Ground truth template created: {output_file}")
        print(
            "  Fill in the 'ground_truth_transcript' field with the correct transcription."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ground Truth Evaluation")
    parser.add_argument(
        "video",
        nargs="?",
        help="Specific video name to evaluate (optional). If omitted, evaluates all.",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create a ground truth template for the specified video",
    )

    args = parser.parse_args()
    evaluator = GroundTruthEvaluator()

    if args.create:
        if not args.video:
            print("Error: Please specify a video name to create a template for.")
            print('Usage: python ground_truth_eval.py "Video Name" --create')
        else:
            evaluator.create_ground_truth_template(args.video)

    elif args.video:
        # Single video mode
        evaluator.evaluate_video_all_models(args.video)

    else:
        # Batch mode (default)
        print(
            "No video specified. Running batch evaluation for all available ground truth files..."
        )
        evaluator.batch_evaluate_all_videos()
