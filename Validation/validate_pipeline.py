"""
Pipeline Validation Script
Analyzes and validates the output of the video processing pipeline.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from datetime import timedelta


class PipelineValidator:
    """Validates pipeline outputs and calculates quality metrics."""

    def __init__(self, processed_dir: str = "processed"):
        self.processed_dir = Path(processed_dir)
        self.metrics = {}

    def validate_all(self) -> Dict:
        """Run all validation checks on processed videos."""
        # Update search pattern to match new structure: processed/results/VIDEO_NAME/results.json
        results_dir = self.processed_dir / "results"
        print(f"Searching for results in: {results_dir.absolute()}")
        results_files = list(results_dir.glob("**/results.json"))

        if not results_files:
            print(f"No processed results found in {results_dir}!")
            return {}

        print(f"\n{'=' * 60}")
        print(f"PIPELINE VALIDATION REPORT")
        print(f"{'=' * 60}")
        print(f"Found {len(results_files)} processed videos\n")

        all_metrics = []

        for results_file in results_files:
            # video_name is the parent folder name
            video_name = results_file.parent.name
            print(f"\n{'─' * 60}")
            print(f"{video_name}")
            print(f"{'─' * 60}")

            metrics = self.validate_video(results_file)
            metrics["video_name"] = video_name
            all_metrics.append(metrics)

            self.print_video_metrics(metrics)

        # Aggregate statistics
        print(f"\n{'=' * 60}")
        print(f"AGGREGATE STATISTICS")
        print(f"{'=' * 60}")
        self.print_aggregate_metrics(all_metrics)

        # Save validation report
        report_file = self.processed_dir / "validation_report.json"
        with open(report_file, "w") as f:
            json.dump(
                {
                    "individual_metrics": all_metrics,
                    "aggregate": self.calculate_aggregate(all_metrics),
                },
                f,
                indent=2,
            )

        print(f"Full validation report saved to: {report_file}")

        return all_metrics

    def validate_video(self, results_file: Path) -> Dict:
        """Validate a single video's processing results."""
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        metrics = {
            "transcription": self.validate_transcription(results),
            "scene_detection": self.validate_scenes(results),
            "alignment": self.validate_alignment(results),
            "completeness": self.validate_completeness(results),
        }

        return metrics

    def validate_transcription(self, results: Dict) -> Dict:
        """Validate transcription quality."""
        trans = results.get("transcription", {})
        segments = trans.get("segments", [])

        # Calculate metrics
        total_duration = sum(s["end"] - s["start"] for s in segments)
        avg_segment_length = total_duration / len(segments) if segments else 0

        # Confidence scores (if available in Whisper output)
        confidences = []
        for seg in segments:
            if "avg_logprob" in seg:
                # Convert log prob to approximate confidence
                confidence = np.exp(seg["avg_logprob"])
                confidences.append(confidence)

        avg_confidence = np.mean(confidences) if confidences else None
        low_confidence_count = (
            sum(1 for c in confidences if c < 0.5) if confidences else 0
        )

        # Speech rate (words per minute)
        word_count = len(trans.get("text", "").split())
        speech_rate = (word_count / total_duration * 60) if total_duration > 0 else 0

        return {
            "segment_count": len(segments),
            "total_speech_duration": round(total_duration, 2),
            "avg_segment_length": round(avg_segment_length, 2),
            "word_count": word_count,
            "speech_rate_wpm": round(speech_rate, 1),
            "avg_confidence": round(avg_confidence, 3) if avg_confidence else "N/A",
            "low_confidence_segments": low_confidence_count,
            "language": trans.get("language", "unknown"),
        }

    def validate_scenes(self, results: Dict) -> Dict:
        """Validate scene detection quality."""
        scenes = results.get("scene_analysis", {}).get("scenes", [])

        if not scenes:
            return {"error": "No scenes detected"}

        durations = [s["duration"] for s in scenes]

        # Find outliers (too short or too long)
        too_short = sum(1 for d in durations if d < 0.5)
        too_long = sum(1 for d in durations if d > 30)

        # Scene density
        total_duration = results["scene_analysis"]["total_duration"]
        scene_density = len(scenes) / (total_duration / 60) if total_duration > 0 else 0

        return {
            "scene_count": len(scenes),
            "total_duration": round(total_duration, 2),
            "avg_scene_duration": round(np.mean(durations), 2),
            "median_scene_duration": round(np.median(durations), 2),
            "std_scene_duration": round(np.std(durations), 2),
            "min_scene_duration": round(min(durations), 2),
            "max_scene_duration": round(max(durations), 2),
            "scenes_per_minute": round(scene_density, 2),
            "too_short_scenes": too_short,  # Potential false positives
            "too_long_scenes": too_long,  # Potential missed cuts
            "keyframes_extracted": sum(1 for s in scenes if s.get("keyframe_path")),
        }

    def validate_alignment(self, results: Dict) -> Dict:
        """Validate transcript-scene alignment quality."""
        scenes = results.get("scene_analysis", {}).get("scenes", [])

        scenes_with_transcript = sum(1 for s in scenes if s.get("transcript_segments"))
        empty_scenes = len(scenes) - scenes_with_transcript

        # Calculate transcript distribution
        segments_per_scene = []
        for scene in scenes:
            segments = scene.get("transcript_segments", [])
            segments_per_scene.append(len(segments))

        coverage = (scenes_with_transcript / len(scenes) * 100) if scenes else 0

        return {
            "scenes_with_transcript": scenes_with_transcript,
            "empty_scenes": empty_scenes,
            "coverage_percentage": round(coverage, 1),
            "avg_segments_per_scene": round(np.mean(segments_per_scene), 2)
            if segments_per_scene
            else 0,
            "max_segments_per_scene": max(segments_per_scene)
            if segments_per_scene
            else 0,
        }

    def validate_completeness(self, results: Dict) -> Dict:
        """Check if all expected outputs are present."""
        issues = []

        # Check for missing data
        if not results.get("transcription", {}).get("segments"):
            issues.append("Missing transcript segments")

        if not results.get("scene_analysis", {}).get("scenes"):
            issues.append("Missing scene data")

        if not results.get("alignment", {}).get("aligned_scenes"):
            issues.append("Missing alignment data")

        # Check file existence
        video_name = Path(results["video"]["path"]).stem

        expected_files = [
            self.processed_dir / f"transcripts/{video_name}/transcript.json",
            self.processed_dir
            / f"transcripts/{video_name}/{video_name}_transcript.txt",
            self.processed_dir / f"scenes/{video_name}/scenes.json",
            self.processed_dir / f"results/{video_name}/results.json",
            self.processed_dir / f"results/{video_name}/report.html",
        ]

        missing_files = [str(f.name) for f in expected_files if not f.exists()]

        return {
            "is_complete": len(issues) == 0 and len(missing_files) == 0,
            "data_issues": issues,
            "missing_files": missing_files,
        }

    def print_video_metrics(self, metrics: Dict):
        """Print metrics for a single video."""
        print("Transcription:")
        t = metrics["transcription"]
        print(f"  - Segments: {t['segment_count']}")
        print(f"  - Words: {t['word_count']}")
        print(f"  - Speech rate: {t['speech_rate_wpm']} WPM")
        print(f"  - Avg confidence: {t['avg_confidence']}")
        if t["low_confidence_segments"] > 0:
            print(f"Low confidence segments: {t['low_confidence_segments']}")

        print("Scene Detection:")
        s = metrics["scene_detection"]
        if "error" in s:
            print(f"{s['error']}")
        else:
            print(f"  - Scenes: {s['scene_count']}")
            print(f"  - Avg duration: {s['avg_scene_duration']}s")
            print(f"  - Density: {s['scenes_per_minute']} scenes/min")
            if s["too_short_scenes"] > 0:
                print(f"Too short: {s['too_short_scenes']} scenes")
            if s["too_long_scenes"] > 0:
                print(f"Too long: {s['too_long_scenes']} scenes")

        print("Alignment:")
        a = metrics["alignment"]
        print(f"  - Coverage: {a['coverage_percentage']}%")
        print(f"  - Empty scenes: {a['empty_scenes']}")

        print("Completeness:")
        c = metrics["completeness"]
        if c["is_complete"]:
            print("All outputs present")
        else:
            if c["data_issues"]:
                print(f"Issues: {', '.join(c['data_issues'])}")
            if c["missing_files"]:
                print(f"Missing: {', '.join(c['missing_files'])}")

    def print_aggregate_metrics(self, all_metrics: List[Dict]):
        """Print aggregate statistics across all videos."""

        # Transcription stats
        speech_rates = [m["transcription"]["speech_rate_wpm"] for m in all_metrics]
        print(f"Transcription (across {len(all_metrics)} videos):")
        print(
            f"  - Avg speech rate: {np.mean(speech_rates):.1f} WPM (std: {np.std(speech_rates):.1f})"
        )

        # Scene detection stats
        scene_densities = [
            m["scene_detection"]["scenes_per_minute"]
            for m in all_metrics
            if "error" not in m["scene_detection"]
        ]
        print(f"Scene Detection:")
        print(f"  - Avg scene density: {np.mean(scene_densities):.2f} scenes/min")

        # Alignment stats
        coverages = [m["alignment"]["coverage_percentage"] for m in all_metrics]
        print(f"Alignment:")
        print(f"  - Avg coverage: {np.mean(coverages):.1f}%")

        # Overall completeness
        complete = sum(1 for m in all_metrics if m["completeness"]["is_complete"])
        print(f"Overall:")
        print(
            f"  - Complete outputs: {complete}/{len(all_metrics)} ({complete / len(all_metrics) * 100:.1f}%)"
        )

    def calculate_aggregate(self, all_metrics: List[Dict]) -> Dict:
        """Calculate aggregate statistics."""
        return {
            "total_videos": len(all_metrics),
            "avg_speech_rate": round(
                np.mean([m["transcription"]["speech_rate_wpm"] for m in all_metrics]), 1
            ),
            "avg_scene_density": round(
                np.mean(
                    [
                        m["scene_detection"]["scenes_per_minute"]
                        for m in all_metrics
                        if "error" not in m["scene_detection"]
                    ]
                ),
                2,
            ),
            "avg_coverage": round(
                np.mean([m["alignment"]["coverage_percentage"] for m in all_metrics]), 1
            ),
            "completion_rate": round(
                sum(1 for m in all_metrics if m["completeness"]["is_complete"])
                / len(all_metrics)
                * 100,
                1,
            ),
        }


def create_ground_truth_template(video_name: str):
    """Create a template for manual ground truth annotation."""
    template = {
        "video": video_name,
        "ground_truth_transcript": [""],
        "instructions": "Manually transcribe the video audio exactly as spoken, including filler words.",
        "scene_annotations": [],
        "scene_annotation_instructions": "Mark scene changes with timestamps (seconds)",
    }

    output_file = Path("ground_truth") / f"{video_name}_gt.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(template, f, indent=2)

    print(f"Ground truth template created: {output_file}")
    print("  Fill in the 'ground_truth_transcript' field manually.")


if __name__ == "__main__":
    validator = PipelineValidator()

    # Run validation
    metrics = validator.validate_all()

    # Optionally create ground truth templates for manual validation
    # Uncomment the following to create templates:
    create_ground_truth_template("AkerBP 3")
