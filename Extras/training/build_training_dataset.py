"""
build_training_dataset.py

Build a contrastive retrieval training dataset from processed pipeline outputs.
Generates query→segment pairs with 3-tier negatives (easy/medium/hard).

Usage:
    python -m training.build_training_dataset                        # full build
    python -m training.build_training_dataset --dry-run              # schema check only
    python -m training.build_training_dataset --auto-queries         # LLM-generate queries
    python -m training.build_training_dataset --from-judgments       # from relevance_judgments.csv
"""

import json
import random
import argparse
import csv
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────


@dataclass
class SegmentInfo:
    """Flattened transcript segment with scene metadata."""

    video: str
    video_id: Optional[int]
    segment_index: int
    start_time: float
    end_time: float
    text: str
    language: str
    scene_id: Optional[int]
    caption: Optional[str] = None
    ocr_text: Optional[str] = None
    object_labels: List[str] = field(default_factory=list)

    @property
    def enriched_text_ocr(self) -> str:
        """Transcript + OCR (Version B)."""
        parts = [self.text]
        if self.ocr_text:
            parts.append(self.ocr_text)
        return " ".join(parts)

    @property
    def enriched_text_full(self) -> str:
        """Transcript + caption + OCR + labels (Version C)."""
        parts = [self.text]
        if self.caption:
            parts.append(self.caption)
        if self.ocr_text:
            parts.append(self.ocr_text)
        if self.object_labels:
            parts.append(" ".join(str(lbl) for lbl in self.object_labels))
        return " ".join(parts)


@dataclass
class TrainingEntry:
    """One query→segment training example."""

    query_id: str
    query: str
    query_type: str
    language: str
    video: str
    positive: Dict
    hard_negatives: List[Dict]
    metadata: Dict


# ──────────────────────────────────────────────
# Load processed data
# ──────────────────────────────────────────────


def load_segments_from_processed(processed_dir: str = "processed") -> List[SegmentInfo]:
    """
    Load all transcript segments from processed pipeline results.
    Walks processed/<Model>/<Video>/results.json files.
    """
    processed_dir = Path(processed_dir)
    results_files = list(processed_dir.glob("*/*/results.json"))

    if not results_files:
        # Try legacy layout: processed/results/<Video>/results.json
        results_files = list(processed_dir.glob("results/*/results.json"))

    if not results_files:
        print(f"No results.json files found under {processed_dir}")
        return []

    all_segments: List[SegmentInfo] = []
    print(f"Loading segments from {len(results_files)} result files...")

    for rf in results_files:
        try:
            with open(rf, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception as e:
            print(f"  Skipping {rf}: {e}")
            continue

        video_name = results.get("video", {}).get("filename", rf.parent.name)
        video_stem = Path(video_name).stem
        language = results.get("transcription", {}).get("language", "en")

        # Build scene lookup: scene_id → scene metadata
        scenes = results.get("scene_analysis", {}).get("scenes", [])
        scene_lookup: Dict[int, Dict] = {}
        for scene in scenes:
            sid = scene.get("scene_id")
            if sid is not None:
                scene_lookup[sid] = scene

        # Build segment→scene mapping
        segments = results.get("transcription", {}).get("segments", [])

        for idx, seg in enumerate(segments):
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            seg_text = seg.get("text", "").strip()

            if not seg_text:
                continue

            # Find which scene this segment belongs to
            matched_scene_id = None
            matched_scene = None
            for sid, scene in scene_lookup.items():
                if seg_start >= scene["start_time"] and seg_start <= scene["end_time"]:
                    matched_scene_id = sid
                    matched_scene = scene
                    break

            segment = SegmentInfo(
                video=video_stem,
                video_id=None,  # filled if using DB
                segment_index=idx,
                start_time=seg_start,
                end_time=seg_end,
                text=seg_text,
                language=language,
                scene_id=matched_scene_id,
                caption=matched_scene.get("caption") if matched_scene else None,
                ocr_text=matched_scene.get("ocr_text") if matched_scene else None,
                object_labels=matched_scene.get("object_labels", [])
                if matched_scene
                else [],
            )
            all_segments.append(segment)

    print(f"Loaded {len(all_segments)} segments from {len(results_files)} videos")
    return all_segments


def load_segments_from_db() -> List[SegmentInfo]:
    """
    Load segments directly from the database (alternative to file-based loading).
    """
    try:
        from database.config import SessionLocal
        from database.models import Video, Scene, TranscriptSegment
    except ImportError:
        print("Database modules not available. Use file-based loading instead.")
        return []

    session = SessionLocal()
    all_segments: List[SegmentInfo] = []

    try:
        segments = (
            session.query(TranscriptSegment)
            .join(Video)
            .outerjoin(Scene, TranscriptSegment.scene_id == Scene.id)
            .all()
        )

        for seg in segments:
            scene = seg.scene if seg.scene_id else None
            segment = SegmentInfo(
                video=Path(seg.video.filename).stem,
                video_id=seg.video_id,
                segment_index=seg.segment_index,
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=seg.text,
                language=seg.language or "en",
                scene_id=scene.scene_id if scene else None,
                caption=scene.caption if scene else None,
                ocr_text=scene.ocr_text if scene else None,
                object_labels=scene.object_labels
                if scene and scene.object_labels
                else [],
            )
            all_segments.append(segment)

        print(f"Loaded {len(all_segments)} segments from database")
    finally:
        session.close()

    return all_segments


# ──────────────────────────────────────────────
# Negative sampling (3-tier strategy)
# ──────────────────────────────────────────────


def _words(text: str) -> set:
    """Extract lowercase word set from text."""
    return set(re.findall(r"\w+", text.lower()))


def sample_negatives(
    positive: SegmentInfo,
    query_text: str,
    all_segments: List[SegmentInfo],
    easy_count: int = 2,
    medium_count: int = 2,
    hard_temporal_count: int = 1,
    hard_keyword_count: int = 1,
) -> List[Dict]:
    """
    Sample negatives using the 3-tier strategy.

    Tier 1 (Easy):   Random segments from different videos
    Tier 2 (Medium): Same video, different scene (scene_id gap > 2)
    Tier 3 (Hard):   Temporal neighbors or keyword overlaps within same scene
    """
    negatives: List[Dict] = []
    used_indices: set = set()  # (video, segment_index) pairs already used

    # Index segments by video
    by_video: Dict[str, List[SegmentInfo]] = {}
    for seg in all_segments:
        by_video.setdefault(seg.video, []).append(seg)

    # ── Tier 1: Easy negatives ──
    other_video_segs = [s for s in all_segments if s.video != positive.video]
    if other_video_segs:
        easy_samples = random.sample(
            other_video_segs, min(easy_count, len(other_video_segs))
        )
        for s in easy_samples:
            negatives.append(
                {
                    "segment_text": s.text,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "video": s.video,
                    "reason": "easy_different_video",
                }
            )
            used_indices.add((s.video, s.segment_index))

    # ── Tier 2: Medium negatives ──
    same_video_segs = [
        s
        for s in by_video.get(positive.video, [])
        if s.segment_index != positive.segment_index
        and (
            positive.scene_id is None
            or s.scene_id is None
            or abs(s.scene_id - positive.scene_id) > 2
        )
        and (s.video, s.segment_index) not in used_indices
    ]
    if same_video_segs:
        medium_samples = random.sample(
            same_video_segs, min(medium_count, len(same_video_segs))
        )
        for s in medium_samples:
            negatives.append(
                {
                    "segment_text": s.text,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "video": s.video,
                    "reason": "medium_different_scene",
                }
            )
            used_indices.add((s.video, s.segment_index))

    # ── Tier 3a: Hard — temporal neighbor ──
    same_video_all = by_video.get(positive.video, [])
    same_video_by_idx = {s.segment_index: s for s in same_video_all}

    temporal_added = 0
    for offset in [1, -1]:
        if temporal_added >= hard_temporal_count:
            break
        neighbor_idx = positive.segment_index + offset
        neighbor = same_video_by_idx.get(neighbor_idx)
        if neighbor and (neighbor.video, neighbor.segment_index) not in used_indices:
            negatives.append(
                {
                    "segment_text": neighbor.text,
                    "start_time": neighbor.start_time,
                    "end_time": neighbor.end_time,
                    "video": neighbor.video,
                    "reason": "hard_temporal_neighbor",
                }
            )
            used_indices.add((neighbor.video, neighbor.segment_index))
            temporal_added += 1

    # ── Tier 3b: Hard — keyword overlap ──
    if hard_keyword_count > 0:
        query_words = _words(query_text)
        keyword_candidates = [
            s
            for s in same_video_all
            if s.segment_index != positive.segment_index
            and (s.video, s.segment_index) not in used_indices
            and len(query_words & _words(s.text)) >= 1
            and (positive.scene_id is None or s.scene_id != positive.scene_id)
        ]
        if keyword_candidates:
            kw_samples = random.sample(
                keyword_candidates, min(hard_keyword_count, len(keyword_candidates))
            )
            for s in kw_samples:
                overlap = query_words & _words(s.text)
                negatives.append(
                    {
                        "segment_text": s.text,
                        "start_time": s.start_time,
                        "end_time": s.end_time,
                        "video": s.video,
                        "reason": f"hard_keyword_overlap:{','.join(sorted(overlap)[:3])}",
                    }
                )

    return negatives


# ──────────────────────────────────────────────
# Query generation
# ──────────────────────────────────────────────


def generate_queries_auto(
    segments: List[SegmentInfo],
    queries_per_video: int = 5,
) -> List[TrainingEntry]:
    """
    Auto-generate candidate training queries from transcript segments.
    Uses heuristics to create different query types.
    These should be MANUALLY REVIEWED before training.
    """
    entries: List[TrainingEntry] = []
    by_video: Dict[str, List[SegmentInfo]] = {}
    for seg in segments:
        by_video.setdefault(seg.video, []).append(seg)

    query_counter = 0

    for video, video_segs in by_video.items():
        if len(video_segs) < 3:
            continue

        # Sort by time
        video_segs.sort(key=lambda s: s.start_time)

        # Strategy 1: Pick content-rich segments (longer text = likely more substance)
        content_segs = sorted(video_segs, key=lambda s: len(s.text), reverse=True)
        selected = content_segs[:queries_per_video]

        for seg in selected:
            query_counter += 1

            # Generate a natural-language search query from the segment text
            # Simple heuristic: extract key phrases and form a question
            words = seg.text.split()

            # Type A: "where do they talk about X?" using first ~5 content words
            content_words = [w for w in words if len(w) > 3][:5]
            if content_words:
                query_a = f"where do they talk about {' '.join(content_words)}?"
                entry = TrainingEntry(
                    query_id=f"auto_{query_counter:04d}",
                    query=query_a,
                    query_type="natural_search",
                    language=seg.language,
                    video=video,
                    positive={
                        "segment_text": seg.text,
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "scene_id": seg.scene_id,
                        "segment_index": seg.segment_index,
                    },
                    hard_negatives=[],  # filled in later
                    metadata={
                        "caption": seg.caption,
                        "ocr_text": seg.ocr_text,
                        "object_labels": seg.object_labels,
                        "auto_generated": True,
                        "needs_review": True,
                    },
                )
                entries.append(entry)

            # Type B: Exact keyword using a distinctive phrase from the segment
            if len(words) >= 4:
                query_counter += 1
                # Pick a 3-4 word phrase from the middle of the segment
                mid = len(words) // 2
                phrase = " ".join(words[mid : mid + 4])
                entry_kw = TrainingEntry(
                    query_id=f"auto_{query_counter:04d}",
                    query=phrase,
                    query_type="exact_keyword",
                    language=seg.language,
                    video=video,
                    positive={
                        "segment_text": seg.text,
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "scene_id": seg.scene_id,
                        "segment_index": seg.segment_index,
                    },
                    hard_negatives=[],
                    metadata={
                        "caption": seg.caption,
                        "ocr_text": seg.ocr_text,
                        "object_labels": seg.object_labels,
                        "auto_generated": True,
                        "needs_review": True,
                    },
                )
                entries.append(entry_kw)

    print(
        f"Auto-generated {len(entries)} candidate queries from {len(by_video)} videos"
    )
    return entries


def load_queries_from_judgments(
    judgments_path: str,
    segments: List[SegmentInfo],
) -> List[TrainingEntry]:
    """
    Load training queries from a manually curated relevance_judgments.csv file.
    """
    judgments_path = Path(judgments_path)
    if not judgments_path.exists():
        print(f"Judgments file not found: {judgments_path}")
        return []

    # Index segments for fast lookup
    seg_index: Dict[Tuple[str, int], SegmentInfo] = {}
    for seg in segments:
        seg_index[(seg.video, seg.segment_index)] = seg

    # Group rows by query_id
    from collections import defaultdict

    query_groups = defaultdict(list)

    with open(judgments_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_groups[row["query_id"]].append(row)

    entries: List[TrainingEntry] = []

    for query_id, rows in query_groups.items():
        # Find the positive (highest relevance)
        rows.sort(key=lambda r: int(r.get("relevance", 0)), reverse=True)
        pos_row = rows[0]

        if int(pos_row.get("relevance", 0)) < 2:
            continue  # no good positive for this query

        # Look up the segment
        video = pos_row["video"]
        seg_idx = int(pos_row.get("segment_index", -1))
        seg = seg_index.get((video, seg_idx))

        positive = {
            "segment_text": seg.text if seg else "",
            "start_time": float(pos_row["start_time"]),
            "end_time": float(pos_row["end_time"]),
            "scene_id": seg.scene_id if seg else None,
            "segment_index": seg_idx,
        }

        # Hard negatives from rows with relevance 0 or 1 for the same query
        hard_negatives = []
        for row in rows[1:]:
            if int(row.get("relevance", 0)) <= 1:
                neg_seg = seg_index.get(
                    (row["video"], int(row.get("segment_index", -1)))
                )
                hard_negatives.append(
                    {
                        "segment_text": neg_seg.text if neg_seg else "",
                        "start_time": float(row["start_time"]),
                        "end_time": float(row["end_time"]),
                        "video": row["video"],
                        "reason": f"manual_judgment_rel{row.get('relevance', 0)}",
                    }
                )

        entry = TrainingEntry(
            query_id=query_id,
            query=pos_row.get("query_text", ""),
            query_type=pos_row.get("query_type", "natural_search"),
            language=pos_row.get("language", "en"),
            video=video,
            positive=positive,
            hard_negatives=hard_negatives,
            metadata={
                "caption": seg.caption if seg else None,
                "ocr_text": seg.ocr_text if seg else None,
                "object_labels": seg.object_labels if seg else [],
                "auto_generated": False,
                "needs_review": False,
            },
        )
        entries.append(entry)

    print(f"Loaded {len(entries)} queries from {judgments_path}")
    return entries


# ──────────────────────────────────────────────
# Build full dataset
# ──────────────────────────────────────────────


def build_dataset(
    segments: List[SegmentInfo],
    entries: List[TrainingEntry],
    config: Dict,
) -> Dict:
    """
    Attach negatives to each query and produce the final dataset.
    """
    neg_cfg = config.get("data", {}).get("negatives", {})

    for entry in entries:
        # Sample negatives for entries that don't already have enough
        existing_negs = len(entry.hard_negatives)
        target_negs = config.get("training", {}).get("hard_negatives_per_query", 5)

        if existing_negs < target_negs:
            # Find the positive segment for reference
            pos_seg = None
            for seg in segments:
                if seg.video == entry.video and seg.segment_index == entry.positive.get(
                    "segment_index"
                ):
                    pos_seg = seg
                    break

            if pos_seg:
                auto_negs = sample_negatives(
                    positive=pos_seg,
                    query_text=entry.query,
                    all_segments=segments,
                    easy_count=neg_cfg.get("easy_count", 2),
                    medium_count=neg_cfg.get("medium_count", 2),
                    hard_temporal_count=neg_cfg.get("hard_temporal_count", 1),
                    hard_keyword_count=neg_cfg.get("hard_keyword_count", 1),
                )
                entry.hard_negatives.extend(auto_negs)

        # Trim to target count
        entry.hard_negatives = entry.hard_negatives[:target_negs]

    # Build output
    dataset = {
        "dataset_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "total_entries": len(entries),
        "total_videos": len(set(e.video for e in entries)),
        "query_type_distribution": {},
        "entries": [asdict(e) for e in entries],
    }

    # Count query types
    for e in entries:
        qt = e.query_type
        dataset["query_type_distribution"][qt] = (
            dataset["query_type_distribution"].get(qt, 0) + 1
        )

    return dataset


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def load_config(config_path: str = "training/config.yaml") -> Dict:
    """Load training configuration."""
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Config not found at {config_path}, using defaults")
        return {
            "data": {
                "dataset_path": "training/retrieval_dataset.json",
                "negatives": {
                    "easy_count": 2,
                    "medium_count": 2,
                    "hard_temporal_count": 1,
                    "hard_keyword_count": 1,
                },
            },
            "training": {"hard_negatives_per_query": 5},
        }

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Build retrieval training dataset for ATLAS"
    )
    parser.add_argument(
        "--config", default="training/config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--processed-dir", default="processed", help="Path to processed/ directory"
    )
    parser.add_argument(
        "--output", default=None, help="Output path (default: from config)"
    )
    parser.add_argument(
        "--auto-queries",
        action="store_true",
        help="Auto-generate candidate queries from transcript text",
    )
    parser.add_argument(
        "--from-judgments",
        default=None,
        help="Path to relevance_judgments.csv for manual queries",
    )
    parser.add_argument(
        "--queries-per-video",
        type=int,
        default=5,
        help="Number of queries to generate per video (with --auto-queries)",
    )
    parser.add_argument(
        "--use-db",
        action="store_true",
        help="Load segments from database instead of processed/ files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just load and validate data, don't write output",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    config = load_config(args.config)

    # 1. Load segments
    print("=" * 60)
    print("ATLAS Training Dataset Builder")
    print("=" * 60)

    if args.use_db:
        segments = load_segments_from_db()
    else:
        segments = load_segments_from_processed(args.processed_dir)

    if not segments:
        print("ERROR: No segments found. Exiting.")
        return

    # Statistics
    videos = set(s.video for s in segments)
    print(f"\nDataset statistics:")
    print(f"  Videos:   {len(videos)}")
    print(f"  Segments: {len(segments)}")
    print(f"  With captions: {sum(1 for s in segments if s.caption)}")
    print(f"  With OCR:      {sum(1 for s in segments if s.ocr_text)}")
    print(f"  Languages:     {set(s.language for s in segments)}")

    if args.dry_run:
        print("\n[DRY RUN] Data loaded and validated. No output written.")
        return

    # 2. Generate or load queries
    entries: List[TrainingEntry] = []

    if args.from_judgments:
        entries = load_queries_from_judgments(args.from_judgments, segments)
    elif args.auto_queries:
        entries = generate_queries_auto(
            segments, queries_per_video=args.queries_per_video
        )
    else:
        print("\nNo query source specified. Use --auto-queries or --from-judgments.")
        print("Run with --dry-run to just validate data loading.")
        return

    if not entries:
        print("ERROR: No queries generated. Exiting.")
        return

    # 3. Build dataset with negatives
    print(f"\nAttaching negatives to {len(entries)} queries...")
    dataset = build_dataset(segments, entries, config)

    # 4. Save
    output_path = Path(
        args.output
        or config.get("data", {}).get(
            "dataset_path", "training/retrieval_dataset_text.json"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Dataset saved to: {output_path}")
    print(f"  Total queries: {dataset['total_entries']}")
    print(f"  Videos covered: {dataset['total_videos']}")
    print(f"  Query types: {dataset['query_type_distribution']}")
    print(f"{'=' * 60}")

    # Warn about review
    auto_count = sum(1 for e in entries if e.metadata.get("auto_generated"))
    if auto_count > 0:
        print(f"\n{auto_count} queries were auto-generated and need MANUAL REVIEW")
        print(f"   Open {output_path} and verify positive/negative assignments")


if __name__ == "__main__":
    main()
