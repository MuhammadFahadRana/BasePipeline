#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TED dataset runner for ATLAS BasicVideoPipeline.

What it does
------------
1. Loads ted_main.csv for TED metadata.
2. Loads transcripts.xlsx for ground-truth transcript text.
3. Merges rows on normalized TED talk URL.
4. Downloads the TED talk video with yt-dlp.
5. Reuses your BasicVideoPipeline class directly to:
   - transcribe the video
   - detect/refine/enrich scenes
   - save transcript/scenes/results in the same output structure as basic_pipeline.py
6. Compares the predicted transcript against the provided transcript.
7. Writes per-video evaluation CSV/JSON summaries.

Place this file in the SAME folder as basic_pipeline.py on the cluster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from jiwer import cer, wer

# Make sure local imports work when this file is placed beside basic_pipeline.py
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from basic_pipeline import BasicVideoPipeline  # noqa: E402


# --------------------------------------------------
# Logging / helpers
# --------------------------------------------------

def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_url(url: Any) -> Optional[str]:
    if url is None:
        return None
    if isinstance(url, float) and pd.isna(url):
        return None
    text = str(url).strip().replace("\n", "").replace("\r", "")
    if not text:
        return None
    text = re.sub(r"#.*$", "", text)
    text = re.sub(r"\?.*$", "", text)
    text = text.rstrip("/")
    return text or None


def slugify(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:max_len] or "item"


def sha1_short(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def normalize_eval_text(text: str) -> str:
    """Light normalization for more stable WER/CER comparisons."""
    text = str(text or "")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predicted_text_from_results(results: Dict[str, Any]) -> str:
    transcription = results.get("transcription", {}) if isinstance(results, dict) else {}
    text = transcription.get("text", "") if isinstance(transcription, dict) else ""
    if text:
        return normalize_eval_text(text)

    segments = transcription.get("segments", []) if isinstance(transcription, dict) else []
    if isinstance(segments, list):
        joined = " ".join(
            str(seg.get("text", "")).strip()
            for seg in segments
            if isinstance(seg, dict) and seg.get("text")
        ).strip()
        return normalize_eval_text(joined)
    return ""


# --------------------------------------------------
# Dataset model / loading
# --------------------------------------------------

@dataclass
class TedItem:
    idx: int
    url: str
    title: str
    speaker: str
    slug: str
    ground_truth: str
    metadata: Dict[str, Any]


def load_dataset(ted_csv: Path, transcripts_xlsx: Path) -> List[TedItem]:
    ted_df = pd.read_csv(ted_csv)
    gt_df = pd.read_excel(transcripts_xlsx)

    if "url" not in ted_df.columns:
        raise ValueError("ted_main.csv must contain a 'url' column.")
    if "url" not in gt_df.columns or "transcript" not in gt_df.columns:
        raise ValueError("transcripts.xlsx must contain 'url' and 'transcript' columns.")

    ted_df["url_norm"] = ted_df["url"].map(normalize_url)
    gt_df["url_norm"] = gt_df["url"].map(normalize_url)

    ted_df = ted_df.dropna(subset=["url_norm"]).copy()
    gt_df = gt_df.dropna(subset=["url_norm", "transcript"]).copy()

    gt_df["transcript"] = gt_df["transcript"].astype(str)
    gt_df = gt_df[["url_norm", "transcript"]].drop_duplicates(subset=["url_norm"])

    merged = ted_df.merge(gt_df, on="url_norm", how="inner")
    items: List[TedItem] = []

    for _, row in merged.iterrows():
        speaker = str(row.get("name", row.get("main_speaker", "")) or "")
        title = str(row.get("title", "") or "")
        url = str(row["url_norm"])
        ground_truth = normalize_eval_text(row.get("transcript", ""))

        slug = slugify(f"{speaker}-{title}")
        if slug == "item":
            slug = f"ted-{sha1_short(url)}"

        metadata = {}
        for key, value in row.to_dict().items():
            if pd.isna(value):
                metadata[key] = None
            else:
                metadata[key] = value

        items.append(
            TedItem(
                idx=len(items),
                url=url,
                title=title,
                speaker=speaker,
                slug=slug,
                ground_truth=ground_truth,
                metadata=metadata,
            )
        )

    return items


# --------------------------------------------------
# yt-dlp download
# --------------------------------------------------

def ensure_yt_dlp() -> None:
    if shutil.which("yt-dlp"):
        return
    raise RuntimeError("yt-dlp is not available in PATH. Install it in the active environment.")


def existing_video_for_slug(videos_dir: Path, slug: str) -> Optional[Path]:
    ignore_suffixes = {".json", ".part", ".temp", ".ytdl"}
    matches = []
    for path in sorted(videos_dir.glob(f"{slug}.*")):
        if path.suffix.lower() in ignore_suffixes:
            continue
        matches.append(path)
    return matches[0] if matches else None


def download_video(
    item: TedItem,
    videos_dir: Path,
    ydl_format: str,
    cookies_from_browser: Optional[str] = None,
    force_redownload: bool = False,
) -> Path:
    ensure_dir(videos_dir)
    ensure_yt_dlp()

    if not force_redownload:
        existing = existing_video_for_slug(videos_dir, item.slug)
        if existing is not None:
            log(f"Reusing existing download: {existing.name}")
            return existing

    outtmpl = str(videos_dir / f"{item.slug}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--newline",
        "--no-playlist",
        "-f",
        ydl_format,
        "--merge-output-format",
        "mp4",
        "-o",
        outtmpl,
    ]

    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])

    cmd.append(item.url)

    log(f"Downloading: {item.url}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed for {item.url}\n{result.stdout}")

    downloaded = existing_video_for_slug(videos_dir, item.slug)
    if downloaded is None:
        raise FileNotFoundError(f"Download finished but no video file was found for slug: {item.slug}")

    return downloaded


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def compare_texts(reference: str, hypothesis: str) -> Dict[str, Any]:
    reference = normalize_eval_text(reference)
    hypothesis = normalize_eval_text(hypothesis)

    if not reference:
        return {
            "wer": None,
            "cer": None,
            "reference_chars": 0,
            "hypothesis_chars": len(hypothesis),
        }
    if not hypothesis:
        return {
            "wer": 1.0,
            "cer": 1.0,
            "reference_chars": len(reference),
            "hypothesis_chars": 0,
        }

    return {
        "wer": float(wer(reference, hypothesis)),
        "cer": float(cer(reference, hypothesis)),
        "reference_chars": len(reference),
        "hypothesis_chars": len(hypothesis),
    }


def resolve_output_paths(processed_dir: Path, model_name: str, video_stem: str) -> Dict[str, Path]:
    return {
        "transcript_json": processed_dir / "transcripts" / model_name / video_stem / "transcript.json",
        "transcript_txt": processed_dir / "transcripts" / model_name / video_stem / "transcript.txt",
        "scenes_json": processed_dir / "scenes" / video_stem / "scenes.json",
        "results_json": processed_dir / "results" / video_stem / "results.json",
        "report_html": processed_dir / "results" / video_stem / "report.html",
        "manifest_json": processed_dir / "results" / video_stem / "manifest.json",
    }


# --------------------------------------------------
# Main processing
# --------------------------------------------------

def process_item(
    item: TedItem,
    pipeline: BasicVideoPipeline,
    videos_dir: Path,
    processed_dir: Path,
    results_dir: Path,
    ydl_format: str,
    cookies_from_browser: Optional[str],
    use_hash: bool,
    force_pipeline: bool,
    force_redownload: bool,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "dataset_index": item.idx,
        "url": item.url,
        "speaker": item.speaker,
        "title": item.title,
        "slug": item.slug,
        "status": "started",
    }

    try:
        video_path = download_video(
            item=item,
            videos_dir=videos_dir,
            ydl_format=ydl_format,
            cookies_from_browser=cookies_from_browser,
            force_redownload=force_redownload,
        )
        row["video_path"] = str(video_path)

        log(f"Running pipeline on: {video_path.name}")
        results = pipeline.process_video(
            str(video_path),
            output_base=str(processed_dir),
            use_hash=use_hash,
            force=force_pipeline,
            generate_embeddings=False,
        )

        predicted_text = predicted_text_from_results(results)
        metrics = compare_texts(item.ground_truth, predicted_text)

        model_name = getattr(pipeline.transcriber, "model_name", "unknown")
        paths = resolve_output_paths(processed_dir=processed_dir, model_name=model_name, video_stem=video_path.stem)

        scene_analysis = results.get("scene_analysis", {}) if isinstance(results, dict) else {}
        processing_info = results.get("processing_info", {}) if isinstance(results, dict) else {}

        row.update({
            "status": "ok",
            "model_name": model_name,
            "scene_count": scene_analysis.get("num_scenes"),
            "processing_duration": processing_info.get("processing_duration"),
            "language": results.get("transcription", {}).get("language") if isinstance(results, dict) else None,
            "ground_truth_chars": len(item.ground_truth),
            "predicted_chars": len(predicted_text),
            **metrics,
            **{k: str(v) if v.exists() else None for k, v in paths.items()},
        })

        per_item_json = results_dir / f"{item.slug}.json"
        payload = {
            "meta": row,
            "ground_truth": item.ground_truth,
            "predicted_text": predicted_text,
            "ted_metadata": item.metadata,
        }
        per_item_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)

    return row


# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download TED talks with yt-dlp, run BasicVideoPipeline, and compare against provided transcripts."
    )
    parser.add_argument("--ted-csv", type=Path, required=True, help="Path to ted_main.csv")
    parser.add_argument("--transcripts-xlsx", type=Path, required=True, help="Path to transcripts.xlsx")
    parser.add_argument("--work-dir", type=Path, default=Path("ted_eval_work"))
    parser.add_argument("--videos-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)

    parser.add_argument("--backend", type=str, default="whisper")
    parser.add_argument("--model", type=str, default="large-v3")
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-db", action="store_true",
                        help="Skip database ingestion. Recommended for evaluation jobs.")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--use-hash", action="store_true")
    parser.add_argument("--force-pipeline", action="store_true",
                        help="Force reprocessing even if manifest/output cache exists.")
    parser.add_argument("--force-redownload", action="store_true",
                        help="Force yt-dlp to redownload even if a local file already exists.")

    parser.add_argument("--ydl-format", type=str, default="bv*+ba/b",
                        help="yt-dlp format selector. Default tries best video+audio, else best.")
    parser.add_argument("--cookies-from-browser", type=str, default=None,
                        help="Browser name for yt-dlp cookies, e.g. chrome, firefox, edge")
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=None,
        help="Path to local LoRA adapter folder, e.g. training/asr_checkpoints/lora_adapter",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    work_dir = args.work_dir.resolve()
    videos_dir = args.videos_dir.resolve() if args.videos_dir else work_dir / "TEDVideos"
    processed_dir = args.processed_dir.resolve() if args.processed_dir else work_dir / "TEDprocessed"
    results_dir = args.results_dir.resolve() if args.results_dir else work_dir / "TEDevaluation"

    ensure_dir(work_dir)
    ensure_dir(videos_dir)
    ensure_dir(processed_dir)
    ensure_dir(results_dir)

    items = load_dataset(args.ted_csv, args.transcripts_xlsx)
    if args.limit is not None:
        items = items[: args.limit]
    if args.start_index or args.end_index is not None:
        items = items[args.start_index : args.end_index]

    if not items:
        log("No TED items matched between ted_main.csv and transcripts.xlsx.")
        return 1

    merged_manifest = results_dir / "matched_dataset_preview.csv"
    preview_rows = [
        {
            "dataset_index": item.idx,
            "speaker": item.speaker,
            "title": item.title,
            "url": item.url,
            "slug": item.slug,
        }
        for item in items
    ]
    pd.DataFrame(preview_rows).to_csv(merged_manifest, index=False)

    log(f"Matched TED items: {len(items)}")
    log(f"Using backend={args.backend}, model={args.model}, threshold={args.threshold}, device={args.device}")


    if args.lora_path is not None:
        resolved_lora = args.lora_path.resolve()
        os.environ["ASR_LORA_PATH"] = str(resolved_lora)
        log(f"Using LoRA adapter: {resolved_lora}")
        
    pipeline = BasicVideoPipeline(
        backend=args.backend,
        model_variant={"name": args.model},
        scene_threshold=args.threshold,
        device=args.device,
        skip_ingest=args.skip_db,
    )

    all_rows: List[Dict[str, Any]] = []
    for i, item in enumerate(items, start=1):
        log(f"[{i}/{len(items)}] {item.speaker}: {item.title}")
        row = process_item(
            item=item,
            pipeline=pipeline,
            videos_dir=videos_dir,
            processed_dir=processed_dir,
            results_dir=results_dir,
            ydl_format=args.ydl_format,
            cookies_from_browser=args.cookies_from_browser,
            use_hash=args.use_hash,
            force_pipeline=args.force_pipeline,
            force_redownload=args.force_redownload,
        )
        all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(results_dir / "per_video_results.csv", index=False)

    df = pd.DataFrame(all_rows)
    ok_df = df[df["status"] == "ok"] if "status" in df else pd.DataFrame()

    summary = {
        "total": int(len(df)),
        "ok": int((df["status"] == "ok").sum()) if "status" in df else 0,
        "error": int((df["status"] == "error").sum()) if "status" in df else 0,
        "backend": args.backend,
        "model": args.model,
        "threshold": args.threshold,
        "device": args.device,
        "mean_wer": float(ok_df["wer"].dropna().mean()) if (not ok_df.empty and "wer" in ok_df) else None,
        "median_wer": float(ok_df["wer"].dropna().median()) if (not ok_df.empty and "wer" in ok_df) else None,
        "mean_cer": float(ok_df["cer"].dropna().mean()) if (not ok_df.empty and "cer" in ok_df) else None,
        "mean_scene_count": float(ok_df["scene_count"].dropna().mean()) if (not ok_df.empty and "scene_count" in ok_df) else None,
    }

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Done. Summary saved to: {summary_path}")
    log(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
