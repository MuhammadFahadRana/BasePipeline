"""
evaluate_retrieval.py

Evaluate retrieval quality for ATLAS by running queries against the database
and computing IR metrics + timestamp-specific metrics.

Usage:
    python -m training.evaluate_retrieval                                    # baseline eval
    python -m training.evaluate_retrieval --adapter training/checkpoints/lora_adapter  # fine-tuned
    python -m training.evaluate_retrieval --model Qwen/Qwen3-Embedding-0.6B  # explicit model
"""

import json
import argparse
import sys
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


# ──────────────────────────────────────────────
# Metric computations
# ──────────────────────────────────────────────

def recall_at_k(ranks: List[int], k: int) -> float:
    """Fraction of queries where the correct answer is in the top-k results."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r <= k) / len(ranks)


def mrr(ranks: List[int]) -> float:
    """Mean Reciprocal Rank."""
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks) / len(ranks)


def ndcg_at_k(ranked_relevances: List[List[int]], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k.

    Args:
        ranked_relevances: For each query, a list of relevance scores
                           in the order returned by the model
        k: Cutoff
    """
    if not ranked_relevances:
        return 0.0

    ndcg_scores = []
    for rels in ranked_relevances:
        dcg = sum(
            rel / np.log2(i + 2) for i, rel in enumerate(rels[:k])
        )
        # Ideal DCG: sort relevances descending
        ideal_rels = sorted(rels, reverse=True)[:k]
        idcg = sum(
            rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels)
        )
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return sum(ndcg_scores) / len(ndcg_scores)


def timestamp_iou(
    pred_start: float, pred_end: float,
    gt_start: float, gt_end: float,
) -> float:
    """Intersection-over-Union for two time intervals."""
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0.0, inter_end - inter_start)
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    return intersection / union if union > 0 else 0.0


def timestamp_hit(
    pred_start: float, gt_start: float, tolerance: float
) -> bool:
    """Is the predicted start time within ±tolerance seconds of ground truth?"""
    return abs(pred_start - gt_start) <= tolerance


# ──────────────────────────────────────────────
# Load evaluation data
# ──────────────────────────────────────────────

def load_eval_queries(
    dataset_path: Optional[str] = None,
    judgments_path: Optional[str] = None,
) -> List[Dict]:
    """
    Load evaluation queries from either retrieval_dataset.json or relevance_judgments.csv.

    Returns list of dicts with: query_id, query, video, start_time, end_time, segment_text
    """
    queries = []

    # Try judgments CSV first (manual annotations take priority)
    if judgments_path and Path(judgments_path).exists():
        with open(judgments_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            from collections import defaultdict
            by_qid = defaultdict(list)
            for row in reader:
                by_qid[row["query_id"]].append(row)

            for qid, rows in by_qid.items():
                # Find the best positive (highest relevance)
                rows.sort(key=lambda r: int(r.get("relevance", 0)), reverse=True)
                best = rows[0]
                if int(best.get("relevance", 0)) >= 2:
                    queries.append({
                        "query_id": qid,
                        "query": best.get("query_text", ""),
                        "video": best["video"],
                        "start_time": float(best["start_time"]),
                        "end_time": float(best["end_time"]),
                        "segment_text": "",  # not always in CSV
                        "all_judgments": rows,  # for nDCG computation
                    })

    # Fall through to dataset JSON if judgments yielded nothing
    if not queries and dataset_path and Path(dataset_path).exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        for entry in dataset.get("entries", []):
            pos = entry.get("positive", {})
            queries.append({
                "query_id": entry["query_id"],
                "query": entry["query"],
                "video": entry["video"],
                "start_time": pos.get("start_time", 0),
                "end_time": pos.get("end_time", 0),
                "segment_text": pos.get("segment_text", ""),
            })

    return queries


# ──────────────────────────────────────────────
# Retrieval evaluation
# ──────────────────────────────────────────────

def evaluate_against_corpus(
    model,
    eval_queries: List[Dict],
    corpus_segments: List[Dict],
    batch_size: int = 32,
) -> Dict:
    """
    Run all eval queries against the full corpus and compute metrics.

    Args:
        model: SentenceTransformer (or fine-tuned variant) with .encode()
        eval_queries: List of query dicts from load_eval_queries
        corpus_segments: List of dicts with keys: text, video, start_time, end_time
        batch_size: Encoding batch size
    """
    if not eval_queries or not corpus_segments:
        return {"error": "No queries or corpus segments"}

    # Encode corpus
    print(f"Encoding {len(corpus_segments)} corpus segments...")
    corpus_texts = [s["text"] for s in corpus_segments]
    corpus_embeddings = model.encode(
        corpus_texts, batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=True,
    )

    # Encode queries
    print(f"Encoding {len(eval_queries)} queries...")
    query_texts = [q["query"] for q in eval_queries]
    query_embeddings = model.encode(
        query_texts, batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=False,
    )

    # Compute similarities and find ranks
    ranks = []
    timestamp_ious = []
    hits_5s = []
    hits_10s = []
    abs_errors = []

    for i, query in enumerate(eval_queries):
        q_emb = query_embeddings[i]
        scores = np.dot(corpus_embeddings, q_emb)  # (corpus_size,)
        ranked_indices = np.argsort(-scores)

        # Find the rank of any segment from the correct video
        # that overlaps with the ground truth time range
        gt_video = query["video"]
        gt_start = query["start_time"]
        gt_end = query["end_time"]

        rank = len(corpus_segments)  # worst case
        best_iou = 0.0
        pred_start = None

        for rank_pos, seg_idx in enumerate(ranked_indices):
            seg = corpus_segments[seg_idx]
            if seg["video"] == gt_video:
                iou = timestamp_iou(
                    seg["start_time"], seg["end_time"],
                    gt_start, gt_end,
                )
                if iou > 0.1:  # consider a match if IoU > 0.1
                    rank = rank_pos + 1  # 1-indexed
                    best_iou = iou
                    pred_start = seg["start_time"]
                    break

        ranks.append(rank)
        timestamp_ious.append(best_iou)

        if pred_start is not None:
            abs_errors.append(abs(pred_start - gt_start))
            hits_5s.append(timestamp_hit(pred_start, gt_start, 5.0))
            hits_10s.append(timestamp_hit(pred_start, gt_start, 10.0))
        else:
            abs_errors.append(float("inf"))
            hits_5s.append(False)
            hits_10s.append(False)

    # Compute aggregate metrics
    metrics = {
        # IR metrics
        "recall@1": round(recall_at_k(ranks, 1), 4),
        "recall@5": round(recall_at_k(ranks, 5), 4),
        "recall@10": round(recall_at_k(ranks, 10), 4),
        "mrr": round(mrr(ranks), 4),

        # Timestamp metrics
        "avg_timestamp_iou": round(np.mean(timestamp_ious), 4),
        "median_abs_error_s": round(np.median([e for e in abs_errors if e != float("inf")]), 2)
            if any(e != float("inf") for e in abs_errors) else None,
        "hit_rate_5s": round(sum(hits_5s) / len(hits_5s), 4) if hits_5s else 0,
        "hit_rate_10s": round(sum(hits_10s) / len(hits_10s), 4) if hits_10s else 0,

        # Counts
        "total_queries": len(eval_queries),
        "corpus_size": len(corpus_segments),
    }

    return metrics


def load_corpus_from_processed(processed_dir: str = "processed") -> List[Dict]:
    """Load all transcript segments as a flat corpus for evaluation."""
    processed_dir = Path(processed_dir)
    results_files = list(processed_dir.glob("*/*/results.json"))
    if not results_files:
        results_files = list(processed_dir.glob("results/*/results.json"))

    corpus = []
    for rf in results_files:
        try:
            with open(rf, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            continue

        video_stem = Path(results.get("video", {}).get("filename", rf.parent.name)).stem
        for seg in results.get("transcription", {}).get("segments", []):
            text = seg.get("text", "").strip()
            if text:
                corpus.append({
                    "text": text,
                    "video": video_stem,
                    "start_time": seg.get("start", 0),
                    "end_time": seg.get("end", 0),
                })

    return corpus


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ATLAS retrieval quality"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-Embedding-0.6B",
        help="Base embedding model name"
    )
    parser.add_argument(
        "--adapter", default=None,
        help="Path to LoRA adapter directory (for fine-tuned evaluation)"
    )
    parser.add_argument(
        "--dataset", default="training/retrieval_dataset.json",
        help="Path to retrieval_dataset.json"
    )
    parser.add_argument(
        "--judgments", default="training/relevance_judgments.csv",
        help="Path to relevance_judgments.csv"
    )
    parser.add_argument(
        "--processed-dir", default="processed",
        help="Path to processed/ directory for corpus"
    )
    parser.add_argument(
        "--output", default="training/eval_results.json",
        help="Output path for evaluation results"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Encoding batch size"
    )
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    print(f"\n{'=' * 60}")
    print(f"ATLAS Retrieval Evaluation")
    print(f"{'=' * 60}")

    # Load model
    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model, trust_remote_code=True)

    # Load LoRA adapter if provided
    if args.adapter and Path(args.adapter).exists():
        print(f"Loading LoRA adapter: {args.adapter}")
        try:
            from peft import PeftModel
            base_transformer = model[0].auto_model
            model[0].auto_model = PeftModel.from_pretrained(
                base_transformer, args.adapter
            )
            print("LoRA adapter applied successfully")
        except Exception as e:
            print(f"Warning: Could not load LoRA adapter: {e}")
            print("Proceeding with base model")

    # Load evaluation queries
    judgments_path = args.judgments if Path(args.judgments).exists() else None
    dataset_path = args.dataset if Path(args.dataset).exists() else None

    eval_queries = load_eval_queries(
        dataset_path=dataset_path,
        judgments_path=judgments_path,
    )

    if not eval_queries:
        print("ERROR: No evaluation queries found. Create queries first.")
        print(f"  Expected: {args.judgments} or {args.dataset}")
        return

    print(f"Loaded {len(eval_queries)} evaluation queries")

    # Load corpus
    corpus = load_corpus_from_processed(args.processed_dir)
    if not corpus:
        print(f"ERROR: No corpus segments found in {args.processed_dir}")
        return

    print(f"Loaded {len(corpus)} corpus segments from {len(set(s['video'] for s in corpus))} videos")

    # Run evaluation
    print(f"\nRunning evaluation...")
    metrics = evaluate_against_corpus(
        model, eval_queries, corpus, batch_size=args.batch_size
    )

    # Display results
    print(f"\n{'=' * 60}")
    print(f"RETRIEVAL EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"\nIR Metrics:")
    print(f"  Recall@1:  {metrics.get('recall@1', 'N/A')}")
    print(f"  Recall@5:  {metrics.get('recall@5', 'N/A')}")
    print(f"  Recall@10: {metrics.get('recall@10', 'N/A')}")
    print(f"  MRR:       {metrics.get('mrr', 'N/A')}")
    print(f"\nTimestamp Metrics:")
    print(f"  Avg IoU:         {metrics.get('avg_timestamp_iou', 'N/A')}")
    print(f"  Median Abs Err:  {metrics.get('median_abs_error_s', 'N/A')}s")
    print(f"  Hit Rate ±5s:    {metrics.get('hit_rate_5s', 'N/A')}")
    print(f"  Hit Rate ±10s:   {metrics.get('hit_rate_10s', 'N/A')}")
    print(f"\nCorpus: {metrics.get('corpus_size', 0)} segments | "
          f"Queries: {metrics.get('total_queries', 0)}")
    print(f"{'=' * 60}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "adapter": args.adapter,
        "metrics": metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
