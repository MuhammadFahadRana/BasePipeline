# Retrieval Evaluation Evidence for Research Questions

Generated: 2026-05-11T11:24:24
Queries: 20
Processed units: 6911

## RQ1: What multimodal fusion strategies best align text queries to timestamped video segments?

- Best Recall@10 in this run: `BM25-only` with `1.0`.
- Best MRR in this run: `BM25-only` with `0.95`.
- Compare `Transcript only`, `Transcript + OCR`, `Transcript + captions`, and `Full multimodal ATLAS` rows to isolate the contribution of each evidence source.

## RQ2: How should we structure a persisted video index for fast retrieval and future reuse?

- The corpus rows in `retrieval_topk_results.csv` expose the persisted unit shape needed for reuse: video id/name, segment index, scene id, timestamp span, transcript, OCR, caption, labels, and scores.
- The summary table reports `corpus_units`, `coverage_pct`, `build_s`, and embedding-cache hits; use these to motivate precomputed text/vector fields and metadata indexes.

## RQ3: Which retrieval setup optimizes Precision/Recall@k and MRR under realistic latency constraints?

- Fastest P95 query latency in this run: `OCR only BM25` with `3.407` ms.
- Use `precision@k`, `recall@k`, `mrr`, `ndcg@10`, and `p95_query_latency_ms` together; high recall with unacceptable latency is a different trade-off than a lower-recall lexical baseline.

## RQ4: What are the trade-offs between on-the-fly processing and precomputed, database-backed indices?

- `build_s` and `corpus_encode_s` approximate index/precompute cost. `mean_query_latency_ms` and `p95_query_latency_ms` approximate online retrieval cost once evidence and embeddings exist.
- Dense and LoRA variants benefit strongly from cached/precomputed corpus embeddings; BM25 has cheaper build cost but may underperform semantic matching.
