"""Ingest processed document JSON files into the database.

Usage:
    python ingest_processed_documents.py
    python ingest_processed_documents.py --results-base processed/documents
"""

import argparse
import json
from pathlib import Path

from document_ingestion.ingest_documents import DocumentIngester


def iter_result_files(results_base: Path, recursive: bool):
    pattern = "**/results/results.json" if recursive else "*/results/results.json"
    return sorted(p for p in results_base.glob(pattern) if p.is_file())


def main():
    parser = argparse.ArgumentParser(
        description="Ingest processed document results JSON files into PostgreSQL"
    )
    parser.add_argument(
        "--results-base",
        type=str,
        default="processed/documents",
        help="Base folder containing per-document results folders",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan nested folders",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one document fails ingestion",
    )
    args = parser.parse_args()

    base = Path(args.results_base)
    if not base.exists():
        raise FileNotFoundError(f"Results folder not found: {base}")

    files = list(iter_result_files(base, recursive=args.recursive))
    if not files:
        print(f"No results files found under: {base}")
        return

    print(f"Found {len(files)} results files under: {base}")

    ok = 0
    failed = 0
    with DocumentIngester() as ingester:
        for idx, file_path in enumerate(files, 1):
            print(f"\n[{idx}/{len(files)}] Ingesting: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                ingester.ingest_document(results)
                ok += 1
            except Exception as e:
                failed += 1
                print(f"[ERROR] Failed to ingest {file_path}: {e}")
                if args.stop_on_error:
                    raise

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Success: {ok}/{len(files)}")
    print(f"Failed:  {failed}/{len(files)}")


if __name__ == "__main__":
    main()
