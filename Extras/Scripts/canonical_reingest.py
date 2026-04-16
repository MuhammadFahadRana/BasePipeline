from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.config import SessionLocal, test_connection
from database.models import Video
from database.ingest import DataIngester


def main() -> None:
    results_dir = Path("processed") / "results"
    if not results_dir.exists():
        raise SystemExit("processed/results not found")

    result_files = sorted(results_dir.glob("*/results.json"))
    if not result_files:
        raise SystemExit("No results.json files in processed/results")

    expected = {}
    parse_errors = []

    for p in result_files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            name = data.get("video", {}).get("filename")
            if not name:
                parse_errors.append((str(p), "missing video.filename"))
                continue
            expected[name] = p
        except Exception as e:
            parse_errors.append((str(p), str(e)))

    print(f"Canonical results files: {len(result_files)}")
    print(f"Parsable canonical videos: {len(expected)}")
    print(f"Parse issues: {len(parse_errors)}")
    for p, e in parse_errors[:10]:
        print(f"  - {p} | {e}")

    if not test_connection():
        raise SystemExit("DB connection failed")

    repaired = 0
    failed = []

    with DataIngester() as ing:
        # Force refresh each canonical video by changing fingerprint first
        for i, (name, path) in enumerate(sorted(expected.items()), 1):
            try:
                existing = ing.db.query(Video).filter(Video.filename == name).first()
                if existing:
                    existing.video_fingerprint = "__canonical_reingest__"
                    ing.db.commit()

                result = ing.ingest_video(
                    path,
                    generate_embeddings=True,
                    generate_visual_embeddings=True,
                    skip_existing=True,
                    update_existing=True,
                )
                print(f"[{i}/{len(expected)}] {name} -> {result.get('status', 'ok')}")
                repaired += 1
            except Exception as e:
                ing.db.rollback()
                failed.append((name, str(e)))
                print(f"[{i}/{len(expected)}] FAIL {name}: {e}")

    print("\nCanonical re-ingest summary:")
    print(f"  processed: {repaired}")
    print(f"  failed: {len(failed)}")
    for name, err in failed[:20]:
        print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
