from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.config import SessionLocal, test_connection
from database.models import Video, Scene, VisualEmbedding
from database.ingest import DataIngester


def main() -> None:
    processed = Path("processed")
    result_files = set()
    if (processed / "results").exists():
        result_files.update((processed / "results").glob("*/results.json"))
    result_files.update(processed.glob("*/*/results.json"))

    expected = {}
    for p in sorted({x.resolve() for x in result_files}):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            name = data.get("video", {}).get("filename")
            if name:
                expected[name] = p
        except Exception:
            continue

    if not test_connection():
        raise SystemExit("DB connection failed")

    s = SessionLocal()
    try:
        by_name = {v.filename: v for v in s.query(Video).all()}
        targets = []

        for name, v in by_name.items():
            sc = s.query(Scene).filter(Scene.video_id == v.id).count()
            vc = s.query(VisualEmbedding).join(Scene).filter(Scene.video_id == v.id).count()
            if sc > 0 and vc == 0 and not name.lower().endswith(".wav") and name in expected:
                targets.append(name)

        print(f"Videos needing visual backfill: {len(targets)}")

        fixed = 0
        failed = []

        with DataIngester() as ing:
            for i, name in enumerate(sorted(targets), 1):
                path = expected[name]
                try:
                    result = ing.ingest_video(
                        path,
                        generate_embeddings=False,
                        generate_visual_embeddings=True,
                        skip_existing=True,
                        update_existing=True,
                    )
                    print(f"[{i}/{len(targets)}] {name} -> {result.get('status', 'ok')}")
                    fixed += 1
                except Exception as e:
                    ing.db.rollback()
                    failed.append((name, str(e)))
                    print(f"[{i}/{len(targets)}] FAIL {name}: {e}")

        print("\nBackfill summary:")
        print(f"  fixed: {fixed}")
        print(f"  failed: {len(failed)}")
        for name, err in failed[:20]:
            print(f"  - {name}: {err}")

    finally:
        s.close()


if __name__ == "__main__":
    main()
