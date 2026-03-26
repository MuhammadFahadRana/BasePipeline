from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.config import SessionLocal, test_connection
from database.models import Video, Scene, TranscriptSegment
from database.ingest import DataIngester


def discover_expected(processed: Path):
    result_files = set()
    if (processed / "results").exists():
        result_files.update((processed / "results").glob("*/results.json"))
    result_files.update(processed.glob("*/*/results.json"))
    result_files = sorted({p.resolve() for p in result_files})

    expected = {}
    parse_errors = []

    for p in result_files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            vname = data.get("video", {}).get("filename")
            scenes = len(data.get("scene_analysis", {}).get("scenes", []) or [])
            segs = len(data.get("transcription", {}).get("segments", []) or [])
            if not vname:
                parse_errors.append((str(p), "missing video.filename"))
                continue
            expected[vname] = {"path": p, "scenes": scenes, "segments": segs}
        except Exception as e:
            parse_errors.append((str(p), str(e)))

    return expected, parse_errors


def audit(session, expected):
    db_videos = session.query(Video).all()
    db_by_name = {v.filename: v for v in db_videos}

    missing = []
    mismatch = []

    for name, info in expected.items():
        v = db_by_name.get(name)
        if not v:
            missing.append(name)
            continue
        s_count = session.query(Scene).filter(Scene.video_id == v.id).count()
        t_count = session.query(TranscriptSegment).filter(TranscriptSegment.video_id == v.id).count()
        if s_count != info["scenes"] or t_count != info["segments"]:
            mismatch.append((name, info["scenes"], s_count, info["segments"], t_count))

    return missing, mismatch


def main():
    processed = Path("processed")
    expected, parse_errors = discover_expected(processed)

    print(f"Discovered expected videos from results: {len(expected)}")
    print(f"Parse issues: {len(parse_errors)}")
    for p, e in parse_errors[:5]:
        print(f"  - {p} | {e}")

    if not test_connection():
        raise SystemExit("DB connection failed")

    session = SessionLocal()
    try:
        missing, mismatch = audit(session, expected)
        print(f"Before repair -> missing: {len(missing)}, mismatch: {len(mismatch)}")

        targets = sorted(set(missing + [m[0] for m in mismatch]))
        if not targets:
            print("No repair needed.")
            return

        repaired = 0
        failed = []

        with DataIngester() as ingester:
            for idx, name in enumerate(targets, 1):
                path = expected[name]["path"]
                try:
                    existing = ingester.db.query(Video).filter(Video.filename == name).first()
                    if existing:
                        existing.video_fingerprint = "__force_reingest__"
                        ingester.db.commit()

                    result = ingester.ingest_video(
                        path,
                        generate_embeddings=True,
                        generate_visual_embeddings=True,
                        skip_existing=True,
                        update_existing=True,
                    )
                    print(f"[{idx}/{len(targets)}] {name} -> {result.get('status', 'ok')}")
                    repaired += 1
                except Exception as e:
                    ingester.db.rollback()
                    failed.append((name, str(e)))
                    print(f"[{idx}/{len(targets)}] FAIL {name}: {e}")

        missing_after, mismatch_after = audit(session, expected)
        print("\n=== FINAL AUDIT ===")
        print(f"Repaired: {repaired}")
        print(f"Failed during repair: {len(failed)}")
        if failed:
            for n, e in failed[:20]:
                print(f"  - {n}: {e}")
        print(f"Missing after repair: {len(missing_after)}")
        print(f"Mismatch after repair: {len(mismatch_after)}")
        for row in mismatch_after[:20]:
            n, exp_s, db_s, exp_t, db_t = row
            print(f"  - {n} | scenes {exp_s}/{db_s} | segs {exp_t}/{db_t}")

    finally:
        session.close()


if __name__ == "__main__":
    main()
