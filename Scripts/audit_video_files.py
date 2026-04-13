"""Audit video rows against files on disk and optionally delete broken rows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEOS_DIR = PROJECT_ROOT / "videos"
sys.path.insert(0, str(PROJECT_ROOT))

from database.config import SessionLocal
from database.models import Video


def resolve_video_file_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None

    candidate = Path(raw_path)
    if candidate.exists():
        return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)

    local_candidate = VIDEOS_DIR / os.path.basename(raw_path)
    if local_candidate.exists():
        return local_candidate

    if not candidate.is_absolute():
        relative_candidate = PROJECT_ROOT / candidate
        if relative_candidate.exists():
            return relative_candidate

    return None


def collect_missing_videos(session) -> list[Video]:
    missing: list[Video] = []
    for video in session.query(Video).order_by(Video.id).all():
        if resolve_video_file_path(video.file_path) is None:
            missing.append(video)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report videos whose DB file_path no longer resolves on disk."
    )
    parser.add_argument(
        "--delete-missing",
        action="store_true",
        help="Delete rows whose files are missing from disk.",
    )
    args = parser.parse_args()

    session = SessionLocal()
    try:
        all_videos = session.query(Video).count()
        missing = collect_missing_videos(session)

        print(f"Checked {all_videos} video rows under: {PROJECT_ROOT}")
        print(f"Missing rows: {len(missing)}")

        if missing:
            print("\nMissing video rows:")
            for video in missing:
                print(
                    f"  id={video.id} | filename={video.filename} | file_path={video.file_path}"
                )

        if not args.delete_missing:
            return 0

        if not missing:
            print("\nNo missing rows to delete.")
            return 0

        for video in missing:
            session.delete(video)
        session.commit()

        print(f"\nDeleted {len(missing)} missing video row(s).")
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
