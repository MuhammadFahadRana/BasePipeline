#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script exports text and visual embeddings from the database to TSV files.
# These files can be used with the TensorFlow Projector (https://projector.tensorflow.org/) to visualize the embeddings.

import os
import sys
import json
from sqlalchemy import text
from typing import List

sys.path.insert(0, os.getcwd())
from database.config import SessionLocal


def clean_text(text_val: str) -> str:
    if not text_val:
        return ""
    # Remove newlines and tabs to not break the TSV format
    return text_val.replace("\n", " ").replace("\t", " ").replace("\r", "").strip()


def export_text_embeddings():
    print("Exporting text embeddings...")
    db = SessionLocal()
    try:
        # Fetch up to 10,000 for projector (it gets slow with too many, but we'll try fetching all)
        query = text("""
            SELECT e.embedding, ts.text, v.filename, ts.start_time
            FROM embeddings e
            JOIN transcript_segments ts ON e.segment_id = ts.id
            JOIN videos v ON ts.video_id = v.id
            WHERE e.embedding IS NOT NULL
        """)
        result = db.execute(query).fetchall()

        with (
            open("projector_text_tensors.tsv", "w", encoding="utf-8") as f_tensors,
            open("projector_text_metadata.tsv", "w", encoding="utf-8") as f_meta,
        ):
            # Write header for metadata
            f_meta.write("Filename\tStart_Time\tText\n")

            count = 0
            for row in result:
                embedding = row[0]
                text_content = clean_text(row[1])
                filename = clean_text(row[2])
                start_time = row[3]

                # Parse embedding if it's a string, pgvector sometimes returns strings
                if isinstance(embedding, str):
                    vec = json.loads(embedding)
                else:
                    vec = list(embedding)

                # Write tensors (tab separated floats)
                f_tensors.write("\t".join(str(x) for x in vec) + "\n")

                # Write metadata
                time_str = f"{int(start_time // 3600):02d}:{int((start_time % 3600) // 60):02d}:{int(start_time % 60):02d}"
                f_meta.write(f"{filename}\t{time_str}\t{text_content}\n")
                count += 1

        print(f"Exported {count} text embeddings.")
    finally:
        db.close()


def export_visual_embeddings():
    print("Exporting visual embeddings...")
    db = SessionLocal()
    try:
        query = text("""
            SELECT ve.embedding, s.caption, v.filename, s.start_time
            FROM visual_embeddings ve
            JOIN scenes s ON ve.scene_id = s.id
            JOIN videos v ON s.video_id = v.id
            WHERE ve.embedding IS NOT NULL
        """)
        result = db.execute(query).fetchall()

        with (
            open("projector_visual_tensors.tsv", "w", encoding="utf-8") as f_tensors,
            open("projector_visual_metadata.tsv", "w", encoding="utf-8") as f_meta,
        ):
            # Write header for metadata
            f_meta.write("Filename\tStart_Time\tCaption\n")

            count = 0
            for row in result:
                embedding = row[0]
                caption = clean_text(row[1] or "No caption")
                filename = clean_text(row[2])
                start_time = row[3]

                if isinstance(embedding, str):
                    vec = json.loads(embedding)
                else:
                    vec = list(embedding)

                # Write tensors (tab separated floats)
                f_tensors.write("\t".join(str(x) for x in vec) + "\n")

                # Write metadata
                time_str = f"{int(start_time // 3600):02d}:{int((start_time % 3600) // 60):02d}:{int(start_time % 60):02d}"
                f_meta.write(f"{filename}\t{time_str}\t{caption}\n")
                count += 1

        print(f"Exported {count} visual embeddings.")
    finally:
        db.close()


if __name__ == "__main__":
    export_text_embeddings()
    export_visual_embeddings()
    print("Done! Files are ready for https://projector.tensorflow.org/")
