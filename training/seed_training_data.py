import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.config import get_db
from database.models import Base
from sqlalchemy import text
from datetime import datetime

# Direct query to populate relevance_judgments from retrieval_dataset.json
def seed_training_data():
    dataset_path = Path(__file__).parent / "retrieval_dataset.json"
    if not dataset_path.exists():
        print(f"Dataset {dataset_path} not found.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", [])
    if not entries:
        print("No entries found in dataset.")
        return
    
    db = next(get_db())
    
    # Check if data already exists to avoid duplicates
    existing_count = db.execute(text("SELECT COUNT(*) FROM relevance_judgments")).scalar()
    if existing_count > 0:
        print(f"relevance_judgments already has {existing_count} records. Skipping seed.")
        return

    print(f"Seeding {len(entries)} entries into relevance_judgments...")
    inserted = 0

    for idx, entry in enumerate(entries):
        video_name = entry.get("video_name")
        segment_text = entry.get("segment_text")
        start_time = entry.get("start_time", 0.0)
        end_time = entry.get("end_time", 0.0)
        
        # Get video ID
        video_res = db.execute(text("SELECT id FROM videos WHERE filename = :v"), {"v": video_name}).fetchone()
        if not video_res:
            continue
            
        video_id = video_res[0]
        
        # Determine exact segment id via timestamp and text bounds, or fallback to start/end checks
        segment_res = db.execute(
            text("SELECT id FROM transcript_segments WHERE video_id = :v AND start_time = :s"), 
            {"v": video_id, "s": start_time}
        ).fetchone()
        
        segment_id = segment_res[0] if segment_res else None

        # Add queries (anchor, easy, hard negatives, etc. usually mapped to the positive anchor query)
        for anchor_query in entry.get("anchor_queries", []):
            db.execute(
                text("""
                    INSERT INTO relevance_judgments 
                    (query_id, query_text, video_id, segment_id, start_time, end_time, relevance) 
                    VALUES (:qid, :qtext, :vid, :seg, :st, :et, :rel)
                """),
                {
                    "qid": f"auto_{video_id}_{idx}",
                    "qtext": anchor_query,
                    "vid": video_id,
                    "seg": segment_id,
                    "st": start_time,
                    "et": end_time,
                    "rel": 3  # exact answer
                }
            )
            inserted += 1
            
    db.commit()
    print(f"Successfully seeded {inserted} relevance judgments.")
    
    # Also seed a mock/template model_run if eval_results.json exists
    eval_path = Path(__file__).parent / "eval_results.json"
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        
        db.execute(
            text("""
                INSERT INTO model_runs (run_name, embedding_model, mrr, recall_at_1, recall_at_5, recall_at_10, train_samples, eval_samples)
                VALUES (:name, :model, :mrr, :r1, :r5, :r10, :train, :eval)
            """),
            {
                "name": "LoRA Embedding Evaluation (eval_results.json)",
                "model": "Qwen/Qwen3-Embedding-0.6B (LoRA Fine-tuned)",
                "mrr": eval_data.get("mrr", 0.0),
                "r1": eval_data.get("recall@1", 0.0),
                "r5": eval_data.get("recall@5", 0.0),
                "r10": eval_data.get("recall@10", 0.0),
                "train": eval_data.get("train_samples", 0),
                "eval": eval_data.get("eval_samples", 0)
            }
        )
        db.commit()
        print("Successfully seeded model_runs evaluation.")

if __name__ == "__main__":
    seed_training_data()
