import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.config import get_db
from database.models import TranscriptSegment, Embedding
from embeddings.text_embeddings import get_embedding_generator
from dotenv import load_dotenv

load_dotenv()

def re_embed_database():
    print("="*60)
    print("DATABASE EMBEDDING BACKFILL SCRIPT")
    print("="*60)
    
    db = next(get_db())
    
    # Check total segments
    total_segments = db.query(TranscriptSegment).count()
    print(f"Found {total_segments} transcript segments in the database.")
    
    if total_segments == 0:
        print("No segments found. Skipping.")
        return
        
    model_name = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    lora_path = os.getenv("EMBEDDING_LORA_PATH")
    print(f"Target Model: {model_name}")
    if lora_path:
        print(f"Target LoRA:  {lora_path}")
        
    print("\nInitializing embedding generator... (This may take a moment)")
    generator = get_embedding_generator(model_name=model_name)
    
    print("\nStarting backfill process...")
    
    # We process in batches
    batch_size = 64
    segments = db.query(TranscriptSegment).all()
    
    # Wipe old embeddings for this model to prevent duplicates
    deleted = db.query(Embedding).filter(Embedding.embedding_model == model_name).delete()
    db.commit()
    print(f"Cleared {deleted} old embeddings for base model '{model_name}'.")
    
    successful = 0
    
    for i in tqdm(range(0, len(segments), batch_size), desc="Re-embedding segments"):
        batch = segments[i:i+batch_size]
        texts = [seg.text for seg in batch]
        
        try:
            # Generate new embeddings using the fine-tuned model
            # For Qwen3, it's recommended to add an instruction if it's asymmetric search,
            # but since we fine-tuned it, we stick to the format used in training.
            embeddings_matrix = generator.encode(texts, batch_size=len(texts), show_progress=False)
            
            # Create embedding DB objects
            new_embeddings = []
            for j, seg in enumerate(batch):
                emb_vector = embeddings_matrix[j].tolist()
                new_embeddings.append(
                    Embedding(
                        segment_id=seg.id,
                        scene_id=seg.scene_id,
                        embedding=emb_vector,
                        embedding_model=model_name
                    )
                )
            
            # Bulk save
            db.bulk_save_objects(new_embeddings)
            db.commit()
            successful += len(batch)
            
        except Exception as e:
            db.rollback()
            print(f"\nError processing batch {i} to {i+batch_size}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n" + "="*60)
    print(f"BACKFILL COMPLETE: Successfully embedded {successful}/{total_segments} segments.")
    print("="*60)

if __name__ == "__main__":
    re_embed_database()
