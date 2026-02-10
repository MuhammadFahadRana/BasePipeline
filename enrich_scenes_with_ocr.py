"""
Scene Enrichment Pipeline

Enriches existing scenes with OCR text extracted from keyframes.
Integrates seamlessly with existing infrastructure:
- Uses SQLAlchemy (existing database/models.py, database/config.py)
- Reuses vision embeddings module (embeddings/vision_embeddings.py)
- Reuses text embeddings module (embeddings/text_embeddings.py)
- New: OCR module (embeddings/ocr.py)

Features:
1. OCR text extraction (finds "Deepsea Stavanger" and other visible text)
2. Smart reprocessing (skips already processed scenes)
3. Batch processing with progress tracking
4. Error handling and logging

Usage:
    python enrich_scenes_with_ocr.py

    # Or import:
    from enrich_scenes_with_ocr import enrich_all_scenes
    enrich_all_scenes(skip_existing=True)
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Use existing infrastructure
from database.config import SessionLocal, engine
from database.models import Scene, Base
from embeddings.ocr import get_ocr_reader


class SceneEnricher:
    """Enrich scenes with OCR and other metadata."""
    
    def __init__(self):
        """Initialize enricher with existing modules."""
        self.ocr = None  # Lazy load
    
    def _ensure_ocr(self):
        """Lazy load OCR reader."""
        if self.ocr is None:
            print("Initializing OCR...")
            self.ocr = get_ocr_reader(languages=['en'], use_gpu=True)
    
    def enrich_scene_ocr(
        self,
        scene: Scene,
        skip_if_exists: bool = True,
        confidence_threshold: float = 0.5
    ) -> bool:
        """
        Extract OCR text from scene keyframe.
        
        Args:
            scene: Scene object from database
            skip_if_exists: Skip if ocr_text already exists
            confidence_threshold: Minimum OCR confidence
            
        Returns:
            True if processed, False if skipped
        """
        # Skip if already processed
        if skip_if_exists and scene.ocr_text:
            return False
        
        # Skip if no keyframe
        if not scene.keyframe_path:
            return False
        
        if not Path(scene.keyframe_path).exists():
            print(f"  Warning: Keyframe not found: {scene.keyframe_path}")
            return False
        
        # Extract text
        self._ensure_ocr()
        
        try:
            ocr_text = self.ocr.extract_text(
                scene.keyframe_path,
                confidence_threshold=confidence_threshold,
                clean=True
            )
            
            # Update scene
            scene.ocr_text = ocr_text if ocr_text else None
            scene.ocr_processed_at = datetime.utcnow()
            
            return True
            
        except Exception as e:
            print(f"  Error processing scene {scene.id}: {e}")
            return False


def enrich_all_scenes(
    skip_existing: bool = True,
    limit: Optional[int] = None,
    batch_size: int = 100
):
    """
    Enrich all scenes in database with OCR text.
    
    Args:
        skip_existing: Skip scenes that already have OCR text
        limit: Maximum number of scenes to process (None = all)
        batch_size: Commit every N scenes
    """
    print("=" * 60)
    print("SCENE ENRICHMENT PIPELINE")
    print("=" * 60)
    print()
    
    # Ensure database schema is up to date
    print("Checking database schema...")
    try:
        Base.metadata.create_all(bind=engine)
        print("Schema up to date")
    except Exception as e:
        print(f"Schema error: {e}")
        return
    
    # Initialize
    db = SessionLocal()
    enricher = SceneEnricher()
    
    try:
        # Get scenes to process
        query = db.query(Scene)
        
        if skip_existing:
            query = query.filter(
                (Scene.ocr_text == None) | (Scene.ocr_text == '')
            )
        
        if limit:
            query = query.limit(limit)
        
        scenes = query.all()
        total = len(scenes)
        
        print(f"\nFound {total} scenes to process")
        if skip_existing:
            print("  (Skipping scenes with existing OCR text)")
        print()
        
        if total == 0:
            print("Nothing to do!")
            return
        
        # Process scenes
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        for i, scene in enumerate(tqdm(scenes, desc="Enriching scenes"), 1):
            try:
                was_processed = enricher.enrich_scene_ocr(
                    scene,
                    skip_if_exists=skip_existing
                )
                
                if was_processed:
                    processed_count += 1
                else:
                    skipped_count += 1
                
                # Commit in batches
                if i % batch_size == 0:
                    db.commit()
                    tqdm.write(f"Committed {i} scenes")
                
            except Exception as e:
                error_count += 1
                tqdm.write(f"Error processing scene {scene.id}: {e}")
        
        # Final commit
        db.commit()
        
        # Summary
        print()
        print("=" * 60)
        print("ENRICHMENT COMPLETE")
        print("=" * 60)
        print(f"Total scenes: {total}")
        print(f"Processed:    {processed_count}")
        print(f"Skipped:      {skipped_count}")
        print(f"Errors:       {error_count}")
        print()
        
        # Show some examples
        if processed_count > 0:
            print("Sample OCR results:")
            print("-" * 60)
            
            sample_scenes = db.query(Scene).filter(
                Scene.ocr_text != None,
                Scene.ocr_text != ''
            ).limit(5).all()
            
            for scene in sample_scenes:
                video_name = scene.video.filename if scene.video else "Unknown"
                ocr_preview = (scene.ocr_text[:100] + "...") if len(scene.ocr_text) > 100 else scene.ocr_text
                print(f"\n{video_name} (Scene {scene.scene_id}, {scene.start_time:.1f}s):")
                print(f"  OCR: {ocr_preview}")
        
    finally:
        db.close()


def get_enrichment_stats():
    """Get statistics about OCR enrichment."""
    db = SessionLocal()
    
    try:
        total_scenes = db.query(Scene).count()
        scenes_with_ocr = db.query(Scene).filter(
            Scene.ocr_text != None,
            Scene.ocr_text != ''
        ).count()
        
        print("=" * 60)
        print("ENRICHMENT STATISTICS")
        print("=" * 60)
        print(f"Total scenes:        {total_scenes}")
        print(f"Scenes with OCR:     {scenes_with_ocr}")
        print(f"Coverage:            {scenes_with_ocr/total_scenes*100:.1f}%")
        print()
        
        # Show scenes with longest OCR text (likely title cards)
        print("Top scenes by OCR text length:")
        print("-" * 60)
        
        top_scenes = db.query(Scene).filter(
            Scene.ocr_text != None
        ).order_by(
            Scene.ocr_text.op('length')().desc()
        ).limit(10).all()
        
        for scene in top_scenes:
            video_name = scene.video.filename if scene.video else "Unknown"
            ocr_len = len(scene.ocr_text) if scene.ocr_text else 0
            ocr_preview = scene.ocr_text[:80] if scene.ocr_text else ""
            
            print(f"\n{video_name} (Scene {scene.scene_id}):")
            print(f"  Length: {ocr_len} chars")
            print(f"  Text: {ocr_preview}...")
        
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enrich scenes with OCR text")
    parser.add_argument(
        '--force',
        action='store_true',
        help='Reprocess scenes even if they already have OCR text'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of scenes to process'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show enrichment statistics'
    )
    
    args = parser.parse_args()
    
    if args.stats:
        get_enrichment_stats()
    else:
        enrich_all_scenes(
            skip_existing=not args.force,
            limit=args.limit
        )
