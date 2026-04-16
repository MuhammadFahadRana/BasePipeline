"""LLM Enrichment (Hybrid Layer) for Documents.

Uses Qwen2-VL to optionally analyze page images and generate:
- Page summaries
- Entity tags
"""

import os
from PIL import Image
from typing import Dict, Optional

class DocumentEnricher:
    """Enriches document pages using Qwen2-VL."""

    def __init__(self):
        self.enabled = os.getenv("VISUAL_ENRICHMENT_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
        self._extractor = None

    def _ensure_extractor(self) -> bool:
        if not self.enabled:
            return False
            
        if self._extractor is not None:
            return True

        try:
            from extract_visual_features import VisualFeatureExtractor
            from transcriber_utils import get_device
            
            device = get_device()
            load_in_4bit = os.getenv("VISUAL_ENRICHMENT_LOAD_IN_4BIT", "false").strip().lower() in ("1", "true", "yes", "on")
            
            print("Loading Qwen2-VL for Document Enrichment...")
            self._extractor = VisualFeatureExtractor(
                model_name=os.getenv("VISUAL_ENRICHMENT_MODEL", "Qwen/Qwen2-VL-7B-Instruct"),
                device=device,
                load_in_4bit=load_in_4bit and device == "cuda"
            )
            return True
        except Exception as e:
            print(f"Warning: Document LLM enrichment disabled. Failed to load VisualFeatureExtractor: {e}")
            self.enabled = False
            return False

    def enrich_page(self, image_path: str) -> Optional[str]:
        """
        Analyze a page image to produce a summary.
        Returns the caption/summary.
        """
        if not self._ensure_extractor():
            return None

        try:
            result = self._extractor.analyze_image(image_path)
            # The prompt in exact_visual_features.py asks for a one-sentence description.
            caption = result.get("caption")
            return caption
        except Exception as e:
            print(f"Enrichment failed for {image_path}: {e}")
            return None
