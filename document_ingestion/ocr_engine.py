"""OCR Engine for documents.

Supports Surya OCR (primary) and EasyOCR (fallback).
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image

class DocumentOCREngine:
    """Wrapper for Document OCR."""

    def __init__(self, force_easyocr: bool = False):
        self.device = "cuda" # assuming CUDA, adapt as needed from transcriber_utils
        self.force_easyocr = force_easyocr
        self._surya_model = None
        self._surya_processor = None
        self._easy_reader = None

    def _ensure_surya(self) -> bool:
        if self._surya_model is not None:
            return True
            
        if self.force_easyocr:
            return False

        try:
            from surya.ocr import run_ocr
            from surya.model.detection.model import load_model as load_det_model
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor
            from surya.settings import settings
            
            print("Loading Surya OCR (layout-aware, best for documents)...")
            self.det_model = load_det_model()
            self.rec_model = load_rec_model()
            self.rec_processor = load_rec_processor()
            self._has_surya = True
            print("[OK] Surya OCR loaded.")
            return True
        except ImportError:
            print("Surya OCR not installed. Falling back to EasyOCR.")
            return False
        except Exception as e:
            print(f"Failed to load Surya OCR: {e}. Falling back to EasyOCR.")
            return False

    def _ensure_easyocr(self) -> bool:
        if self._easy_reader is not None:
            return True

        print("Loading EasyOCR (fallback)...")
        from embeddings.ocr import get_ocr_reader
        self._easy_reader = get_ocr_reader(languages=["en", "no"], use_gpu=(self.device == "cuda"))
        return True

    def extract_text_from_image(self, image: Image.Image, langs: List[str] = ["en", "no"]) -> Tuple[str, float]:
        """
        Extract text from a PIL Image.
        Returns: (extracted_text, average_confidence)
        """
        if not self.force_easyocr and self._ensure_surya():
            try:
                from surya.ocr import run_ocr
                predictions = run_ocr([image], [langs], self.det_model, self.rec_model, self.rec_processor)
                
                # predictions is a list containing a dict or object per image
                if not predictions:
                    return "", 0.0
                    
                result = predictions[0]
                text_lines = []
                confidences = []
                
                for line in result.text_lines:
                    text_lines.append(line.text)
                    confidences.append(line.confidence)
                    
                text = "\n".join(text_lines)
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                
                return text, avg_conf
            except Exception as e:
                print(f"Surya OCR failed: {e}. Retrying with EasyOCR...")

        # Fallback to EasyOCR
        if self._ensure_easyocr():
            # EasyOCR works on paths or numpy arrays. Convert PIL to numpy-like if needed or save temp.
            # Easiest is saving to temp, or we can use numpy.
            import numpy as np
            img_np = np.array(image)
            
            detections = self._easy_reader.extract_with_confidence(img_np, confidence_threshold=0.35)
            if not detections:
                return "", 0.0
                
            text_parts = []
            confidences = []
            for det in detections:
                text_parts.append(det.get("text", ""))
                conf = det.get("confidence")
                if conf is not None:
                    confidences.append(float(conf))
                    
            text = " ".join(text_parts).strip()
            avg_conf = sum(confidences) / len(confidences) if confidences else 1.0
            
            return text, avg_conf
            
        return "", 0.0

    def extract_from_pdf_page(self, pdf_path: str, page_num: int) -> Tuple[str, float]:
        """
        Render a PDF page to image and perform OCR.
        page_num is 0-indexed.
        """
        import PyMuPDF
        doc = PyMuPDF.open(pdf_path)
        if page_num >= len(doc):
            return "", 0.0
            
        page = doc[page_num]
        
        # Render at ~300 DPI for good OCR quality
        zoom = 300 / 72 
        mat = PyMuPDF.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        
        return self.extract_text_from_image(img)
