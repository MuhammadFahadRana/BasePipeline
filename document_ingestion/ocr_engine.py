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
        try:
            from transcriber_utils import get_device

            self.device = get_device()
        except Exception:
            self.device = "cpu"
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
        try:
            import easyocr
        except ImportError:
            print("EasyOCR is not installed. Install with: pip install easyocr")
            return False

        self._easy_reader = easyocr.Reader(
            ["en", "no"],
            gpu=(self.device == "cuda"),
            verbose=False,
        )
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
            import numpy as np
            img_np = np.array(image)

            # easyocr.Reader.readtext returns tuples: [bbox, text, confidence]
            raw_results = self._easy_reader.readtext(img_np)
            detections = []
            for item in raw_results:
                try:
                    bbox = item[0] if len(item) > 0 else None
                    text = item[1] if len(item) > 1 else ""
                    conf = float(item[2]) if len(item) > 2 else 1.0
                except (TypeError, ValueError):
                    continue
                if conf >= 0.35 and str(text).strip():
                    detections.append(
                        {"text": str(text).strip(), "confidence": conf, "bbox": bbox}
                    )
            if not detections:
                return "", 0.0
                
            text_parts = []
            confidences = []
            for det in detections:
                text_parts.append(det.get("text", ""))
                conf = det.get("confidence")
                if conf is not None:
                    confidences.append(float(conf))
                    
            text = "\n".join(text_parts).strip()
            avg_conf = sum(confidences) / len(confidences) if confidences else 1.0
            
            return text, avg_conf
            
        return "", 0.0

    def extract_from_pdf_page(self, pdf_path: str, page_num: int) -> Tuple[str, float]:
        """
        Render a PDF page to image and perform OCR.
        page_num is 0-indexed.
        """
        try:
            import fitz as pymupdf
        except ImportError:
            try:
                import pymupdf
            except ImportError as exc:
                raise ImportError(
                    "PyMuPDF is not installed. Run: pip install pymupdf"
                ) from exc

        doc = pymupdf.open(pdf_path)
        if page_num >= len(doc):
            return "", 0.0
            
        page = doc[page_num]
        
        # Render at ~300 DPI for good OCR quality
        zoom = 300 / 72 
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        
        return self.extract_text_from_image(img)
