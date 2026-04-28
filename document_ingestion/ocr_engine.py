"""OCR Engine for documents.

Supports Surya OCR (primary), PaddleOCR (secondary), and EasyOCR (fallback).
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
        self._paddle_ocr = None
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
            print(f"Failed to load Surya OCR: {e}. Falling back to PaddleOCR.")
            return False

    def _ensure_paddleocr(self) -> bool:
        if self._paddle_ocr is not None:
            return True

        print("Loading PaddleOCR (modern, fast layout-aware OCR)...")
        try:
            from paddleocr import PaddleOCR
            
            # Initialize PaddleOCR with English and Norwegian
            self._paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang=['en', 'no'],
                use_gpu=(self.device == "cuda"),
                show_log=False
            )
            print("[OK] PaddleOCR loaded.")
            return True
        except ImportError:
            print("PaddleOCR not installed. Install with: pip install paddleocr")
            return False
        except Exception as e:
            print(f"Failed to load PaddleOCR: {e}. Falling back to EasyOCR.")
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

    def _merge_text_lines(self, primary: str, secondary: str) -> str:
        """Merge lines from secondary into primary keeping novel lines only."""
        if not primary:
            return secondary or ""
        if not secondary:
            return primary

        def normalize_line(s: str) -> str:
            s = s or ""
            import re

            s = s.strip()
            s = re.sub(r"\s+", " ", s)
            s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
            return s.lower()

        primary_norm = normalize_line(primary)
        novel = []
        seen = set()
        for raw in secondary.splitlines():
            line = raw.strip()
            if len(line) < 3:
                continue
            n = normalize_line(line)
            if not n or n in seen:
                continue
            seen.add(n)
            if n in primary_norm:
                continue
            novel.append(line)

        if not novel:
            return primary

        merged = f"{primary}\n\n[OCR Enrichment]\n" + "\n".join(novel)
        return merged

    def extract_text_from_image(
        self,
        image: Image.Image,
        langs: List[str] = ["en", "no"],
        source: str = "document",
        enrich_with_paddle: bool = True,
    ) -> Tuple[str, float]:
        """
        Extract text from a PIL Image.

        Args:
            image: PIL Image
            langs: languages list
            source: "document" or "video". For "video" uses PaddleOCR first
                    and falls back to EasyOCR. For "document" prefers Surya but
                    can enrich with PaddleOCR if `enrich_with_paddle` is True.

        Returns: (extracted_text, average_confidence)
        """
        # VIDEO: force PaddleOCR -> EasyOCR (no Surya)
        if str(source).strip().lower() == "video":
            # Try Paddle first
            if self._ensure_paddleocr():
                try:
                    import numpy as np

                    img_np = np.array(image)
                    result = self._paddle_ocr.ocr(img_np, cls=True)
                    if result and result[0]:
                        text_lines = []
                        confidences = []
                        for line in result[0]:
                            if len(line) >= 2:
                                text = line[1][0]
                                conf = float(line[1][1]) if len(line[1]) > 1 else 0.9
                                if text and str(text).strip():
                                    text_lines.append(text)
                                    confidences.append(conf)
                        text = "\n".join(text_lines).strip()
                        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                        if text:
                            return text, avg_conf
                except Exception as e:
                    print(f"PaddleOCR (video) failed: {e}. Trying EasyOCR...")

            # Fallback to EasyOCR
            if self._ensure_easyocr():
                import numpy as np

                img_np = np.array(image)
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
                        detections.append({"text": str(text).strip(), "confidence": conf, "bbox": bbox})
                if not detections:
                    return "", 0.0
                text_parts = [d.get("text", "") for d in detections]
                confidences = [float(d.get("confidence", 1.0)) for d in detections]
                text = "\n".join(text_parts).strip()
                avg_conf = sum(confidences) / len(confidences) if confidences else 1.0
                return text, avg_conf

            return "", 0.0

        # DOCUMENT (default): prefer Surya, optionally enrich with Paddle
        if not self.force_easyocr and self._ensure_surya():
            try:
                from surya.ocr import run_ocr

                predictions = run_ocr([image], [langs], self.det_model, self.rec_model, self.rec_processor)
                if not predictions:
                    surya_text = ""
                    surya_conf = 0.0
                else:
                    result = predictions[0]
                    text_lines = []
                    confidences = []
                    for line in result.text_lines:
                        text_lines.append(line.text)
                        confidences.append(line.confidence)
                    surya_text = "\n".join(text_lines)
                    surya_conf = sum(confidences) / len(confidences) if confidences else 0.0

                # Optionally enrich Surya with Paddle results
                if enrich_with_paddle and self._ensure_paddleocr():
                    try:
                        import numpy as np

                        img_np = np.array(image)
                        pres = self._paddle_ocr.ocr(img_np, cls=True)
                        paddle_text = ""
                        paddle_conf = 0.0
                        if pres and pres[0]:
                            p_lines = []
                            p_confs = []
                            for pl in pres[0]:
                                if len(pl) >= 2:
                                    t = pl[1][0]
                                    c = float(pl[1][1]) if len(pl[1]) > 1 else 0.9
                                    if t and str(t).strip():
                                        p_lines.append(t)
                                        p_confs.append(c)
                            paddle_text = "\n".join(p_lines)
                            paddle_conf = sum(p_confs) / len(p_confs) if p_confs else 0.0

                        if paddle_text:
                            merged = self._merge_text_lines(surya_text, paddle_text)
                            combined_conf = max(surya_conf, paddle_conf)
                            return merged, combined_conf
                    except Exception as e:
                        print(f"PaddleOCR enrichment failed: {e}")

                return surya_text, surya_conf
            except Exception as e:
                print(f"Surya OCR failed: {e}. Trying PaddleOCR...")

        # If Surya not available or failed, fall back to Paddle then EasyOCR
        if self._ensure_paddleocr():
            try:
                import numpy as np

                img_np = np.array(image)
                result = self._paddle_ocr.ocr(img_np, cls=True)
                if result and result[0]:
                    text_lines = []
                    confidences = []
                    for line in result[0]:
                        if len(line) >= 2:
                            text = line[1][0]
                            conf = float(line[1][1]) if len(line[1]) > 1 else 0.9
                            text_lines.append(text)
                            confidences.append(conf)
                    text = "\n".join(text_lines).strip()
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                    return text, avg_conf
            except Exception as e:
                print(f"PaddleOCR failed: {e}. Falling back to EasyOCR...")

        # EasyOCR fallback
        if self._ensure_easyocr():
            import numpy as np

            img_np = np.array(image)
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
                    detections.append({"text": str(text).strip(), "confidence": conf, "bbox": bbox})
            if not detections:
                return "", 0.0
            text_parts = [d.get("text", "") for d in detections]
            confidences = [float(d.get("confidence", 1.0)) for d in detections]
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
