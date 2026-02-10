"""OCR module for extracting text from video keyframes.

This module provides OCR capabilities using EasyOCR to extract visible text
from video keyframes (like "Deepsea Stavanger" in title cards).
"""

import re
from pathlib import Path
from typing import Optional, List
from PIL import Image


class FrameOCR:
    """Extract text from images using OCR."""
    
    def __init__(self, languages: List[str] = None, use_gpu: bool = True):
        """
        Initialize OCR.
        
        Args:
            languages: List of language codes (default: ['en'])
            use_gpu: Use GPU if available
        """
        if languages is None:
            languages = ['en']
        
        self.languages = languages
        self.use_gpu = use_gpu
        self._reader = None
        
    def _ensure_reader(self):
        """Lazy load the OCR reader."""
        if self._reader is not None:
            return
        
        try:
            import easyocr
        except ImportError:
            raise RuntimeError(
                "EasyOCR not installed. Install with: pip install easyocr"
            )
        
        print(f"Loading EasyOCR model (languages: {self.languages})...")
        self._reader = easyocr.Reader(
            self.languages,
            gpu=self.use_gpu
        )
        print("✓ OCR model loaded")
    
    def extract_text(
        self,
        image_path: str,
        confidence_threshold: float = 0.5,
        clean: bool = True
    ) -> str:
        """
        Extract text from an image.
        
        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence score (0-1)
            clean: Clean and normalize extracted text
            
        Returns:
            Extracted text as single string
        """
        if not image_path or not Path(image_path).exists():
            return ""
        
        self._ensure_reader()
        
        try:
            # Perform OCR
            results = self._reader.readtext(str(image_path))
            
            # Extract text with confidence above threshold
            # results format: [[bbox, text, confidence], ...]
            texts = []
            for bbox, text, conf in results:
                if conf >= confidence_threshold and text.strip():
                    texts.append(text.strip())
            
            # Join all text
            full_text = " ".join(texts)
            
            # Clean if requested
            if clean:
                full_text = self._clean_text(full_text)
            
            return full_text
            
        except Exception as e:
            print(f"OCR failed for {image_path}: {e}")
            return ""
    
    def extract_with_confidence(
        self,
        image_path: str,
        confidence_threshold: float = 0.5
    ) -> List[dict]:
        """
        Extract text with bounding boxes and confidence scores.
        
        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of dicts with: {text, confidence, bbox}
        """
        if not image_path or not Path(image_path).exists():
            return []
        
        self._ensure_reader()
        
        try:
            results = self._reader.readtext(str(image_path))
            
            detections = []
            for bbox, text, conf in results:
                if conf >= confidence_threshold and text.strip():
                    detections.append({
                        'text': text.strip(),
                        'confidence': float(conf),
                        'bbox': bbox
                    })
            
            return detections
            
        except Exception as e:
            print(f"OCR failed for {image_path}: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize OCR output."""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special OCR artifacts
        # Keep alphanumeric, spaces, and common punctuation
        text = re.sub(r'[^\w\s\-.,!?@#&()\[\]]', '', text)
        
        return text.strip()
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """
        Extract meaningful keywords from OCR text.
        
        Args:
            text: OCR extracted text
            min_length: Minimum word length to keep
            
        Returns:
            List of keywords
        """
        if not text:
            return []
        
        # Split into words
        words = text.split()
        
        # Filter: length, alphanumeric
        keywords = []
        for word in words:
            word = word.strip('.,!?()[]')
            if len(word) >= min_length and any(c.isalnum() for c in word):
                keywords.append(word)
        
        return keywords


# Singleton instance for reuse
_ocr_instance: Optional[FrameOCR] = None


def get_ocr_reader(languages: List[str] = None, use_gpu: bool = True) -> FrameOCR:
    """
    Get or create global OCR reader instance.
    
    Args:
        languages: List of language codes
        use_gpu: Use GPU if available
        
    Returns:
        FrameOCR instance
    """
    global _ocr_instance
    
    if _ocr_instance is None:
        _ocr_instance = FrameOCR(languages=languages, use_gpu=use_gpu)
    
    return _ocr_instance
