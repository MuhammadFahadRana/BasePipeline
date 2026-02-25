"""
Qwen2-VL Visual Feature Extractor

Uses Alibaba's Qwen2-VL model to extract semantic information from video keyframes,
including descriptive captions and a list of detected objects.
"""

import torch
import json
import os
import warnings
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("Required libraries missing. Run: pip install transformers accelerate qwen-vl-utils")

warnings.filterwarnings("ignore")

class VisualFeatureExtractor:
    """
    Extracts visual features (captions, labels) using Qwen2-VL.
    """
    
    DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        load_in_4bit: bool = False,
        trust_remote_code: bool = True
    ):
        """
        Initialize Qwen2-VL extractor.
        
        Args:
            model_name: HuggingFace model ID
            device: "auto", "cuda", or "cpu"
            load_in_4bit: Whether to use 4-bit quantization (requires bitsandbytes)
            trust_remote_code: Whether to trust remote code from HF
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        
        print(f"\n{'='*60}")
        print(f"Qwen2-VL Visual Feature Extractor")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        print(f"4-bit:  {load_in_4bit}")
        print(f"{'='*60}\n")
        
        self._load_model(trust_remote_code)

    def _load_model(self, trust_remote_code: bool):
        """Load Qwen2-VL model and processor."""
        print(f"Loading {self.model_name}...")
        
        # Quantization config if requested
        quantization_config = None
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                print("Warning: bitsandbytes not installed. Falling back to FP16.")
                self.load_in_4bit = False

        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)
        
        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "trust_remote_code": trust_remote_code,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        elif self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        if self.device == "cpu":
            self.model = self.model.to("cpu")
            
        print(f"✓ {self.model_name} loaded successfully.")

    def analyze_image(self, image_path: Union[str, Path]) -> Dict[str, any]:
        """
        Analyze an image to get a caption, object labels, and OCR text.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing 'caption', 'object_labels', and 'ocr_text'
        """
        if not os.path.exists(image_path):
            return {"caption": "", "object_labels": [], "ocr_text": ""}
            
        # Prepare the query
        query = (
            "1. Describe this video scene in a short, descriptive sentence.\n"
            "2. List all important objects visible in the scene as comma-separated tags.\n"
            "3. Extract all visible text (OCR) from the scene. If no text is visible, say 'None'."
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": query},
                ],
            }
        ]
        
        # Process inputs
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Parse output_text (heuristic parsing)
        return self._parse_output(output_text)

    def _parse_output(self, text: str) -> Dict[str, any]:
        """
        Regex-anchored parser for the model's numbered-list output.

        Expected model format:
            1. <scene description>
            2. <comma-separated object tags>
            3. <OCR text, or "None">

        Using re.match anchored to line-start prevents false triggers on
        content that happens to contain "2." (e.g. "25th year stand-up").
        """
        import re

        # Pattern: line starts with a digit 1-3, then '.' or ')', optional space
        SECTION_RE = re.compile(r'^([1-3])[.)]\s*(.*)', re.IGNORECASE)
        # Named keyword markers that also switch sections
        KEYWORD_MAP = {
            "description": "caption", "caption": "caption",
            "tags": "labels", "objects": "labels", "object labels": "labels",
            "ocr": "ocr", "text": "ocr", "visible text": "ocr",
        }
        NONE_VALS = {"none", "none.", "n/a", "no text", "no visible text", ""}

        caption = ""
        object_labels: list = []
        ocr_parts: list = []
        current_section: str | None = None

        lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]

        for line in lines:
            lower = line.lower()

            # Try numbered-list anchor first
            m = SECTION_RE.match(line)
            if m:
                num, content = int(m.group(1)), m.group(2).strip()
                # Strip a leading "keyword:" if present
                if ":" in content:
                    key, _, content = content.partition(":")
                    content = content.strip()
                if num == 1:
                    current_section = "caption"
                    caption = content
                elif num == 2:
                    current_section = "labels"
                    object_labels = [t.strip() for t in content.split(",") if t.strip()]
                elif num == 3:
                    current_section = "ocr"
                    if content.lower() not in NONE_VALS:
                        ocr_parts = [content]
                continue

            # Try keyword marker (e.g. "Caption: …", "Tags: …", "OCR: …")
            hit_keyword = False
            for kw, section in KEYWORD_MAP.items():
                if lower.startswith(kw + ":") or lower.startswith("**" + kw):
                    current_section = section
                    content = line.split(":", 1)[1].strip() if ":" in line else ""
                    if section == "caption":
                        caption = content
                    elif section == "labels":
                        object_labels = [t.strip() for t in content.split(",") if t.strip()]
                    elif section == "ocr":
                        if content.lower() not in NONE_VALS:
                            ocr_parts = [content]
                    hit_keyword = True
                    break
            if hit_keyword:
                continue

            # Continuation of OCR section (multi-line OCR text)
            if current_section == "ocr":
                # Stop appending if we hit what looks like a new numbered item
                if re.match(r'^\d+[.)]\s', line):
                    continue
                if lower not in NONE_VALS:
                    ocr_parts.append(line)

        # Final cleanup
        caption = caption.strip()
        ocr_text = " ".join(ocr_parts).strip()
        if ocr_text.lower() in NONE_VALS:
            ocr_text = ""

        return {
            "caption":       caption or None,
            "object_labels": object_labels,
            "ocr_text":      ocr_text or None,
        }

if __name__ == "__main__":
    # Test script if run directly
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        extractor = VisualFeatureExtractor(load_in_4bit=True)
        result = extractor.analyze_image(img_path)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python extract_visual_features.py <path_to_image>")
