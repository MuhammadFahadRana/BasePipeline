"""
Visual feature extractor for image/video frames.

Primarily designed for Qwen-VL checkpoints, but can also load other
Hugging Face vision-language models that support image-text generation.
"""

import torch
import json
import os
import re
import warnings
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    from transformers import AutoProcessor
except ImportError:
    raise ImportError("Required library missing. Run: pip install transformers")

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # Removed from some newer Transformers builds.
    AutoModelForVision2Seq = None

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # Older Transformers builds.
    AutoModelForImageTextToText = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:  # Optional; Auto classes may still work.
    Qwen2_5_VLForConditionalGeneration = None

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

warnings.filterwarnings("ignore")

class VisualFeatureExtractor:
    """
    Extracts visual features (captions, labels) using a HF vision-language model.
    """
    
    DEFAULT_MODEL = os.getenv("VISUAL_ENRICHMENT_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        load_in_4bit: bool = False,
        trust_remote_code: bool = True,
        torch_dtype: str = "auto",
    ):
        """
        Initialize visual feature extractor.
        
        Args:
            model_name: HuggingFace model ID
            device: "auto", "cuda", or "cpu"
            load_in_4bit: Whether to use 4-bit quantization (requires bitsandbytes)
            trust_remote_code: Whether to trust remote code from HF
            torch_dtype: "auto", "bf16", "fp16", or "fp32"
        """
        if device == "auto":
            from transcriber_utils import get_device
            self.device = get_device()
        else:
            self.device = device
            
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.torch_dtype_name = (torch_dtype or "auto").strip().lower()
        self._ocr_reader = None
        self._surya_initialized = False
        self._surya_available = False
        self._surya_run_ocr = None
        self._surya_det_model = None
        self._surya_rec_model = None
        self._surya_rec_processor = None
        self._paddle_initialized = False
        self._paddle_available = False
        self._paddle_model = None
        self._paddle_processor = None
        self.enable_ocr_fallback = (
            os.getenv("VISUAL_OCR_FALLBACK", "true").strip().lower()
            in ("1", "true", "yes", "on")
        )
        # For video frames prefer PaddleOCR first, then EasyOCR as fallback.
        # Keep an env override for advanced users.
        self.ocr_fallback_engine = os.getenv(
            "VISUAL_OCR_FALLBACK_ENGINE", "paddle_easyocr"
        ).strip().lower()
        
        print(f"\n{'='*60}")
        print("Visual Feature Extractor")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        print(f"4-bit:  {load_in_4bit}")
        print(f"Dtype:  {self.torch_dtype_name}")
        print(f"{'='*60}\n")
        
        self._load_model(trust_remote_code)

    def _load_model(self, trust_remote_code: bool):
        """Load model + processor for visual inference."""
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
        model_dtype = self._resolve_torch_dtype()
        
        model_kwargs = {
            "torch_dtype": model_dtype,
            "trust_remote_code": trust_remote_code,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            
        try:
            model_cls = self._resolve_model_class()
            self.model = model_cls.from_pretrained(
                self.model_name,
                **model_kwargs
            )
        except Exception as e:
            if self.device == "cuda":
                print(f"  Warning: Loading {self.model_name} on CUDA failed: {e}")
                print("  Retrying on CPU...")
                self.device = "cpu"
                model_kwargs["torch_dtype"] = torch.float32
                if "device_map" in model_kwargs:
                    del model_kwargs["device_map"]
                if "quantization_config" in model_kwargs:
                    del model_kwargs["quantization_config"]
                
                model_cls = self._resolve_model_class()
                self.model = model_cls.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            else:
                raise e
        
        if self.device == "cpu":
            self.model = self.model.to("cpu")
            
        print(f"[OK] {self.model_name} loaded successfully.")

    def _resolve_model_class(self):
        """Prefer model-specific/current multimodal loaders when available."""
        model_lower = self.model_name.lower()
        if "qwen2.5-vl" in model_lower and Qwen2_5_VLForConditionalGeneration is not None:
            return Qwen2_5_VLForConditionalGeneration
        if AutoModelForImageTextToText is not None:
            return AutoModelForImageTextToText
        if AutoModelForVision2Seq is not None:
            return AutoModelForVision2Seq
        raise ImportError(
            "No supported vision-language model class found. "
            "Install a compatible Transformers build, e.g. transformers>=4.57,<5.0."
        )

    def _resolve_torch_dtype(self):
        """Resolve user-configured dtype to a torch dtype."""
        name = self.torch_dtype_name
        alias = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
        name = alias.get(name, name)

        if self.device != "cuda":
            return torch.float32

        if name == "auto":
            bf16_supported = False
            try:
                bf16_supported = (
                    torch.cuda.is_available()
                    and hasattr(torch.cuda, "is_bf16_supported")
                    and torch.cuda.is_bf16_supported()
                )
            except Exception:
                bf16_supported = False
            return torch.bfloat16 if bf16_supported else torch.float16
        if name == "bfloat16":
            return torch.bfloat16
        if name == "float32":
            return torch.float32
        return torch.float16

    def _build_model_inputs(self, image_path: Union[str, Path], prompt: str):
        """Prepare model inputs with Qwen-first path and generic fallback."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_prompt = prompt
        if hasattr(self.processor, "apply_chat_template"):
            try:
                text_prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text_prompt = prompt

        # Best path for Qwen-VL processors.
        if process_vision_info is not None and "qwen" in self.model_name.lower():
            try:
                image_inputs, video_inputs = process_vision_info(messages)
                return self.processor(
                    text=[text_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
            except Exception as exc:
                print(
                    "Warning: qwen_vl_utils preprocessing failed; using generic path "
                    f"instead ({exc})"
                )

        # Generic fallback for other HF VLM checkpoints.
        image = Image.open(str(image_path)).convert("RGB")
        try:
            return self.processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        except Exception:
            return self.processor(
                text=text_prompt,
                images=image,
                return_tensors="pt",
            ).to(self.device)

    def _decode_generated_text(self, inputs, generated_ids) -> str:
        """Decode generation and trim prompt tokens when possible."""
        try:
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            return self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception:
            return self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

    def _ensure_surya_ocr(self) -> bool:
        """Lazy-load Surya OCR for fallback extraction."""
        if self._surya_initialized:
            return self._surya_available

        self._surya_initialized = True
        try:
            from surya.ocr import run_ocr
            from surya.model.detection.model import load_model as load_det_model
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import (
                load_processor as load_rec_processor,
            )

            print("Loading Surya OCR fallback for visual extraction...")
            self._surya_run_ocr = run_ocr
            self._surya_det_model = load_det_model()
            self._surya_rec_model = load_rec_model()
            self._surya_rec_processor = load_rec_processor()
            self._surya_available = True
            print("[OK] Surya OCR fallback ready.")
        except Exception as exc:
            self._surya_available = False
            print(f"Warning: Surya OCR fallback unavailable: {exc}")

        return self._surya_available

    def _fallback_ocr_text_surya(self, image_path: Union[str, Path]) -> Optional[str]:
        """Extract OCR text via Surya."""
        if not self._ensure_surya_ocr():
            return None

        try:
            image = Image.open(str(image_path)).convert("RGB")
            predictions = self._surya_run_ocr(
                [image],
                [["en", "no"]],
                self._surya_det_model,
                self._surya_rec_model,
                self._surya_rec_processor,
            )
        except Exception as exc:
            print(f"Warning: Surya OCR fallback failed for {image_path}: {exc}")
            return None

        if not predictions:
            return None

        result = predictions[0]
        text_lines = getattr(result, "text_lines", None)
        if text_lines is None and isinstance(result, dict):
            text_lines = result.get("text_lines") or result.get("lines")

        if not text_lines:
            return None

        parts = []
        for line in text_lines:
            txt = getattr(line, "text", None)
            if txt is None and isinstance(line, dict):
                txt = line.get("text")
            if txt:
                parts.append(str(txt).strip())

        merged = " ".join(parts).strip()
        if not merged:
            return None
        if merged.lower() in {"none", "null", "n/a", "na", "no text"}:
            return None
        return merged

    def _get_ocr_reader(self):
        """Lazy-load EasyOCR fallback reader."""
        if not self.enable_ocr_fallback:
            return None
        if self._ocr_reader is not None:
            return self._ocr_reader

        try:
            from embeddings.ocr import get_ocr_reader

            use_gpu = self.device == "cuda"
            self._ocr_reader = get_ocr_reader(languages=["en", "no"], use_gpu=use_gpu)
        except Exception as exc:
            print(f"Warning: OCR fallback unavailable: {exc}")
            self._ocr_reader = None

        return self._ocr_reader

    def _ensure_paddle_ocr_vl(self) -> bool:
        """Lazy-load PaddleOCR-VL for diagram/presentation OCR fallback."""
        if self._paddle_initialized:
            return self._paddle_available

        self._paddle_initialized = True
        try:
            try:
                from transformers import PaddleOCRVLForConditionalGeneration
            except ImportError:
                print("Warning: PaddleOCR-VL class unavailable in installed Transformers.")
                self._paddle_available = False
                return False

            model_name = os.getenv("PADDLE_OCR_VL_MODEL", "PaddlePaddle/PaddleOCR-VL")
            dtype = self._resolve_torch_dtype()
            print(f"Loading PaddleOCR-VL fallback: {model_name}")
            self._paddle_processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            self._paddle_model = PaddleOCRVLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            if self.device == "cpu":
                self._paddle_model = self._paddle_model.to("cpu")
            elif self.device == "cuda":
                self._paddle_model = self._paddle_model.to("cuda")
            self._paddle_model.eval()
            self._paddle_available = True
            print("[OK] PaddleOCR-VL fallback ready.")
        except Exception as exc:
            self._paddle_available = False
            print(f"Warning: PaddleOCR-VL fallback unavailable: {exc}")

        return self._paddle_available

    def _fallback_ocr_text_paddle(self, image_path: Union[str, Path]) -> Optional[str]:
        """Extract visible text using PaddleOCR-VL when available."""
        if not self._ensure_paddle_ocr_vl():
            return None

        prompt = (
            "Extract all readable text visible in this image. Preserve reading order. "
            "Return only the text; do not describe the image."
        )
        try:
            image = Image.open(str(image_path)).convert("RGB")
            inputs = self._paddle_processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                k: v.to(self.device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }
            with torch.no_grad():
                generated_ids = self._paddle_model.generate(**inputs, max_new_tokens=512)
            input_ids = inputs.get("input_ids")
            try:
                if input_ids is not None:
                    generated_ids = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(input_ids, generated_ids)
                    ]
                output_text = self._paddle_processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            except Exception:
                output_text = self._paddle_processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
        except Exception as exc:
            print(f"Warning: PaddleOCR-VL OCR failed for {image_path}: {exc}")
            return None

        text = " ".join((output_text or "").split()).strip()
        if not text or text.lower() in {"none", "null", "n/a", "na", "no text"}:
            return None
        return text

    def _fallback_ocr_text(self, image_path: Union[str, Path]) -> Optional[str]:
        """Fallback OCR extraction when the vision model omits OCR text."""
        if not self.enable_ocr_fallback:
            return None

        if self.ocr_fallback_engine in {
            "paddle",
            "paddle_surya",
            "paddle_easyocr",
            "paddle_surya_easyocr",
            "auto",
        }:
            text = self._fallback_ocr_text_paddle(image_path)
            if text:
                return text
            if self.ocr_fallback_engine == "paddle":
                return None

        # Prefer Surya (document OCR) when available, then fall back to EasyOCR.
        if self.ocr_fallback_engine in {
            "surya",
            "surya_easyocr",
            "paddle_surya",
            "paddle_surya_easyocr",
            "auto",
        }:
            text = self._fallback_ocr_text_surya(image_path)
            if text:
                return text
            if self.ocr_fallback_engine == "surya":
                return None

        if self.ocr_fallback_engine in {
            "easyocr",
            "surya_easyocr",
            "paddle_easyocr",
            "paddle_surya_easyocr",
            "auto",
        }:
            reader = self._get_ocr_reader()
            if reader is None:
                return None

            try:
                detections = reader.extract_with_confidence(
                    str(image_path), confidence_threshold=0.35
                )
            except Exception as exc:
                print(f"Warning: OCR fallback failed for {image_path}: {exc}")
                return None

            if not detections:
                return None

            text_parts = []
            for det in detections:
                txt = str(det.get("text", "")).strip()
                if txt:
                    text_parts.append(txt)

            merged = " ".join(text_parts).strip()
            if not merged:
                return None
            if merged.lower() in {"none", "null", "n/a", "na", "no text"}:
                return None
            return merged

        return None

    def analyze_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze an image to get a caption, object labels, and OCR text.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing 'caption', 'object_labels', and 'ocr_text'
        """
        if not os.path.exists(image_path):
            return {"caption": "", "object_labels": [], "ocr_text": ""}
        
        try:
            result = self._run_inference(image_path)
            if not result.get("caption"):
                fallback_caption = self._run_caption_only(image_path)
                if fallback_caption:
                    result["caption"] = fallback_caption
            if not result.get("ocr_text"):
                ocr_fallback = self._fallback_ocr_text(image_path)
                if ocr_fallback:
                    result["ocr_text"] = ocr_fallback
            return result
        except Exception as e:
            print(f"  Warning: analyze_image failed for {image_path}: {e}")
            ocr_fallback = self._fallback_ocr_text(image_path)
            return {
                "caption": None,
                "object_labels": [],
                "ocr_text": ocr_fallback,
            }

    def _run_inference(self, image_path) -> dict:
        """Internal: run the model on a single image."""
        query = (
            "Analyze this video frame for retrieval indexing. Return valid JSON only with keys: "
            "caption, object_labels, ocr_text.\n"
            "- caption: one concise sentence describing the visible scene.\n"
            "- object_labels: 8 to 20 lowercase searchable tags for visible subjects, objects, "
            "scene type, diagram/chart type, and visual style. Include specific labels when visually "
            "supported, for example: samurai, sword, armor, cherry blossom, japanese art, subtitle, "
            "flow diagram, process diagram, pump, valve, platform, timeline.\n"
            "- ocr_text: all readable on-screen text in reading order, or an empty string.\n"
            "Do not include explanations outside JSON."
        )

        inputs = self._build_model_inputs(image_path=image_path, prompt=query)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)

        output_text = self._decode_generated_text(inputs, generated_ids)
        
        # Parse output_text (heuristic parsing)
        return self._parse_output(output_text)

    def _run_caption_only(self, image_path) -> Optional[str]:
        """Fallback caption pass when the structured response has no caption."""
        prompt = (
            "Describe what is visible in this video frame in one concise sentence. "
            "Do not return JSON or numbered lists."
        )
        inputs = self._build_model_inputs(image_path=image_path, prompt=prompt)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        output_text = self._decode_generated_text(inputs, generated_ids)

        caption = output_text.strip()
        if not caption:
            return None
        caption = " ".join(caption.split())
        if caption.lower() in {"none", "null", "n/a", "na"}:
            return None
        return caption[:700]

    def _parse_output(self, text: str) -> Dict[str, Any]:
        """
        Regex-anchored parser for the model's numbered-list output.

        Expected model format:
            1. <scene description>
            2. <comma-separated object tags>
            3. <OCR text, or "None">

        Using re.match anchored to line-start prevents false triggers on
        content that happens to contain "2." (e.g. "25th year stand-up").
        """
        # JSON fallback first (some checkpoints answer in JSON-like format).
        stripped = (text or "").strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)
        if not (stripped.startswith("{") and stripped.endswith("}")):
            match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
            if match:
                stripped = match.group(0)
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                payload = json.loads(stripped)
                caption = payload.get("caption") or payload.get("description")
                labels = payload.get("object_labels") or payload.get("labels") or []
                ocr_text = payload.get("ocr_text") or payload.get("ocr") or None
                if isinstance(labels, str):
                    labels = re.split(r"[,;\n]+", labels)
                    labels = [v.strip() for v in labels if v.strip()]
                elif not isinstance(labels, list):
                    labels = [str(labels)] if labels else []
                labels = self._clean_labels(labels)
                return {
                    "caption": caption.strip() if isinstance(caption, str) else caption,
                    "object_labels": labels,
                    "ocr_text": ocr_text.strip() if isinstance(ocr_text, str) else ocr_text,
                }
            except Exception:
                pass

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
                    object_labels = self._clean_labels(
                        [t.strip() for t in content.split(",") if t.strip()]
                    )
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
                        object_labels = self._clean_labels(
                            [t.strip() for t in content.split(",") if t.strip()]
                        )
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

        # Last-resort fallback: if parsing failed completely, keep a concise
        # description so retrieval still has a semantic text signal.
        if not caption and not object_labels and not ocr_text:
            fallback = re.sub(r"\s+", " ", stripped).strip()
            fallback = re.sub(r"^[\-\*\d\.\)\s:]+", "", fallback).strip()
            if fallback:
                caption = fallback[:700]

        object_labels = self._clean_labels(object_labels)
        return {
            "caption":       caption or None,
            "object_labels": object_labels,
            "ocr_text":      ocr_text or None,
        }

    @staticmethod
    def _clean_labels(labels: List[Any]) -> List[str]:
        """Normalize object/style tags for stable indexing."""
        invalid = {"none", "null", "n/a", "na", "no objects", "unknown", ""}
        out: List[str] = []
        for raw in labels or []:
            label = str(raw).strip().lower()
            label = re.sub(r"^[\-\*\d\.\)\s:]+", "", label).strip()
            label = re.sub(r"\s+", " ", label)
            label = label.strip(" .,;:/|")
            if not label or label in invalid:
                continue
            # Keep labels compact and search-friendly.
            if len(label) > 80:
                label = label[:80].rstrip()
            out.append(label)
        return list(dict.fromkeys(out))

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
