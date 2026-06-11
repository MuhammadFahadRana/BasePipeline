"""LLM enrichment for document pages."""

import os
from typing import Optional


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _normalize_device(value: str) -> str:
    device = (value or "auto").strip().lower()
    return device if device in {"auto", "cuda", "cpu"} else "auto"


class DocumentEnricher:
    """Enriches document pages using a configurable vision-language model."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        load_in_4bit: Optional[bool] = None,
        device: str = "auto",
        trust_remote_code: Optional[bool] = None,
        torch_dtype: str = "fp32",
    ):
        self.enabled = _env_bool("VISUAL_ENRICHMENT_ENABLED", True)
        self.model_name = model_name or os.getenv(
            "VISUAL_ENRICHMENT_MODEL", "Qwen/Qwen2-VL-7B-Instruct"
        )
        if load_in_4bit is None:
            load_in_4bit = _env_bool("VISUAL_ENRICHMENT_LOAD_IN_4BIT", False)
        self.load_in_4bit = bool(load_in_4bit)
        self.device = _normalize_device(
            os.getenv("VISUAL_ENRICHMENT_DEVICE", device)
        )
        if trust_remote_code is None:
            trust_remote_code = _env_bool("VISUAL_ENRICHMENT_TRUST_REMOTE_CODE", True)
        self.trust_remote_code = bool(trust_remote_code)
        self.torch_dtype = (
            os.getenv("VISUAL_ENRICHMENT_DTYPE", torch_dtype).strip().lower()
        )
        self._extractor = None

    def _ensure_extractor(self) -> bool:
        if not self.enabled:
            return False
        if self._extractor is not None:
            return True

        try:
            from extract_visual_features import VisualFeatureExtractor
            from transcriber_utils import get_device

            resolved_device = get_device() if self.device == "auto" else self.device
            use_4bit = self.load_in_4bit and resolved_device == "cuda"

            print("Loading visual enricher for document processing...")
            print(f"  Model: {self.model_name}")
            print(f"  Device: {resolved_device}")
            print(f"  4-bit: {use_4bit}")
            print(f"  Dtype: {self.torch_dtype}")

            self._extractor = VisualFeatureExtractor(
                model_name=self.model_name,
                device=resolved_device,
                load_in_4bit=use_4bit,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=self.torch_dtype,
            )
            return True
        except Exception as e:
            print(
                "Warning: Document LLM enrichment disabled. "
                f"Failed to load VisualFeatureExtractor: {e}"
            )
            self.enabled = False
            return False

    def enrich_page(self, image_path: str) -> Optional[str]:
        """Analyze a page image and return a summary caption."""
        if not self._ensure_extractor():
            return None

        try:
            result = self._extractor.analyze_image(image_path)
            return result.get("caption")
        except Exception as e:
            print(f"Enrichment failed for {image_path}: {e}")
            return None
