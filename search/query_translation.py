"""Small, bounded query translation helpers for multilingual retrieval."""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set, Tuple


def normalize_lang_code(lang: Optional[str]) -> Optional[str]:
    """Normalize language identifiers used by Whisper, DB rows, and UI hints."""
    if not lang:
        return None
    code = str(lang).strip().lower()
    if not code:
        return None
    if code.startswith("no") or code in {
        "nb",
        "nn",
        "nor",
        "nob",
        "nno",
        "norwegian",
        "bokmal",
        "bokm\u00e5l",
        "nynorsk",
    }:
        return "no"
    if code.startswith("en") or code in {"eng", "english"}:
        return "en"
    return code


def detect_query_language(query: str) -> str:
    """Cheap EN/NO detector; enough to decide the opposite query expansion."""
    text = (query or "").lower()
    if re.search(r"[\u00e6\u00f8\u00e5]", text):
        return "no"

    norwegian_markers = {
        "hva",
        "hvem",
        "hvor",
        "hvordan",
        "hvorfor",
        "er",
        "nar",
        "n\u00e5r",
        "og",
        "pa",
        "p\u00e5",
        "som",
        "til",
        "fra",
        "med",
        "det",
        "den",
        "de",
        "en",
        "et",
        "av",
        "for",
        "om",
        "kan",
        "har",
        "var",
        "vil",
        "skal",
        "ikke",
        "eller",
        "men",
        "ogsa",
        "ogs\u00e5",
        "denne",
        "dette",
        "bronn",
        "br\u00f8nn",
    }
    words = set(re.findall(r"[\w]+", text, flags=re.UNICODE))
    return "no" if len(words & norwegian_markers) >= 2 else "en"


def normalize_language_set(languages: Optional[Iterable[str]]) -> Set[str]:
    normalized: Set[str] = set()
    for lang in languages or []:
        code = normalize_lang_code(lang)
        if code:
            normalized.add(code)
    return normalized


@lru_cache(maxsize=16)
def _load_marian_pair(source: str, target: str):
    """Lazy-load a small local Marian model for an EN/NO pair."""
    from transformers import MarianMTModel, MarianTokenizer

    model_map = {
        ("en", "no"): os.getenv(
            "SEARCH_TRANSLATION_MODEL_EN_NO", "Helsinki-NLP/opus-mt-en-gmq"
        ),
        ("no", "en"): os.getenv(
            "SEARCH_TRANSLATION_MODEL_NO_EN", "Helsinki-NLP/opus-mt-gmq-en"
        ),
    }
    model_name = model_map.get((source, target))
    if not model_name:
        raise ValueError(f"No Marian translation model configured for {source}->{target}")

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


@lru_cache(maxsize=4)
def _load_nllb_model():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = os.getenv(
        "SEARCH_TRANSLATION_NLLB_MODEL", "facebook/nllb-200-distilled-600M"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


class QueryTranslator:
    """Query translator with hard limits so search latency stays predictable."""

    def __init__(
        self,
        enabled: bool = True,
        provider: str = "mymemory",
        targets: Tuple[str, ...] = ("en", "no"),
        timeout_seconds: float = 0.75,
        max_chars: int = 500,
        max_variants: int = 2,
    ) -> None:
        self.enabled = enabled
        self.provider = (provider or "none").strip().lower()
        self.targets = tuple(
            code
            for code in (normalize_lang_code(t) for t in targets)
            if code is not None
        )
        self.timeout_seconds = max(0.05, float(timeout_seconds))
        self.max_chars = max(50, int(max_chars))
        self.max_variants = max(1, int(max_variants))
        self._cache: Dict[Tuple[str, str, str, str], Optional[str]] = {}

    @classmethod
    def from_env(cls) -> "QueryTranslator":
        enabled = os.getenv("SEARCH_QUERY_TRANSLATION_ENABLED", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
            "y",
        }
        provider = os.getenv("SEARCH_QUERY_TRANSLATION_PROVIDER", "mymemory")
        targets = tuple(
            item.strip()
            for item in os.getenv("SEARCH_QUERY_TRANSLATION_TARGETS", "en,no").split(",")
            if item.strip()
        )
        timeout = float(os.getenv("SEARCH_QUERY_TRANSLATION_TIMEOUT", "0.75"))
        max_chars = int(os.getenv("SEARCH_QUERY_TRANSLATION_MAX_CHARS", "500"))
        max_variants = int(os.getenv("SEARCH_QUERY_TRANSLATION_MAX_VARIANTS", "2"))
        return cls(
            enabled=enabled,
            provider=provider,
            targets=targets,
            timeout_seconds=timeout,
            max_chars=max_chars,
            max_variants=max_variants,
        )

    @property
    def cache_key_settings(self) -> Dict[str, object]:
        return {
            "translation_enabled": self.enabled,
            "translation_provider": self.provider,
            "translation_targets": self.targets,
            "translation_max_variants": self.max_variants,
            "multilingual_vector_search": True,
        }

    def translate(self, query: str, source_lang: str, target_lang: str) -> Optional[str]:
        source = normalize_lang_code(source_lang)
        target = normalize_lang_code(target_lang)
        clean_query = (query or "").strip()
        if (
            not self.enabled
            or not clean_query
            or not source
            or not target
            or source == target
            or self.provider in {"none", "off", "disabled"}
        ):
            return None

        cache_key = (self.provider, clean_query.lower(), source, target)
        if cache_key in self._cache:
            return self._cache[cache_key]

        translated: Optional[str] = None
        try:
            if self.provider in {"mymemory", "api"}:
                translated = self._translate_mymemory(clean_query, source, target)
            elif self.provider in {"marian", "helsinki", "local"}:
                translated = self._translate_marian(clean_query, source, target)
            elif self.provider in {"nllb", "nllb200"}:
                translated = self._translate_nllb(clean_query, source, target)
        except Exception:
            translated = None

        if translated and translated.strip().lower() == clean_query.lower():
            translated = None

        self._cache[cache_key] = translated
        return translated

    def build_variants(
        self,
        query: str,
        available_languages: Optional[Iterable[str]] = None,
    ) -> List[str]:
        base = (query or "").strip()
        if not base:
            return []

        variants: List[str] = [base]
        seen = {base.lower()}
        if not self.enabled or self.max_variants <= 1:
            return variants

        source_lang = detect_query_language(base)
        available = normalize_language_set(available_languages)
        targets = list(self.targets)
        if available:
            targets = [target for target in targets if target in available]

        for target in targets:
            if len(variants) >= self.max_variants:
                break
            if target == source_lang:
                continue
            translated = self.translate(base, source_lang, target)
            if not translated:
                continue
            key = translated.lower()
            if key in seen:
                continue
            variants.append(translated)
            seen.add(key)

        return variants

    def _translate_mymemory(self, query: str, source: str, target: str) -> Optional[str]:
        lang_pair = f"{source}|{target}"
        url = (
            "https://api.mymemory.translated.net/get?"
            f"q={urllib.parse.quote(query[: self.max_chars])}&langpair={lang_pair}"
        )
        with urllib.request.urlopen(url, timeout=self.timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        candidate = payload.get("responseData", {}).get("translatedText", "")
        candidate = str(candidate or "").strip()
        return candidate or None

    def _translate_marian(self, query: str, source: str, target: str) -> Optional[str]:
        tokenizer, model = _load_marian_pair(source, target)
        text = query[: self.max_chars]
        if source == "en" and target == "no":
            token = os.getenv("SEARCH_TRANSLATION_MARIAN_NO_TOKEN", ">>nob<<")
            text = f"{token} {text}"
        batch = tokenizer([text], return_tensors="pt", truncation=True)
        generated = model.generate(**batch, max_new_tokens=96)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def _translate_nllb(self, query: str, source: str, target: str) -> Optional[str]:
        tokenizer, model = _load_nllb_model()
        lang_map = {
            "en": "eng_Latn",
            "no": os.getenv("SEARCH_TRANSLATION_NLLB_NO_CODE", "nob_Latn"),
        }
        src_code = lang_map.get(source)
        tgt_code = lang_map.get(target)
        if not src_code or not tgt_code:
            return None

        tokenizer.src_lang = src_code
        batch = tokenizer([query[: self.max_chars]], return_tensors="pt", truncation=True)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
        generated = model.generate(
            **batch,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=96,
        )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
