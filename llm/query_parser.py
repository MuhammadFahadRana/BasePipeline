"""
LLM-based query parser using Qwen2.5-1.5B-Instruct.

Extracts structured intent, keywords, and search targets from natural language queries.
"""

import json
import time
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ParsedQuery:
    original_query: str
    normalized_query: str
    intent: str
    keywords: List[str]
    must_have: List[str]
    targets: List[str]
    time_hint: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "intent": self.intent,
            "keywords": self.keywords,
            "must_have": self.must_have,
            "targets": self.targets,
            "time_hint": self.time_hint,
        }


class QueryParser:
    """Parses natural language queries into structured search intent."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        max_cache_size: int = 1000,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name
        self.max_cache_size = max_cache_size

        print(f"Loading Query Parser model: {model_name}")
        print(f"Device: {device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self.model.eval()
            print(f"[OK] Query Parser loaded ({model_name})")
            self._loaded = True
        except Exception as e:
            print(f"[ERROR] Failed to load Query Parser: {e}")
            self._loaded = False

        # In-memory cache for parsed queries
        self._cache = {}

    def parse(self, query: str) -> ParsedQuery:
        """Parse a user query into structured intent."""
        # Check cache
        if query in self._cache:
            return self._cache[query]

        if not self._loaded:
            # Fallback for when model isn't loaded: basic keyword extraction
            return self._fallback_parse(query)

        start_time = time.time()
        
        # Prepare prompt
        prompt = self._build_prompt(query)
        
        # Generate
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,  # Low temp for deterministic JSON
                    do_sample=False,
                    stop_strings=["```", "\n\n", "}"],
                    tokenizer=self.tokenizer
                )
            
            output_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            parsed_json = self._extract_json(output_text)
            
            result = ParsedQuery(
                original_query=query,
                normalized_query=parsed_json.get("normalized_query", query),
                intent=parsed_json.get("intent", "find_specific"),
                keywords=parsed_json.get("keywords", query.split()),
                must_have=parsed_json.get("must_have", []),
                targets=parsed_json.get("targets", ["transcript", "ocr", "vision"]), # Default to all
                time_hint=parsed_json.get("time_hint")
            )
            
            # Cache result
            self._cache_result(query, result)
            
            elapsed = (time.time() - start_time) * 1000
            print(f"  Query Parsed: {query} -> {result.targets} ({elapsed:.0f}ms)")
            
            return result
            
        except Exception as e:
            print(f"[WARNING] LLM parsing failed: {e}")
            return self._fallback_parse(query)

    def _build_prompt(self, query: str) -> str:
        return f"""You are a smart search query parser. Extract intent and targets.
Valid targets: ["transcript" (speech), "ocr" (text on screen), "vision" (visual content)].

Examples:
Q: "show me the slide about Yggdrasil"
JSON: {{
  "normalized_query": "Yggdrasil slide",
  "intent": "find_specific",
  "keywords": ["Yggdrasil", "slide"],
  "must_have": ["Yggdrasil"],
  "targets": ["ocr", "vision"],
  "time_hint": null
}}

Q: "what did they say about safety?"
JSON: {{
  "normalized_query": "safety discussion",
  "intent": "question",
  "keywords": ["safety"],
  "must_have": ["safety"],
  "targets": ["transcript"],
  "time_hint": null
}}

Q: "orange robot underwater"
JSON: {{
  "normalized_query": "orange robot underwater",
  "intent": "find_specific",
  "keywords": ["orange", "robot", "underwater"],
  "must_have": ["robot"],
  "targets": ["vision"],
  "time_hint": null
}}

Q: "{query}"
JSON: """

    def _extract_json(self, text: str) -> Dict:
        """Robustly extract JSON from model output."""
        try:
            # Try finding first { and last }
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                json_str = text[start : end + 1]
                # Ensure it ends with } if not present (due to stop token)
                if not json_str.endswith("}"):
                   json_str += "}"
                return json.loads(json_str)
            
            # If no opening brace but we have content (rare with stop token)
            # Maybe the model just outputted "}" at the end?
            # With stop token "}", the output might exclude "}". 
            # Transformers semantics for stop_strings: stops generation, but does it include the string?
            # It usually includes it in the generated tokens if it was generated.
            return {}
        except:
            return {}

    def _fallback_parse(self, query: str) -> ParsedQuery:
        """Basic fallback when LLM is unavailable."""
        return ParsedQuery(
            original_query=query,
            normalized_query=query,
            intent="find_specific",
            keywords=query.split(),
            must_have=[],
            targets=["transcript", "ocr", "vision"],
            time_hint=None
        )

    def _cache_result(self, query: str, result: ParsedQuery):
        """Cache result with LRU eviction."""
        if len(self._cache) >= self.max_cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[query] = result


# ── Global singleton ──────────────────────────────

_query_parser: Optional[QueryParser] = None

def get_query_parser(enabled: bool = True) -> Optional[QueryParser]:
    global _query_parser
    if not enabled:
        return None
        
    if _query_parser is None:
        _query_parser = QueryParser()
        
    return _query_parser
