import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Any, List, Dict
import threading
import re

class LLMManager:
    """
    Singleton manager for loading and sharing the core reasoning model (Qwen2.5-1.5B-Instruct).
    Ensures that heavy weights are only loaded into VRAM/RAM once across all components.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = False

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def load_model(self):
        """Load the model if it hasn't been loaded yet."""
        if self._loaded:
            return

        device_tag = "[GPU]" if "cuda" in self.device else "[CPU]"
        print(f"{device_tag} Loading shared model: {self.model_name} on {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Ensure pad_token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }

            if self.device == "cuda":
                # Use device_map="auto" to handle multi-GPU or memory pressure gracefully
                model_kwargs["device_map"] = "auto"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            self.model.eval()
            self._loaded = True
            print(f"[LLMManager] Model loaded successfully.")
        except Exception as e:
            print(f"[LLMManager] Error loading model: {e}")
            # Fallback to CPU if CUDA fails (e.g., paging file too small)
            if self.device == "cuda":
                print("[LLMManager] Retrying on CPU fallback...")
                self.device = "cpu"
                model_kwargs["torch_dtype"] = torch.float32
                if "device_map" in model_kwargs:
                    del model_kwargs["device_map"]
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                self.model.eval()
                self._loaded = True
            else:
                raise e

    def get_model_and_tokenizer(self):
        if not self._loaded:
            self.load_model()
        return self.model, self.tokenizer

    def score_relevance(self, query: str, context: str) -> float:
        """
        Use the 1.5B model to score the relevance of a context segment to a query.
        Returns a float between 0.0 and 1.0.
        """
        if not self._loaded:
            self.load_model()

        # Direct prompt for relevance judgement
        prompt = f"""Task: Rate the semantic relevance of the DOCUMENT to the QUERY.
QUERY: "{query}"
DOCUMENT: "{context}"

Return ONLY a single number between 0 and 100, where 100 is exact match/highly relevant and 0 is completely unrelated.
SCORE:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Extract number
        match = re.search(r"(\d+)", response)
        if match:
            score = float(match.group(1)) / 100.0
            return max(0.0, min(1.0, score))
        return 0.5  # Neutral fallback

    def rerank(self, query: str, results: List[Any], top_n: int = 10) -> List[Any]:
        """
        Rerank a list of search results using the LLM.
        """
        if not results:
            return results
        
        print(f"[LLMManager] Reranking top {min(len(results), top_n)} candidates for: {query}")
        
        # Only rerank the top candidates to save time
        candidates = results[:top_n]
        remainder = results[top_n:]
        
        for res in candidates:
            # We use the raw text and the query
            text = getattr(res, "text", "")
            if not text:
                continue
                
            llm_score = self.score_relevance(query, text)
            
            # Blend with original score (70% LLM, 30% original)
            # This ensures that even if LLM is slightly off, the vector similarity still counts
            res.score = (0.7 * llm_score) + (0.3 * res.score)
            res.match_type = f"smart_{res.match_type}"

        # Sort again
        reranked = candidates + remainder
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked

def get_shared_llm():
    """Helper to get the singleton model components."""
    return LLMManager.get_instance().get_model_and_tokenizer()

def get_llm_manager():
    """Helper to get the manager instance."""
    return LLMManager.get_instance()

