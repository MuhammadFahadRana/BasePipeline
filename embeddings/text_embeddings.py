"""Generate embeddings for text using sentence-transformers."""

import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

FALLBACK_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def get_default_embedding_model() -> str:
    """Resolve the text embedding model from the current environment."""
    return os.getenv(
        "TEXT_EMBEDDING_MODEL",
        os.getenv("EMBEDDING_MODEL", FALLBACK_EMBEDDING_MODEL),
    )


DEFAULT_EMBEDDING_MODEL = get_default_embedding_model()


class EmbeddingGenerator:
    """Generate text embeddings for semantic search."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model name
            device: "auto", "cpu", or "cuda"
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = model_name or get_default_embedding_model()
        self.device = device
        self.model_name = model_name

        device_tag = "[GPU]" if "cuda" in device else "[CPU]"
        print(f"{device_tag} Loading text embedding model: {model_name} on {device}")

        # Note: Qwen3-Embedding models are compatible with SentenceTransformer.
        # Fall back to CPU if CUDA cannot reserve enough memory for the model.
        try:
            self.model = SentenceTransformer(
                model_name, device=device, trust_remote_code=True
            )
        except Exception as exc:
            exc_text = str(exc).lower()
            if "cuda" in device and ("out of memory" in exc_text or "memoryallocation" in exc_text):
                print(
                    f"[WARN] CUDA OOM while loading {model_name}; retrying on CPU."
                )
                self.device = "cpu"
                self.model = SentenceTransformer(
                    model_name, device="cpu", trust_remote_code=True
                )
            else:
                raise
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Avoid Unicode symbols to keep Windows cp1252 consoles happy.
        print(f"[OK] Embedding model loaded (dim={self.embedding_dim})")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        instruction: str = "",
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            instruction: Instruction prefix (e.g., 'Represent this sentence for searching relevant passages: ')

        Returns:
            numpy array of embeddings (shape: [N, embedding_dim])
        """
        if isinstance(texts, str):
            texts = [texts]

        # Add instruction prefix if provided (recommended for Qwen3-Embedding)
        if instruction:
            texts = [f"{instruction}{t}" for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalization for cosine similarity
        )

        return embeddings

    def encode_single(self, text: str, instruction: str = "") -> np.ndarray:
        """
        Encode a single text (convenience method).

        Args:
            text: Text to encode
            instruction: Optional instruction prefix

        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        return self.encode(text, instruction=instruction)[0]


# Global instances (lazy loaded, keyed by model/device)
_embedding_generators = {}


def get_embedding_generator(
    model_name: Optional[str] = None,
    device: str = "auto",
) -> EmbeddingGenerator:
    """Get or create embedding generator instance for the given model/device."""
    resolved_model_name = model_name or get_default_embedding_model()
    cache_key: Tuple[str, str] = (resolved_model_name, device)

    if cache_key not in _embedding_generators:
        _embedding_generators[cache_key] = EmbeddingGenerator(
            model_name=resolved_model_name,
            device=device,
        )

    return _embedding_generators[cache_key]
