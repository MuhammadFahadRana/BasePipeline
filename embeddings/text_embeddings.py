"""Generate embeddings for text using sentence-transformers."""

import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np


class EmbeddingGenerator:
    """Generate text embeddings for semantic search."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "auto"):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model name (default: Qwen/Qwen3-Embedding-0.6B)
            device: "auto", "cpu", or "cuda"
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name

        print(f"Loading embedding model: {model_name}")
        print(f"Device: {device}")

        # Note: Qwen3-Embedding models are compatible with SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        print(f"✓ Embedding model loaded (dim={self.embedding_dim})")

    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32, 
        show_progress: bool = False,
        instruction: str = ""
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


# Global instance (lazy loaded)
_embedding_generator = None


def get_embedding_generator(model_name: str = "Qwen/Qwen3-Embedding-0.6B") -> EmbeddingGenerator:
    """Get or create global embedding generator instance."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator(model_name=model_name)
    return _embedding_generator
