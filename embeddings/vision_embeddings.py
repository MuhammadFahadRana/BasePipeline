"""Generate vision embeddings for keyframes using SigLIP (SOTA CLIP alternative)."""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from typing import List, Union
import numpy as np
from pathlib import Path


class VisionEmbeddingGenerator:
    """Generate vision embeddings for images using SigLIP."""

    def __init__(self, model_name: str = "google/siglip-base-patch16-224", device: str = "auto"):
        """
        Initialize SigLIP vision model.

        Args:
            model_name: HuggingFace model name (default: google/siglip-base-patch16-224)
            SigLIP2:google/siglip2-base-patch16-224
            device: "auto", "cpu", or "cuda"
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name

        print(f"Loading SigLIP model: {model_name}")
        print(f"Device: {device}")

        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.vision_config.hidden_size

        print(f"✓ SigLIP model loaded (dim={self.embedding_dim})")

    def encode_image(
        self, 
        image_input: Union[str, Path, bytes, Image.Image],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for a single image.

        Args:
            image_input: Path to image file, image bytes, or PIL Image object
            normalize: Whether to L2-normalize the embedding

        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        try:
            if isinstance(image_input, (str, Path)):
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, bytes):
                import io
                image = Image.open(io.BytesIO(image_input)).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                raise ValueError("image_input must be a path, bytes, or PIL Image object")
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embedding
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        # Convert to numpy
        embedding = image_features.cpu().numpy()[0]

        # Normalize if requested
        if normalize:
            embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def encode_images(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple images (batched).

        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process at once
            show_progress: Show progress bar
            normalize: Whether to L2-normalize embeddings

        Returns:
            2D numpy array of shape (num_images, embedding_dim)
        """
        from tqdm import tqdm

        embeddings = []
        
        iterator = range(0, len(image_paths), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding images")

        for i in iterator:
            batch_paths = image_paths[i:i+batch_size]
            
            # Load batch of images
            try:
                images = [Image.open(p).convert("RGB") for p in batch_paths]
            except Exception as e:
                print(f"Warning: Failed to load image in batch starting at {i}: {e}")
                continue

            # Preprocess batch
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            # Convert to numpy
            batch_embeddings = image_features.cpu().numpy()
            
            # Normalize if requested
            if normalize:
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                batch_embeddings = batch_embeddings / norms

            embeddings.append(batch_embeddings)

        if not embeddings:
            return np.array([])

        return np.vstack(embeddings)

    def encode_text(
        self,
        text: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for text query (for image-text matching).

        Args:
            text: Single text string or list of texts
            normalize: Whether to L2-normalize the embedding

        Returns:
            numpy array of embeddings
        """
        if isinstance(text, str):
            text = [text]

        # Preprocess text
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embedding
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        # Convert to numpy
        embedding = text_features.cpu().numpy()

        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            embedding = embedding / norms

        if len(text) == 1:
            return embedding[0]
        return embedding


# Global instance (lazy loaded)
_vision_generator = None


def get_vision_embedding_generator(model_name: str = "google/siglip-base-patch16-224") -> VisionEmbeddingGenerator:
    """Get or create global vision embedding generator instance."""
    global _vision_generator
    if _vision_generator is None:
        _vision_generator = VisionEmbeddingGenerator(model_name=model_name)
    return _vision_generator
