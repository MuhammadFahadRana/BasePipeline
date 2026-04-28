"""Generate vision embeddings for keyframes using SigLIP 2."""

import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from typing import List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

FALLBACK_VISION_EMBEDDING_MODEL = "google/siglip2-so400m-patch14-384"


def get_default_vision_embedding_model() -> str:
    return os.getenv("VISION_EMBEDDING_MODEL", FALLBACK_VISION_EMBEDDING_MODEL)


DEFAULT_VISION_EMBEDDING_MODEL = get_default_vision_embedding_model()
LEGACY_VISION_EMBEDDING_MODELS = ("google/siglip-base-patch16-224",)


class VisionEmbeddingGenerator:
    """Generate image/text-aligned vision embeddings using SigLIP-style encoders."""

    def __init__(
        self, model_name: Optional[str] = None, device: str = "auto"
    ):
        """
        Initialize the vision embedding model.

        Args:
            model_name: HuggingFace model name. Defaults to SigLIP 2.
            device: "auto", "cpu", or "cuda"
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = model_name or get_default_vision_embedding_model()
        self.device = device
        self.model_name = model_name

        device_tag = "[GPU]" if "cuda" in self.device else "[CPU]"
        print(f"{device_tag} Loading vision embedding model: {model_name} on {self.device}")

        try:
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.processor = AutoProcessor.from_pretrained(model_name)
        except Exception as exc:
            exc_text = str(exc).lower()
            if "cuda" in device and ("out of memory" in exc_text or "memoryallocation" in exc_text):
                print(f"[WARN] CUDA OOM while loading {model_name}; retrying on CPU.")
                self.device = "cpu"
                self.model = AutoModel.from_pretrained(model_name).to("cpu")
                self.processor = AutoProcessor.from_pretrained(model_name)
            else:
                raise

        # Get embedding dimension
        self.embedding_dim = self.model.config.vision_config.hidden_size

        # Avoid Unicode symbols to keep Windows cp1252 consoles happy.
        print(f"[OK] Vision embedding model loaded (dim={self.embedding_dim})")

    def encode_image(
        self, image_input: Union[str, Path, bytes, Image.Image], normalize: bool = True
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
                raise ValueError(
                    "image_input must be a path, bytes, or PIL Image object"
                )
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embedding
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # get_image_features can return a Tensor or a model output object
        if not isinstance(image_features, torch.Tensor):
            feats = getattr(image_features, "image_embeds", None)
            if feats is None:
                feats = getattr(image_features, "pooler_output", None)
            if feats is None:
                last_hidden = getattr(image_features, "last_hidden_state", None)
                if last_hidden is not None:
                    feats = last_hidden[:, 0, :]
            if feats is None:
                raise TypeError(
                    f"Unexpected get_image_features output type: {type(image_features)}"
                )
            image_features = feats

        # Convert to numpy
        embedding = image_features.detach().float().cpu().numpy()[0]

        # Normalize if requested
        if normalize:
            embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def encode_images(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
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
            batch_paths = image_paths[i : i + batch_size]

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

            # get_image_features can return a Tensor or a model output object
            if not isinstance(image_features, torch.Tensor):
                feats = getattr(image_features, "image_embeds", None)
                if feats is None:
                    feats = getattr(image_features, "pooler_output", None)
                if feats is None:
                    last_hidden = getattr(image_features, "last_hidden_state", None)
                    if last_hidden is not None:
                        feats = last_hidden[:, 0, :]
                if feats is None:
                    raise TypeError(
                        f"Unexpected get_image_features output type: {type(image_features)}"
                    )
                image_features = feats

            # Convert to numpy
            batch_embeddings = image_features.detach().float().cpu().numpy()

            # Normalize if requested
            if normalize:
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                batch_embeddings = batch_embeddings / norms

            embeddings.append(batch_embeddings)

        if not embeddings:
            return np.array([])

        return np.vstack(embeddings)

    def encode_text(
        self, text: Union[str, List[str]], normalize: bool = True
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
        # `get_text_features` can return either a Tensor or a model output object
        # depending on Transformers version/model class. Normalize to a Tensor.
        if isinstance(text_features, torch.Tensor):
            feats = text_features
        else:
            # Common fields across CLIP/SigLIP-like models
            feats = getattr(text_features, "text_embeds", None)
            if feats is None:
                feats = getattr(text_features, "pooler_output", None)
            if feats is None:
                last_hidden = getattr(text_features, "last_hidden_state", None)
                if last_hidden is not None:
                    feats = last_hidden[:, 0, :]
            if feats is None:
                raise TypeError(
                    f"Unexpected get_text_features output type: {type(text_features)}"
                )

        embedding = feats.detach().float().cpu().numpy()

        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            embedding = embedding / norms

        if len(text) == 1:
            return embedding[0]
        return embedding


# Global instances (lazy loaded, keyed by model/device)
_vision_generators = {}


def get_vision_embedding_generator(
    model_name: Optional[str] = None,
    device: str = "auto",
) -> VisionEmbeddingGenerator:
    """Get or create a cached vision embedding generator."""
    resolved_model_name = model_name or get_default_vision_embedding_model()
    cache_key: Tuple[str, str] = (resolved_model_name, device)

    if cache_key not in _vision_generators:
        _vision_generators[cache_key] = VisionEmbeddingGenerator(
            model_name=resolved_model_name,
            device=device,
        )

    return _vision_generators[cache_key]
