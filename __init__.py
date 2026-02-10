"""
BasePipeline package for video processing. 
This package is a collection of tools for processing videos.
It includes tools for scene detection, transcription, and embedding.
"""

from .scene_detector import SceneDetector
from .transcriber import Transcriber
from .basic_pipeline import BasicVideoPipeline
from semantic_refiner import SemanticSceneRefiner, RefinerConfig
# from .embedder import Embedder

__all__ = [
    "SceneDetector",
    "Transcriber",
    "BasicVideoPipeline",
    # "Embedder",
]
