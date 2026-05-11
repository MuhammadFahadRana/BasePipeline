"""
BasePipeline package for video processing. 
This package is a collection of tools for processing videos.
It includes tools for scene detection, transcription, and embedding.
"""

from .scene_detector import SceneDetector
from .transcriber import SimpleTranscriber
from .basic_pipeline import BasicVideoPipeline

Transcriber = SimpleTranscriber

__all__ = [
    "SceneDetector",
    "SimpleTranscriber",
    "Transcriber",
    "BasicVideoPipeline",
]
