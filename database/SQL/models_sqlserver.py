"""SQLAlchemy ORM models for SQL Server Express."""

import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator

Base = declarative_base()

class JSONEncodedDict(TypeDecorator):
    """Represents an immutable structure as a json-encoded string."""
    impl = Text()

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return None

class Video(Base):
    """Video metadata table."""
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False, unique=True, index=True)
    file_path = Column(Text, nullable=False)
    file_size_mb = Column(Float)
    duration_seconds = Column(Float)
    whisper_model = Column(String(50))
    scene_threshold = Column(Float)
    processed_at = Column(DateTime, default=datetime.utcnow)
    video_fingerprint = Column(JSONEncodedDict)  # Stored as JSON string in NVARCHAR(MAX)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    scenes = relationship("Scene", back_populates="video", cascade="all, delete-orphan")
    transcript_segments = relationship(
        "TranscriptSegment", back_populates="video", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Video(id={self.id}, filename='{self.filename}')>"

class Scene(Base):
    """Scene/shot detection table."""
    __tablename__ = "scenes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    scene_id = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)
    start_frame = Column(Integer)
    end_frame = Column(Integer)
    keyframe_path = Column(Text)
    
    ocr_text = Column(Text)
    object_labels = Column(JSONEncodedDict)
    caption = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    video = relationship("Video", back_populates="scenes")
    transcript_segments = relationship("TranscriptSegment", back_populates="scene")

class TranscriptSegment(Base):
    """Transcript segments with timestamps."""
    __tablename__ = "transcript_segments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    scene_id = Column(Integer, ForeignKey("scenes.id", ondelete="SET NULL"))
    segment_index = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    language = Column(String(10), default="en")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    video = relationship("Video", back_populates="transcript_segments")
    scene = relationship("Scene", back_populates="transcript_segments")

class Embedding(Base):
    """Text embeddings for semantic search."""
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    segment_id = Column(
        Integer, ForeignKey("transcript_segments.id", ondelete="CASCADE"), nullable=True
    )
    scene_id = Column(
        Integer, ForeignKey("scenes.id", ondelete="CASCADE"), nullable=True
    )
    # Storing embeddings as JSON strings or byte arrays. SQL Server 2025 has VECTOR, 
    # but for compatibility we'll use NVARCHAR(MAX) or VARBINARY(MAX).
    # Since schema_sqlserver.sql uses VECTOR, we'll try to use a compatible type mapping.
    embedding = Column(Text) 
    embedding_model = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

class VisualEmbedding(Base):
    """Visual embeddings for keyframes/scenes using CLIP."""
    __tablename__ = "visual_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scene_id = Column(
        Integer, ForeignKey("scenes.id", ondelete="CASCADE"), nullable=False
    )
    keyframe_path = Column(Text, nullable=False)
    embedding = Column(Text)
    embedding_model = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
