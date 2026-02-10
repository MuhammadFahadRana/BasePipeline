"""SQLAlchemy ORM models for video semantic search database."""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    ForeignKey,
    JSON,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()


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
    video_fingerprint = Column(JSON)  # {size_bytes, mtime, sha256}
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
    scene_id = Column(Integer, nullable=False)  # Scene number within video
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)
    start_frame = Column(Integer)
    end_frame = Column(Integer)
    keyframe_path = Column(Text)
    
    # OCR and enrichment fields
    ocr_text = Column(Text)  # Text extracted from keyframe via OCR
    ocr_processed_at = Column(DateTime)  # When OCR was last run
    
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    video = relationship("Video", back_populates="scenes")
    transcript_segments = relationship("TranscriptSegment", back_populates="scene")

    __table_args__ = (UniqueConstraint("video_id", "scene_id", name="uq_video_scene"),)

    def __repr__(self):
        return f"<Scene(id={self.id}, video_id={self.video_id}, scene_id={self.scene_id}, {self.start_time:.1f}s-{self.end_time:.1f}s)>"


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
    embeddings = relationship(
        "Embedding", back_populates="segment", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("video_id", "segment_index", name="uq_video_segment"),
    )

    def __repr__(self):
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"<TranscriptSegment(id={self.id}, video_id={self.video_id}, {self.start_time:.1f}s: '{text_preview}')>"


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
    embedding = Column(Vector(1024))  # 1024-dim vector for BAAI/bge-m3
    embedding_model = Column(String(100), default="BAAI/bge-m3")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    segment = relationship("TranscriptSegment", back_populates="embeddings")
    scene = relationship("Scene")

    __table_args__ = (
        UniqueConstraint("segment_id", "scene_id", "embedding_model", name="uq_embedding_source"),
    )

    def __repr__(self):
        source = f"segment={self.segment_id}" if self.segment_id else f"scene={self.scene_id}"
        return f"<Embedding(id={self.id}, {source}, model='{self.embedding_model}')>"


class QueryCache(Base):
    """Cache for search queries to improve performance."""

    __tablename__ = "query_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_text = Column(Text, nullable=False)
    query_hash = Column(String(64), unique=True, index=True, nullable=False)
    query_params = Column(JSON)
    cached_results = Column(JSON)
    hit_count = Column(Integer, default=1)
    last_used = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<QueryCache(id={self.id}, hash='{self.query_hash[:8]}', hits={self.hit_count})>"


class VisualEmbedding(Base):
    """Visual embeddings for keyframes/scenes using CLIP."""

    __tablename__ = "visual_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scene_id = Column(
        Integer, ForeignKey("scenes.id", ondelete="CASCADE"), nullable=False
    )
    keyframe_path = Column(Text, nullable=False)
    embedding = Column(Vector(768))  # 768-dim for SigLIP (google/siglip-base-patch16-224)
    embedding_model = Column(String(100), default="google/siglip-base-patch16-224")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    scene = relationship("Scene")

    __table_args__ = (
        UniqueConstraint("scene_id", "embedding_model", name="uq_scene_visual_embedding"),
    )

    def __repr__(self):
        return f"<VisualEmbedding(id={self.id}, scene_id={self.scene_id}, model='{self.embedding_model}')>"


class SearchQuery(Base):
    """Log search queries for analytics."""

    __tablename__ = "search_queries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_text = Column(Text, nullable=False)
    query_embedding = Column(Vector(1024))
    search_type = Column(String(20), default="text")  # text, visual, image, hybrid
    results_count = Column(Integer)
    top_result_id = Column(
        Integer, ForeignKey("transcript_segments.id", ondelete="SET NULL")
    )
    search_timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<SearchQuery(id={self.id}, type='{self.search_type}', query='{self.query_text[:50]}...')>"


class SearchImageCache(Base):
    """Cache uploaded image embeddings for re-ranking, search history, and 'find more like this'."""

    __tablename__ = "search_image_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255))
    image_hash = Column(String(64), unique=True, index=True)  # SHA256 of image bytes
    embedding = Column(Vector(768))  # 768-dim for SigLIP
    embedding_model = Column(String(100), default="google/siglip-base-patch16-224")
    search_count = Column(Integer, default=1)
    last_used = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<SearchImageCache(id={self.id}, file='{self.filename}', searches={self.search_count})>"
