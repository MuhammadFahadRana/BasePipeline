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


class VideoCategory(Base):
    """Predefined video categories (e.g. Oil & Gas, Maintenance)."""

    __tablename__ = "video_categories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    videos = relationship("Video", back_populates="category_rel")

    def __repr__(self):
        return f"<VideoCategory(id={self.id}, name='{self.name}')>"


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
    label = Column(String(255))  # Human-readable site label, e.g. "Yggdrasil"
    category_id = Column(Integer, ForeignKey("video_categories.id", ondelete="SET NULL"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    category_rel = relationship("VideoCategory", back_populates="videos")
    scenes = relationship("Scene", back_populates="video", cascade="all, delete-orphan")
    transcript_segments = relationship(
        "TranscriptSegment", back_populates="video", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Video(id={self.id}, filename='{self.filename}', label='{self.label}')>"


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
    ocr_text_norm = Column(Text)  # Normalized OCR text for retrieval
    ocr_confidence = Column(Float)  # Mean confidence across OCR detections
    ocr_processed_at = Column(DateTime)  # When OCR was last run
    
    # Semantic enrichment fields
    object_labels = Column(JSON)  # All detected objects and labels
    caption = Column(Text)        # Narrative description of the scene
    
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
    embedding = Column(Vector(1024))
    embedding_model = Column(String(100), default="Qwen/Qwen3-Embedding-0.6B")
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
    sample_time = Column(Float)  # Timestamp (seconds) of sampled frame
    frame_role = Column(String(20), default="mid")  # start/mid/end/extra_n
    frame_index = Column(Integer)  # Absolute frame index in source video
    embedding = Column(Vector(768))  # 768-dim for SigLIP (google/siglip-base-patch16-224)
    embedding_model = Column(String(100), default="google/siglip-base-patch16-224")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    scene = relationship("Scene")

    __table_args__ = (
        UniqueConstraint(
            "scene_id",
            "embedding_model",
            "frame_role",
            "sample_time",
            name="uq_scene_visual_embedding",
        ),
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


# ──────────────────────────────────────────────────────────────────────────
# Access Control models
# ──────────────────────────────────────────────────────────────────────────

# -----------------------------------------------------------------------------
# Feedback/learning telemetry models (Phase 1)
# -----------------------------------------------------------------------------


class SearchRequestLog(Base):
    """One row per user-visible search response."""

    __tablename__ = "search_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_uuid = Column(String(64), nullable=False, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    query_text = Column(Text, nullable=False)
    search_mode = Column(String(40), nullable=False, default="text", index=True)
    facet = Column(String(30))
    filters = Column(JSON)
    results_count = Column(Integer, default=0)
    latency_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    request_impressions = relationship(
        "SearchImpression",
        back_populates="search_request",
        cascade="all, delete-orphan",
    )
    request_interactions = relationship(
        "SearchInteraction",
        back_populates="search_request",
        cascade="all, delete-orphan",
    )
    request_feedback = relationship(
        "SearchFeedback",
        back_populates="search_request",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return (
            f"<SearchRequestLog(id={self.id}, mode='{self.search_mode}', "
            f"query='{self.query_text[:60]}...')>"
        )


class SearchImpression(Base):
    """A ranked result shown to the user for a specific request."""

    __tablename__ = "search_impressions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(
        Integer, ForeignKey("search_requests.id", ondelete="CASCADE"), nullable=False
    )
    impression_rank = Column(Integer, nullable=False)
    source_type = Column(String(20), default="video")  # video, document, image, etc.
    result_segment_id = Column(Integer)  # transcript segment id OR document chunk id
    result_video_id = Column(Integer)
    result_video_filename = Column(String(255))
    result_start_time = Column(Float)
    result_end_time = Column(Float)
    result_score = Column(Float)
    result_payload = Column(JSON)  # lightweight serialized result as shown to user
    created_at = Column(DateTime, default=datetime.utcnow)

    search_request = relationship("SearchRequestLog", back_populates="request_impressions")
    impression_interactions = relationship(
        "SearchInteraction",
        back_populates="search_impression",
        cascade="all, delete-orphan",
    )
    impression_feedback = relationship(
        "SearchFeedback",
        back_populates="search_impression",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("request_id", "impression_rank", name="uq_search_impression_rank"),
    )

    def __repr__(self):
        return (
            f"<SearchImpression(id={self.id}, request_id={self.request_id}, "
            f"rank={self.impression_rank})>"
        )


class SearchInteraction(Base):
    """Implicit behavior signal (click, dwell, open video, etc.)."""

    __tablename__ = "search_interactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(
        Integer, ForeignKey("search_requests.id", ondelete="CASCADE"), nullable=False
    )
    impression_id = Column(
        Integer, ForeignKey("search_impressions.id", ondelete="SET NULL")
    )
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    interaction_type = Column(String(40), nullable=False, index=True)
    dwell_ms = Column(Integer)
    event_metadata = Column("metadata", JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    search_request = relationship("SearchRequestLog", back_populates="request_interactions")
    search_impression = relationship("SearchImpression", back_populates="impression_interactions")

    def __repr__(self):
        return (
            f"<SearchInteraction(id={self.id}, type='{self.interaction_type}', "
            f"request_id={self.request_id})>"
        )


class SearchFeedback(Base):
    """Explicit user judgment signal (relevant / not relevant)."""

    __tablename__ = "search_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(
        Integer, ForeignKey("search_requests.id", ondelete="CASCADE"), nullable=False
    )
    impression_id = Column(
        Integer, ForeignKey("search_impressions.id", ondelete="SET NULL")
    )
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    feedback_value = Column(Integer, nullable=False)  # -1 = irrelevant, +1 = relevant
    comment = Column(Text)
    feedback_metadata = Column("metadata", JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    search_request = relationship("SearchRequestLog", back_populates="request_feedback")
    search_impression = relationship("SearchImpression", back_populates="impression_feedback")

    def __repr__(self):
        return (
            f"<SearchFeedback(id={self.id}, value={self.feedback_value}, "
            f"request_id={self.request_id})>"
        )


class User(Base):
    """Application users with role-based access control."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="viewer")  # "admin" | "viewer"
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    category_access = relationship(
        "UserCategoryAccess", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"


class UserCategoryAccess(Base):
    """Maps which video categories a non-admin user can access."""

    __tablename__ = "user_category_access"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    category = Column(String(100), nullable=False)  # e.g. "Johan Sverdrup", "AkerBP"

    user = relationship("User", back_populates="category_access")

    __table_args__ = (
        UniqueConstraint("user_id", "category", name="uq_user_category"),
    )
