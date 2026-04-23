"""SQLAlchemy ORM models for document processing pipeline.

Separate from the video models (models.py) to keep concerns clean.
Shares the same Base so all tables are managed under one metadata.
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

# Import Base from the existing video models so all ORM classes share one metadata.
from database.models import Base


class Document(Base):
    """Ingested document metadata (PDF, DOCX, PPTX, image)."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(500), nullable=False)
    file_path = Column(Text, nullable=False)
    file_type = Column(String(20), nullable=False)  # pdf, docx, pptx, image
    file_size_mb = Column(Float)
    total_pages = Column(Integer)
    extraction_method = Column(String(20))  # text, ocr, mixed
    ocr_model = Column(String(100))  # e.g. "surya", "easyocr"
    language = Column(String(10), default="en")
    # Reuse video_categories for now (shared taxonomy)
    category_id = Column(
        Integer, ForeignKey("video_categories.id", ondelete="SET NULL")
    )
    label = Column(String(255))  # Site label, same concept as Video.label
    processed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    category_rel = relationship("VideoCategory")
    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("filename", "file_path", name="uq_document_file"),
    )

    def __repr__(self):
        return (
            f"<Document(id={self.id}, filename='{self.filename}', "
            f"type='{self.file_type}', pages={self.total_pages})>"
        )


class DocumentChunk(Base):
    """Text chunk from a document (analogous to TranscriptSegment for videos)."""

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index = Column(Integer, nullable=False)  # Order within document
    page_number = Column(Integer)  # Source page (1-indexed)
    section_heading = Column(Text)  # Detected heading/title
    text = Column(Text, nullable=False)  # Raw extracted text
    summary = Column(Text)  # LLM-generated summary (hybrid enrichment)
    ocr_confidence = Column(Float)  # OCR confidence (NULL if direct text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="chunks")
    embeddings = relationship(
        "DocumentEmbedding", back_populates="chunk", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk"),
    )

    def __repr__(self):
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return (
            f"<DocumentChunk(id={self.id}, doc_id={self.document_id}, "
            f"idx={self.chunk_index}, page={self.page_number}, '{preview}')>"
        )


class DocumentEmbedding(Base):
    """Text embedding for a document chunk (same model as transcript embeddings)."""

    __tablename__ = "document_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(
        Integer,
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=False,
    )
    embedding = Column(Vector(1024))
    embedding_model = Column(String(100), default="Qwen/Qwen3-Embedding-0.6B")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    chunk = relationship("DocumentChunk", back_populates="embeddings")

    __table_args__ = (
        UniqueConstraint("chunk_id", "embedding_model", name="uq_doc_chunk_embedding"),
    )

    def __repr__(self):
        return (
            f"<DocumentEmbedding(id={self.id}, chunk_id={self.chunk_id}, "
            f"model='{self.embedding_model}')>"
        )
