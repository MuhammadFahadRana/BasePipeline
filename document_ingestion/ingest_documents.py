"""Database ingestion logic for the document pipeline."""

import json
from pathlib import Path
from typing import Dict, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import text

# Assuming we run this from the BasePipeline root
from database.config import SessionLocal
from database.document_models import Document, DocumentChunk, DocumentEmbedding

# Reuse text_embeddings logic from the video pipeline
from embeddings.text_embeddings import get_embedding_generator

class DocumentIngester:
    """Ingests documents, chunks, and embeddings into database."""

    def __init__(self, db: Optional[Session] = None):
        self.db = db or SessionLocal()
        self.own_session = db is None
        self.embedding_gen = get_embedding_generator()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.own_session:
            self.db.close()

    def ingest_document(self, results_dict: Dict) -> int:
        """
        Ingests the processed document results into DB.
        Returns document ID.
        """
        meta = results_dict.get("metadata", {})
        chunks = results_dict.get("chunks", [])

        if not meta or not chunks:
            print("Warning: Missing metadata or chunks in result.")
            return -1

        # Check if document already exists
        existing_doc = self.db.query(Document).filter(
            Document.filename == meta.get("filename"),
            Document.file_path == meta.get("file_path")
        ).first()

        if existing_doc:
            print(f"Updating existing document: {existing_doc.filename}")
            doc = existing_doc
            doc.file_size_mb = meta.get("file_size_mb")
            doc.total_pages = meta.get("total_pages")
            doc.extraction_method = meta.get("extraction_method")
            doc.ocr_model = meta.get("ocr_model")
            doc.language = meta.get("language", "en")
            
            # Delete old chunks (cascade handles embeddings)
            self.db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()
        else:
            doc = Document(
                filename=meta.get("filename"),
                file_path=meta.get("file_path"),
                file_type=meta.get("file_type"),
                file_size_mb=meta.get("file_size_mb"),
                total_pages=meta.get("total_pages"),
                extraction_method=meta.get("extraction_method"),
                ocr_model=meta.get("ocr_model"),
                language=meta.get("language", "en")
            )
            self.db.add(doc)

        self.db.flush() # Get doc ID

        chunk_records = []
        texts_to_embed = []

        for c_dict in chunks:
            chunk = DocumentChunk(
                document_id=doc.id,
                chunk_index=c_dict.get("chunk_index"),
                page_number=c_dict.get("page_number"),
                section_heading=c_dict.get("section_heading"),
                text=c_dict.get("text", ""),
                summary=c_dict.get("summary"),
                ocr_confidence=c_dict.get("ocr_confidence")
            )
            chunk_records.append(chunk)

            # Combine summary and text for embedding if available
            embed_text = chunk.text
            if chunk.summary:
                embed_text = f"[{chunk.summary}] {chunk.text}"
                
            texts_to_embed.append(embed_text)

        self.db.add_all(chunk_records)
        self.db.flush() # Get chunk IDs

        # 3. Generate Embeddings
        print(f"Generating embeddings for {len(texts_to_embed)} chunks...")
        if texts_to_embed:
            vectors = self.embedding_gen.encode(texts_to_embed, batch_size=16)
            
            embed_records = []
            for chunk, vec in zip(chunk_records, vectors):
                emb = DocumentEmbedding(
                    chunk_id=chunk.id,
                    embedding=vec.tolist(),
                    embedding_model=self.embedding_gen.model_name
                )
                embed_records.append(emb)
                
            self.db.add_all(embed_records)
        
        self.db.commit()
        print(f"[OK] Ingested document: {doc.filename} ({len(chunk_records)} chunks)")
        return doc.id
