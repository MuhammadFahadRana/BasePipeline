-- ============================================================================
-- Document Processing Pipeline — Database Schema
-- Run AFTER the main schema.sql (requires pgvector and video_categories).
-- ============================================================================

-- Documents table: metadata about ingested documents
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(500) NOT NULL,
    file_path TEXT NOT NULL,
    file_type VARCHAR(20) NOT NULL,              -- pdf, docx, pptx, image
    file_size_mb FLOAT,
    total_pages INTEGER,
    extraction_method VARCHAR(20),                -- text, ocr, mixed
    ocr_model VARCHAR(100),
    language VARCHAR(10) DEFAULT 'en',
    category_id INTEGER REFERENCES video_categories(id) ON DELETE SET NULL,
    label VARCHAR(255),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_document_file UNIQUE(filename, file_path)
);

-- Document chunks: text segments from documents (analogous to transcript_segments)
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    section_heading TEXT,
    text TEXT NOT NULL,
    summary TEXT,                                  -- LLM-generated summary
    ocr_confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_document_chunk UNIQUE(document_id, chunk_index)
);

-- Document embeddings: same model/dimension as transcript embeddings
CREATE TABLE IF NOT EXISTS document_embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    embedding vector(1024),                        -- Qwen3-Embedding-0.6B
    embedding_model VARCHAR(100) DEFAULT 'Qwen/Qwen3-Embedding-0.6B',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_doc_chunk_embedding UNIQUE(chunk_id, embedding_model)
);

-- ── Indexes ─────────────────────────────────────────────────────────────────

-- Fast lookup by document
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_document_id ON document_chunks(document_id);

-- Full-text search on chunk text
CREATE INDEX IF NOT EXISTS idx_doc_chunks_text
    ON document_chunks USING GIN(to_tsvector('simple', text));

-- Vector similarity index (HNSW) — same parameters as transcript embeddings
CREATE INDEX IF NOT EXISTS idx_doc_embeddings_vector
    ON document_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- Model lookup
CREATE INDEX IF NOT EXISTS idx_doc_embeddings_model
    ON document_embeddings(embedding_model);

-- Trigger: update `updated_at` on documents
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
