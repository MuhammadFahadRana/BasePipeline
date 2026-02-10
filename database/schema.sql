-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Videos table: Store video metadata
CREATE TABLE IF NOT EXISTS videos (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    file_size_mb FLOAT,
    duration_seconds FLOAT,
    whisper_model VARCHAR(50),
    scene_threshold FLOAT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    video_fingerprint JSONB,  -- Store size, mtime, sha256
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Scenes table: Store detected scene/shot boundaries
CREATE TABLE IF NOT EXISTS scenes (
    id SERIAL PRIMARY KEY,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    scene_id INTEGER NOT NULL,  -- Scene number within video
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    duration FLOAT NOT NULL,
    start_frame INTEGER,
    end_frame INTEGER,
    keyframe_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(video_id, scene_id)
);

-- Transcript Segments: Store transcript segments with timestamps
CREATE TABLE IF NOT EXISTS transcript_segments (
    id SERIAL PRIMARY KEY,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    scene_id INTEGER REFERENCES scenes(id) ON DELETE SET NULL,  -- Optional: link to scene
    segment_index INTEGER NOT NULL,  -- Order in transcript
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    text TEXT NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(video_id, segment_index)
);

-- Embeddings table: Store text embeddings for semantic search
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    segment_id INTEGER REFERENCES transcript_segments(id) ON DELETE CASCADE,
    scene_id INTEGER REFERENCES scenes(id) ON DELETE CASCADE,
    embedding vector(1024), -- Qwen3-Embedding-0.6B dimension
    embedding_model VARCHAR(100) DEFAULT 'Qwen/Qwen3-Embedding-0.6B',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_embedding_source UNIQUE(segment_id, scene_id, embedding_model)
);

-- Visual Embeddings table: Store visual embeddings for keyframes
CREATE TABLE IF NOT EXISTS visual_embeddings (
    id SERIAL PRIMARY KEY,
    scene_id INTEGER REFERENCES scenes(id) ON DELETE CASCADE,
    keyframe_path TEXT NOT NULL,
    embedding vector(768), -- SigLIP dimension
    embedding_model VARCHAR(100) DEFAULT 'google/siglip-base-patch16-224',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Query cache table: Store query results for performance
CREATE TABLE IF NOT EXISTS query_cache (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_hash VARCHAR(64) UNIQUE NOT NULL,
    query_params JSONB,
    cached_results JSONB,
    hit_count INTEGER DEFAULT 1,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Search queries log: Track user queries for analytics
CREATE TABLE IF NOT EXISTS search_queries (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding vector(1024), -- Unified dimension
    search_type VARCHAR(20) DEFAULT 'text',  -- text, visual, image, hybrid
    results_count INTEGER,
    top_result_id INTEGER REFERENCES transcript_segments(id),
    search_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Search image cache: Store uploaded image embeddings for re-ranking and history
CREATE TABLE IF NOT EXISTS search_image_cache (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255),
    image_hash VARCHAR(64) UNIQUE,
    embedding vector(768), -- SigLIP dimension
    embedding_model VARCHAR(100) DEFAULT 'google/siglip-base-patch16-224',
    search_count INTEGER DEFAULT 1,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_videos_filename ON videos(filename);
CREATE INDEX IF NOT EXISTS idx_scenes_video_id ON scenes(video_id);
CREATE INDEX IF NOT EXISTS idx_scenes_time_range ON scenes(video_id, start_time, end_time);
CREATE INDEX IF NOT EXISTS idx_transcript_video_id ON transcript_segments(video_id);
CREATE INDEX IF NOT EXISTS idx_transcript_time_range ON transcript_segments(video_id, start_time, end_time);
CREATE INDEX IF NOT EXISTS idx_transcript_text_search ON transcript_segments USING GIN(to_tsvector('english', text));

-- Vector similarity indexes (HNSW for fast approximate nearest-neighbor search)
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

CREATE INDEX IF NOT EXISTS idx_visual_embeddings_vector ON visual_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

CREATE INDEX IF NOT EXISTS idx_search_image_cache_vector ON search_image_cache USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- Partitioning support: composite index for fast lookup by model
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(embedding_model);
CREATE INDEX IF NOT EXISTS idx_visual_embeddings_model ON visual_embeddings(embedding_model);

-- Cleanup function: remove orphaned visual embeddings (stale data from re-processing)
CREATE OR REPLACE FUNCTION cleanup_stale_visual_embeddings()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM visual_embeddings ve
    WHERE NOT EXISTS (
        SELECT 1 FROM scenes s WHERE s.id = ve.scene_id
    );
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Cleanup function: remove orphaned text embeddings
CREATE OR REPLACE FUNCTION cleanup_stale_embeddings()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM embeddings e
    WHERE e.segment_id IS NOT NULL
      AND NOT EXISTS (
        SELECT 1 FROM transcript_segments ts WHERE ts.id = e.segment_id
    );
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update `updated_at` timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_videos_updated_at BEFORE UPDATE ON videos
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to search with fuzzy text matching + semantic similarity
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding vector(1024),
    text_weight FLOAT DEFAULT 0.3,
    semantic_weight FLOAT DEFAULT 0.7,
    limit_results INTEGER DEFAULT 10
)
RETURNS TABLE (
    segment_id INTEGER,
    video_filename VARCHAR,
    start_time FLOAT,
    end_time FLOAT,
    text TEXT,
    combined_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH text_scores AS (
        SELECT 
            ts.id,
            ts_rank(to_tsvector('english', ts.text), plainto_tsquery('english', query_text)) AS text_score
        FROM transcript_segments ts
        WHERE to_tsvector('english', ts.text) @@ plainto_tsquery('english', query_text)
    ),
    semantic_scores AS (
        SELECT 
            e.segment_id,
            1 - (e.embedding <=> query_embedding) AS semantic_score
        FROM embeddings e
    )
    SELECT 
        ts.id AS segment_id,
        v.filename AS video_filename,
        ts.start_time,
        ts.end_time,
        ts.text,
        (COALESCE(txt.text_score, 0) * text_weight + 
         COALESCE(sem.semantic_score, 0) * semantic_weight) AS combined_score
    FROM transcript_segments ts
    JOIN videos v ON ts.video_id = v.id
    LEFT JOIN text_scores txt ON txt.id = ts.id
    LEFT JOIN semantic_scores sem ON sem.segment_id = ts.id
    WHERE txt.text_score IS NOT NULL OR sem.semantic_score IS NOT NULL
    ORDER BY combined_score DESC
    LIMIT limit_results;
END;
$$ LANGUAGE plpgsql;
