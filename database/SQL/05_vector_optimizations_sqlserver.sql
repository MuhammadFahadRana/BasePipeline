SET NOCOUNT ON;
USE [VideoSemanticDB];
GO

PRINT 'Applying SQL Server vector optimizations...';

/* ============================================================================
   Vector Indexes for Fast Similarity Search
   ============================================================================ */

-- Check if vector indexes exist and create them
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_embeddings_vector_cosine')
BEGIN
    -- For embeddings table - create a covering index for vector searches
    CREATE INDEX idx_embeddings_vector_cosine
    ON dbo.embeddings(segment_id, embedding_model)
    INCLUDE (embedding)
    WHERE embedding IS NOT NULL;
END;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_visual_embeddings_vector_cosine')
BEGIN
    -- For visual embeddings table
    CREATE INDEX idx_visual_embeddings_vector_cosine
    ON dbo.visual_embeddings(scene_id, embedding_model)
    INCLUDE (embedding)
    WHERE embedding IS NOT NULL;
END;
GO

/* ============================================================================
   Optimized Cosine Similarity Function for VECTOR type
   ============================================================================ */

CREATE OR ALTER FUNCTION dbo.fn_vector_cosine_similarity
(
    @vector_a VECTOR(4096),
    @vector_b VECTOR(4096)
)
RETURNS FLOAT
AS
BEGIN
    -- Use native vector operations if available, fallback to JSON
    IF @vector_a IS NULL OR @vector_b IS NULL
        RETURN NULL;

    -- For VECTOR type, we can use direct operations
    -- This is a placeholder - actual implementation depends on SQL Server vector functions
    -- For now, convert to JSON and use existing function
    DECLARE @json_a NVARCHAR(MAX) = CAST(@vector_a AS NVARCHAR(MAX));
    DECLARE @json_b NVARCHAR(MAX) = CAST(@vector_b AS NVARCHAR(MAX));

    RETURN dbo.fn_cosine_similarity_json(@json_a, @json_b);
END;
GO

/* ============================================================================
   Optimized Hybrid Search Procedure
   ============================================================================ */

CREATE OR ALTER PROCEDURE dbo.hybrid_search_optimized
    @query_text NVARCHAR(MAX),
    @query_embedding NVARCHAR(MAX) = NULL,
    @text_weight FLOAT = 0.3,
    @semantic_weight FLOAT = 0.7,
    @limit_results INT = 10,
    @min_score FLOAT = 0.0
AS
BEGIN
    SET NOCOUNT ON;

    IF @limit_results IS NULL OR @limit_results < 1
        SET @limit_results = 10;

    IF @text_weight IS NULL
        SET @text_weight = 0.3;

    IF @semantic_weight IS NULL
        SET @semantic_weight = 0.7;

    IF @min_score IS NULL
        SET @min_score = 0.0;

    -- Pre-filter candidates using text search first (faster)
    ;WITH text_candidates AS (
        SELECT
            ts.id,
            ts.video_id,
            ts.start_time,
            ts.end_time,
            ts.[text],
            v.filename,
            CASE WHEN ts.[text] LIKE '%' + @query_text + '%' THEN 1.0 ELSE 0.0 END AS text_score
        FROM dbo.transcript_segments ts
        JOIN dbo.videos v ON v.id = ts.video_id
        WHERE ts.[text] LIKE '%' + @query_text + '%'
    ),
    -- Get semantic scores only for text matches (reduce computation)
    semantic_scores AS (
        SELECT
            e.segment_id,
            dbo.fn_cosine_similarity_json(e.embedding, @query_embedding) AS semantic_score
        FROM dbo.embeddings e
        WHERE e.segment_id IS NOT NULL
          AND @query_embedding IS NOT NULL
          AND ISJSON(@query_embedding) = 1
          AND EXISTS (SELECT 1 FROM text_candidates tc WHERE tc.id = e.segment_id)
    )
    SELECT TOP (@limit_results)
        tc.id AS segment_id,
        tc.filename AS video_filename,
        tc.start_time,
        tc.end_time,
        tc.[text],
        (tc.text_score * @text_weight +
         ISNULL(ss.semantic_score, 0.0) * @semantic_weight) AS combined_score
    FROM text_candidates tc
    LEFT JOIN semantic_scores ss ON ss.segment_id = tc.id
    WHERE (tc.text_score * @text_weight + ISNULL(ss.semantic_score, 0.0) * @semantic_weight) >= @min_score
    ORDER BY combined_score DESC, tc.start_time ASC;
END;
GO

/* ============================================================================
   Full Semantic Search (when no text match)
   ============================================================================ */

CREATE OR ALTER PROCEDURE dbo.semantic_search_only
    @query_embedding NVARCHAR(MAX),
    @limit_results INT = 10,
    @min_score FLOAT = 0.1
AS
BEGIN
    SET NOCOUNT ON;

    IF @limit_results IS NULL OR @limit_results < 1
        SET @limit_results = 10;

    IF @min_score IS NULL
        SET @min_score = 0.1;

    SELECT TOP (@limit_results)
        ts.id AS segment_id,
        v.filename AS video_filename,
        ts.start_time,
        ts.end_time,
        ts.[text],
        dbo.fn_cosine_similarity_json(e.embedding, @query_embedding) AS combined_score
    FROM dbo.embeddings e
    JOIN dbo.transcript_segments ts ON ts.id = e.segment_id
    JOIN dbo.videos v ON v.id = ts.video_id
    WHERE @query_embedding IS NOT NULL
      AND ISJSON(@query_embedding) = 1
      AND dbo.fn_cosine_similarity_json(e.embedding, @query_embedding) >= @min_score
    ORDER BY combined_score DESC, ts.start_time ASC;
END;
GO

/* ============================================================================
   Query Cache Optimization
   ============================================================================ */

-- Add index on query cache for faster lookups
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_query_cache_lookup')
BEGIN
    CREATE INDEX idx_query_cache_lookup
    ON dbo.query_cache(query_hash, expires_at)
    INCLUDE (cached_results, hit_count);
END;
GO

/* ============================================================================
   Statistics Update
   ============================================================================ */

EXEC sp_updatestats;
GO

PRINT 'SQL Server vector optimizations completed.';
GO