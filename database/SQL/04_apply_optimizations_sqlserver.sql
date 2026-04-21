SET NOCOUNT ON;
USE [VideoSemanticDB];
GO

PRINT 'Applying SQL Server optimizations...';

/* ============================================================================
   Composite Indexes
   ============================================================================ */

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_transcript_video_time' AND object_id = OBJECT_ID(N'dbo.transcript_segments'))
    CREATE INDEX idx_transcript_video_time ON dbo.transcript_segments(video_id, start_time, end_time);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_embeddings_segment_model' AND object_id = OBJECT_ID(N'dbo.embeddings'))
    CREATE INDEX idx_embeddings_segment_model ON dbo.embeddings(segment_id, embedding_model);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_visual_embeddings_scene_model' AND object_id = OBJECT_ID(N'dbo.visual_embeddings'))
    CREATE INDEX idx_visual_embeddings_scene_model ON dbo.visual_embeddings(scene_id, embedding_model);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_scenes_video_scene' AND object_id = OBJECT_ID(N'dbo.scenes'))
    CREATE INDEX idx_scenes_video_scene ON dbo.scenes(video_id, scene_id);

IF OBJECT_ID(N'dbo.document_embeddings', N'U') IS NOT NULL
   AND NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_doc_embeddings_chunk_model' AND object_id = OBJECT_ID(N'dbo.document_embeddings'))
    CREATE INDEX idx_doc_embeddings_chunk_model ON dbo.document_embeddings(chunk_id, embedding_model);
GO

/* ============================================================================
   Query Cache Indexes
   ============================================================================ */

IF OBJECT_ID(N'dbo.query_cache', N'U') IS NOT NULL
BEGIN
    IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_query_cache_hash' AND object_id = OBJECT_ID(N'dbo.query_cache'))
        CREATE INDEX idx_query_cache_hash ON dbo.query_cache(query_hash);

    IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_query_cache_expires' AND object_id = OBJECT_ID(N'dbo.query_cache'))
        CREATE INDEX idx_query_cache_expires ON dbo.query_cache(expires_at);

    IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_query_cache_hits' AND object_id = OBJECT_ID(N'dbo.query_cache'))
        CREATE INDEX idx_query_cache_hits ON dbo.query_cache(hit_count DESC);
END;
GO

/* ============================================================================
   Statistics and Lightweight Maintenance
   ============================================================================ */

EXEC sp_updatestats;
GO

IF OBJECT_ID(N'dbo.clean_query_cache', N'P') IS NOT NULL
BEGIN
    EXEC dbo.clean_query_cache;
END;
GO

PRINT 'Optimization complete.';
GO
