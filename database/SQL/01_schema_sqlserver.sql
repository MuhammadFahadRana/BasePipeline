SET NOCOUNT ON;
USE [VideoSemanticDB];
GO

/* ============================================================================
   Base Tables
   ============================================================================ */

IF OBJECT_ID(N'dbo.video_categories', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.video_categories (
        id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_video_categories PRIMARY KEY,
        name NVARCHAR(100) NOT NULL,
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_video_categories_created_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT UQ_video_categories_name UNIQUE (name)
    );
END;
GO

IF OBJECT_ID(N'dbo.users', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.users (
        id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_users PRIMARY KEY,
        username NVARCHAR(100) NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        role VARCHAR(20) NOT NULL CONSTRAINT DF_users_role DEFAULT 'viewer',
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_users_created_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT UQ_users_username UNIQUE (username)
    );
END;
GO

IF OBJECT_ID(N'dbo.user_category_access', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.user_category_access (
        id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_user_category_access PRIMARY KEY,
        user_id INT NOT NULL,
        category NVARCHAR(100) NOT NULL,
        CONSTRAINT FK_user_category_access_user
            FOREIGN KEY (user_id) REFERENCES dbo.users(id) ON DELETE CASCADE,
        CONSTRAINT UQ_user_category UNIQUE (user_id, category)
    );
END;
GO

IF OBJECT_ID(N'dbo.videos', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.videos (
        id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_videos PRIMARY KEY,
        filename NVARCHAR(255) NOT NULL,
        file_path NVARCHAR(MAX) NOT NULL,
        file_size_mb FLOAT NULL,
        duration_seconds FLOAT NULL,
        whisper_model VARCHAR(50) NULL,
        scene_threshold FLOAT NULL,
        processed_at DATETIME2(3) NOT NULL CONSTRAINT DF_videos_processed_at DEFAULT SYSUTCDATETIME(),
        video_fingerprint NVARCHAR(MAX) NULL,
        label NVARCHAR(255) NULL,
        category_id INT NULL,
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_videos_created_at DEFAULT SYSUTCDATETIME(),
        updated_at DATETIME2(3) NOT NULL CONSTRAINT DF_videos_updated_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT UQ_videos_filename UNIQUE (filename),
        CONSTRAINT CK_videos_video_fingerprint_json
            CHECK (video_fingerprint IS NULL OR ISJSON(video_fingerprint) = 1),
        CONSTRAINT FK_videos_category
            FOREIGN KEY (category_id) REFERENCES dbo.video_categories(id) ON DELETE SET NULL
    );
END;
GO

IF OBJECT_ID(N'dbo.scenes', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.scenes (
        id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_scenes PRIMARY KEY,
        video_id INT NOT NULL,
        scene_id INT NOT NULL,
        start_time FLOAT NOT NULL,
        end_time FLOAT NOT NULL,
        duration FLOAT NOT NULL,
        start_frame INT NULL,
        end_frame INT NULL,
        keyframe_path NVARCHAR(MAX) NULL,
        ocr_text NVARCHAR(MAX) NULL,
        ocr_text_norm NVARCHAR(MAX) NULL,
        ocr_confidence FLOAT NULL,
        ocr_processed_at DATETIME2(3) NULL,
        object_labels NVARCHAR(MAX) NULL,
        caption NVARCHAR(MAX) NULL,
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_scenes_created_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT CK_scenes_object_labels_json CHECK (object_labels IS NULL OR ISJSON(object_labels) = 1),
        CONSTRAINT FK_scenes_videos
            FOREIGN KEY (video_id) REFERENCES dbo.videos(id) ON DELETE CASCADE,
        CONSTRAINT UQ_scenes_video_scene UNIQUE (video_id, scene_id)
    );
END;
GO

IF OBJECT_ID(N'dbo.transcript_segments', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.transcript_segments (
        id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_transcript_segments PRIMARY KEY,
        video_id INT NOT NULL,
        scene_id INT NULL,
        segment_index INT NOT NULL,
        start_time FLOAT NOT NULL,
        end_time FLOAT NOT NULL,
        [text] NVARCHAR(MAX) NOT NULL,
        [language] VARCHAR(10) NOT NULL CONSTRAINT DF_transcript_language DEFAULT 'en',
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_transcript_created_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT FK_transcript_video
            FOREIGN KEY (video_id) REFERENCES dbo.videos(id) ON DELETE CASCADE,
        CONSTRAINT FK_transcript_scene
            FOREIGN KEY (scene_id) REFERENCES dbo.scenes(id),
        CONSTRAINT UQ_transcript_video_segment UNIQUE (video_id, segment_index)
    );
END;
GO

IF OBJECT_ID(N'dbo.embeddings', N'U') IS NULL
BEGIN
    DECLARE @HasVectorEmb BIT = CASE WHEN EXISTS (SELECT 1 FROM sys.types WHERE name = N'vector') THEN 1 ELSE 0 END;
    DECLARE @EmbeddingDim INT = 1024;
    DECLARE @UseVectorEmb BIT = CASE WHEN @HasVectorEmb = 1 AND @EmbeddingDim <= 1998 THEN 1 ELSE 0 END;
    DECLARE @EmbeddingColumnDef NVARCHAR(200) =
        CASE WHEN @UseVectorEmb = 1 THEN N'embedding VECTOR(1024) NULL,' ELSE N'embedding NVARCHAR(MAX) NULL,' END;
    DECLARE @EmbeddingJsonConstraint NVARCHAR(MAX) =
        CASE WHEN @HasVectorEmb = 1 THEN N'' ELSE N'CONSTRAINT CK_embeddings_embedding_json CHECK (embedding IS NULL OR ISJSON(embedding) = 1),' END;

    DECLARE @EmbSql NVARCHAR(MAX) = N'
CREATE TABLE dbo.embeddings (
    id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_embeddings PRIMARY KEY,
    segment_id INT NULL,
    scene_id INT NULL,
    ' + @EmbeddingColumnDef + N'
    embedding_model VARCHAR(100) NOT NULL CONSTRAINT DF_embeddings_model DEFAULT ''Qwen/Qwen3-Embedding-0.6B'',
    created_at DATETIME2(3) NOT NULL CONSTRAINT DF_embeddings_created_at DEFAULT SYSUTCDATETIME(),
    ' + @EmbeddingJsonConstraint + N'
    CONSTRAINT FK_embeddings_segment
        FOREIGN KEY (segment_id) REFERENCES dbo.transcript_segments(id) ON DELETE CASCADE,
    CONSTRAINT UQ_embeddings_source UNIQUE (segment_id, scene_id, embedding_model)
);';

    EXEC sp_executesql @EmbSql;
END;
GO

IF OBJECT_ID(N'dbo.visual_embeddings', N'U') IS NULL
BEGIN
    DECLARE @HasVectorVisual BIT = CASE WHEN EXISTS (SELECT 1 FROM sys.types WHERE name = N'vector') THEN 1 ELSE 0 END;
    DECLARE @VisualEmbeddingColumnDef NVARCHAR(200) =
        CASE WHEN @HasVectorVisual = 1 THEN N'embedding VECTOR(768) NULL,' ELSE N'embedding NVARCHAR(MAX) NULL,' END;
    DECLARE @VisualEmbeddingJsonConstraint NVARCHAR(MAX) =
        CASE WHEN @HasVectorVisual = 1 THEN N'' ELSE N'CONSTRAINT CK_visual_embeddings_embedding_json CHECK (embedding IS NULL OR ISJSON(embedding) = 1),' END;

    DECLARE @VisualSql NVARCHAR(MAX) = N'
CREATE TABLE dbo.visual_embeddings (
    id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_visual_embeddings PRIMARY KEY,
    scene_id INT NOT NULL,
    keyframe_path NVARCHAR(MAX) NOT NULL,
    sample_time FLOAT NULL,
    frame_role VARCHAR(20) NOT NULL CONSTRAINT DF_visual_embeddings_frame_role DEFAULT ''mid'',
    frame_index INT NULL,
    ' + @VisualEmbeddingColumnDef + N'
    embedding_model VARCHAR(100) NOT NULL CONSTRAINT DF_visual_embeddings_model DEFAULT ''google/siglip2-base-patch16-224'',
    created_at DATETIME2(3) NOT NULL CONSTRAINT DF_visual_embeddings_created_at DEFAULT SYSUTCDATETIME(),
    ' + @VisualEmbeddingJsonConstraint + N'
    CONSTRAINT FK_visual_embeddings_scene
        FOREIGN KEY (scene_id) REFERENCES dbo.scenes(id) ON DELETE CASCADE,
    CONSTRAINT UQ_scene_visual_embedding UNIQUE (scene_id, embedding_model, frame_role, sample_time)
);';

    EXEC sp_executesql @VisualSql;
END;
GO

IF OBJECT_ID(N'dbo.video_embeddings', N'U') IS NULL
BEGIN
    DECLARE @HasVectorVideo BIT = CASE WHEN EXISTS (SELECT 1 FROM sys.types WHERE name = N'vector') THEN 1 ELSE 0 END;
    DECLARE @VideoEmbeddingColumnDef NVARCHAR(200) =
        CASE WHEN @HasVectorVideo = 1 THEN N'embedding VECTOR(768) NULL,' ELSE N'embedding NVARCHAR(MAX) NULL,' END;
    DECLARE @VideoEmbeddingJsonConstraint NVARCHAR(MAX) =
        CASE WHEN @HasVectorVideo = 1 THEN N'' ELSE N'CONSTRAINT CK_video_embeddings_embedding_json CHECK (embedding IS NULL OR ISJSON(embedding) = 1),' END;

    DECLARE @VideoEmbSql NVARCHAR(MAX) = N'
CREATE TABLE dbo.video_embeddings (
    id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_video_embeddings PRIMARY KEY,
    video_id INT NOT NULL,
    ' + @VideoEmbeddingColumnDef + N'
    embedding_model VARCHAR(100) NOT NULL CONSTRAINT DF_video_embeddings_model DEFAULT ''video-temporal-mean:google/siglip2-base-patch16-224'',
    aggregation_method VARCHAR(50) NOT NULL CONSTRAINT DF_video_embeddings_aggregation DEFAULT ''temporal_mean'',
    frame_count INT NULL,
    created_at DATETIME2(3) NOT NULL CONSTRAINT DF_video_embeddings_created_at DEFAULT SYSUTCDATETIME(),
    ' + @VideoEmbeddingJsonConstraint + N'
    CONSTRAINT FK_video_embeddings_video
        FOREIGN KEY (video_id) REFERENCES dbo.videos(id) ON DELETE CASCADE,
    CONSTRAINT UQ_video_embedding UNIQUE (video_id, embedding_model, aggregation_method)
);';

    EXEC sp_executesql @VideoEmbSql;
END;
GO

IF OBJECT_ID(N'dbo.query_cache', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.query_cache (
        id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_query_cache PRIMARY KEY,
        query_text NVARCHAR(MAX) NOT NULL,
        query_hash VARCHAR(64) NOT NULL,
        query_params NVARCHAR(MAX) NULL,
        cached_results NVARCHAR(MAX) NULL,
        hit_count INT NOT NULL CONSTRAINT DF_query_cache_hit_count DEFAULT 1,
        last_used DATETIME2(3) NOT NULL CONSTRAINT DF_query_cache_last_used DEFAULT SYSUTCDATETIME(),
        expires_at DATETIME2(3) NOT NULL,
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_query_cache_created_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT UQ_query_cache_hash UNIQUE (query_hash),
        CONSTRAINT CK_query_cache_query_params_json CHECK (query_params IS NULL OR ISJSON(query_params) = 1),
        CONSTRAINT CK_query_cache_cached_results_json CHECK (cached_results IS NULL OR ISJSON(cached_results) = 1)
    );
END;
GO

IF OBJECT_ID(N'dbo.search_queries', N'U') IS NULL
BEGIN
    DECLARE @HasVectorSearchQuery BIT = CASE WHEN EXISTS (SELECT 1 FROM sys.types WHERE name = N'vector') THEN 1 ELSE 0 END;
    DECLARE @SearchQueryEmbeddingDim INT = 1024;
    DECLARE @UseVectorSearchQuery BIT = CASE WHEN @HasVectorSearchQuery = 1 AND @SearchQueryEmbeddingDim <= 1998 THEN 1 ELSE 0 END;
    DECLARE @SearchQueryEmbeddingColumnDef NVARCHAR(200) =
        CASE WHEN @UseVectorSearchQuery = 1 THEN N'query_embedding VECTOR(1024) NULL,' ELSE N'query_embedding NVARCHAR(MAX) NULL,' END;
    DECLARE @SearchQueryEmbeddingJsonConstraint NVARCHAR(MAX) =
        CASE WHEN @HasVectorSearchQuery = 1 THEN N'' ELSE N'CONSTRAINT CK_search_queries_query_embedding_json CHECK (query_embedding IS NULL OR ISJSON(query_embedding) = 1),' END;

    DECLARE @SearchQuerySql NVARCHAR(MAX) = N'
CREATE TABLE dbo.search_queries (
    id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_search_queries PRIMARY KEY,
    query_text NVARCHAR(MAX) NOT NULL,
    ' + @SearchQueryEmbeddingColumnDef + N'
    search_type VARCHAR(20) NOT NULL CONSTRAINT DF_search_queries_search_type DEFAULT ''text'',
    results_count INT NULL,
    top_result_id INT NULL,
    search_timestamp DATETIME2(3) NOT NULL CONSTRAINT DF_search_queries_search_timestamp DEFAULT SYSUTCDATETIME(),
    ' + @SearchQueryEmbeddingJsonConstraint + N'
    CONSTRAINT FK_search_queries_top_result
        FOREIGN KEY (top_result_id) REFERENCES dbo.transcript_segments(id) ON DELETE SET NULL
);';

    EXEC sp_executesql @SearchQuerySql;
END;
GO

IF OBJECT_ID(N'dbo.search_requests', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.search_requests (
        id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_search_requests PRIMARY KEY,
        request_uuid VARCHAR(64) NOT NULL,
        user_id INT NULL,
        query_text NVARCHAR(MAX) NOT NULL,
        search_mode VARCHAR(40) NOT NULL CONSTRAINT DF_search_requests_search_mode DEFAULT 'text',
        facet VARCHAR(30) NULL,
        filters NVARCHAR(MAX) NULL,
        results_count INT NOT NULL CONSTRAINT DF_search_requests_results_count DEFAULT 0,
        latency_ms FLOAT NULL,
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_search_requests_created_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT UQ_search_requests_uuid UNIQUE (request_uuid),
        CONSTRAINT CK_search_requests_filters_json CHECK (filters IS NULL OR ISJSON(filters) = 1),
        CONSTRAINT FK_search_requests_user
            FOREIGN KEY (user_id) REFERENCES dbo.users(id) ON DELETE SET NULL
    );
END;
GO

IF OBJECT_ID(N'dbo.search_impressions', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.search_impressions (
        id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_search_impressions PRIMARY KEY,
        request_id INT NOT NULL,
        impression_rank INT NOT NULL,
        source_type VARCHAR(20) NOT NULL CONSTRAINT DF_search_impressions_source_type DEFAULT 'video',
        result_segment_id INT NULL,
        result_video_id INT NULL,
        result_video_filename VARCHAR(255) NULL,
        result_start_time FLOAT NULL,
        result_end_time FLOAT NULL,
        result_score FLOAT NULL,
        result_payload NVARCHAR(MAX) NULL,
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_search_impressions_created_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT UQ_search_impression_rank UNIQUE (request_id, impression_rank),
        CONSTRAINT CK_search_impressions_payload_json CHECK (result_payload IS NULL OR ISJSON(result_payload) = 1),
        CONSTRAINT FK_search_impressions_request
            FOREIGN KEY (request_id) REFERENCES dbo.search_requests(id) ON DELETE CASCADE
    );
END;
GO

IF OBJECT_ID(N'dbo.search_interactions', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.search_interactions (
        id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_search_interactions PRIMARY KEY,
        request_id INT NOT NULL,
        impression_id INT NULL,
        user_id INT NULL,
        interaction_type VARCHAR(40) NOT NULL,
        dwell_ms INT NULL,
        metadata NVARCHAR(MAX) NULL,
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_search_interactions_created_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT CK_search_interactions_metadata_json CHECK (metadata IS NULL OR ISJSON(metadata) = 1),
        CONSTRAINT FK_search_interactions_request
            FOREIGN KEY (request_id) REFERENCES dbo.search_requests(id) ON DELETE CASCADE,
        CONSTRAINT FK_search_interactions_impression
            FOREIGN KEY (impression_id) REFERENCES dbo.search_impressions(id),
        CONSTRAINT FK_search_interactions_user
            FOREIGN KEY (user_id) REFERENCES dbo.users(id) ON DELETE SET NULL
    );
END;
GO

IF OBJECT_ID(N'dbo.search_feedback', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.search_feedback (
        id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_search_feedback PRIMARY KEY,
        request_id INT NOT NULL,
        impression_id INT NULL,
        user_id INT NULL,
        feedback_value INT NOT NULL,
        comment NVARCHAR(MAX) NULL,
        metadata NVARCHAR(MAX) NULL,
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_search_feedback_created_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT CK_search_feedback_metadata_json CHECK (metadata IS NULL OR ISJSON(metadata) = 1),
        CONSTRAINT FK_search_feedback_request
            FOREIGN KEY (request_id) REFERENCES dbo.search_requests(id) ON DELETE CASCADE,
        CONSTRAINT FK_search_feedback_impression
            FOREIGN KEY (impression_id) REFERENCES dbo.search_impressions(id),
        CONSTRAINT FK_search_feedback_user
            FOREIGN KEY (user_id) REFERENCES dbo.users(id) ON DELETE SET NULL
    );
END;
GO

IF OBJECT_ID(N'dbo.search_image_cache', N'U') IS NULL
BEGIN
    DECLARE @HasVectorImageCache BIT = CASE WHEN EXISTS (SELECT 1 FROM sys.types WHERE name = N'vector') THEN 1 ELSE 0 END;
    DECLARE @SearchImageEmbeddingColumnDef NVARCHAR(200) =
        CASE WHEN @HasVectorImageCache = 1 THEN N'embedding VECTOR(768) NULL,' ELSE N'embedding NVARCHAR(MAX) NULL,' END;
    DECLARE @SearchImageEmbeddingJsonConstraint NVARCHAR(MAX) =
        CASE WHEN @HasVectorImageCache = 1 THEN N'' ELSE N'CONSTRAINT CK_search_image_cache_embedding_json CHECK (embedding IS NULL OR ISJSON(embedding) = 1),' END;

    DECLARE @SearchImageSql NVARCHAR(MAX) = N'
CREATE TABLE dbo.search_image_cache (
    id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_search_image_cache PRIMARY KEY,
    filename VARCHAR(255) NULL,
    image_hash VARCHAR(64) NULL,
    ' + @SearchImageEmbeddingColumnDef + N'
    embedding_model VARCHAR(100) NOT NULL CONSTRAINT DF_search_image_cache_model DEFAULT ''google/siglip2-base-patch16-224'',
    search_count INT NOT NULL CONSTRAINT DF_search_image_cache_search_count DEFAULT 1,
    last_used DATETIME2(3) NOT NULL CONSTRAINT DF_search_image_cache_last_used DEFAULT SYSUTCDATETIME(),
    created_at DATETIME2(3) NOT NULL CONSTRAINT DF_search_image_cache_created_at DEFAULT SYSUTCDATETIME(),
    CONSTRAINT UQ_search_image_cache_hash UNIQUE (image_hash),
    ' + @SearchImageEmbeddingJsonConstraint + N'
    CONSTRAINT CK_search_image_cache_hash_len CHECK (image_hash IS NULL OR LEN(image_hash) <= 64)
);';

    EXEC sp_executesql @SearchImageSql;
END;
GO

/* ============================================================================
   Seed Defaults
   ============================================================================ */

IF NOT EXISTS (SELECT 1 FROM dbo.video_categories WHERE name = 'Oil & Gas')
    INSERT INTO dbo.video_categories (name) VALUES ('Oil & Gas');
IF NOT EXISTS (SELECT 1 FROM dbo.video_categories WHERE name = 'Maintenance')
    INSERT INTO dbo.video_categories (name) VALUES ('Maintenance');
IF NOT EXISTS (SELECT 1 FROM dbo.video_categories WHERE name = 'Installation')
    INSERT INTO dbo.video_categories (name) VALUES ('Installation');
IF NOT EXISTS (SELECT 1 FROM dbo.video_categories WHERE name = 'Operations')
    INSERT INTO dbo.video_categories (name) VALUES ('Operations');
GO

/* ============================================================================
   Performance Indexes
   ============================================================================ */

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_videos_filename' AND object_id = OBJECT_ID(N'dbo.videos'))
    CREATE INDEX idx_videos_filename ON dbo.videos(filename);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_scenes_video_id' AND object_id = OBJECT_ID(N'dbo.scenes'))
    CREATE INDEX idx_scenes_video_id ON dbo.scenes(video_id);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_scenes_time_range' AND object_id = OBJECT_ID(N'dbo.scenes'))
    CREATE INDEX idx_scenes_time_range ON dbo.scenes(video_id, start_time, end_time);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_transcript_video_id' AND object_id = OBJECT_ID(N'dbo.transcript_segments'))
    CREATE INDEX idx_transcript_video_id ON dbo.transcript_segments(video_id);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_transcript_time_range' AND object_id = OBJECT_ID(N'dbo.transcript_segments'))
    CREATE INDEX idx_transcript_time_range ON dbo.transcript_segments(video_id, start_time, end_time);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_embeddings_model' AND object_id = OBJECT_ID(N'dbo.embeddings'))
    CREATE INDEX idx_embeddings_model ON dbo.embeddings(embedding_model);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_visual_embeddings_model' AND object_id = OBJECT_ID(N'dbo.visual_embeddings'))
    CREATE INDEX idx_visual_embeddings_model ON dbo.visual_embeddings(embedding_model);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_video_embeddings_model' AND object_id = OBJECT_ID(N'dbo.video_embeddings'))
    CREATE INDEX idx_video_embeddings_model ON dbo.video_embeddings(embedding_model);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_search_requests_uuid' AND object_id = OBJECT_ID(N'dbo.search_requests'))
    CREATE INDEX idx_search_requests_uuid ON dbo.search_requests(request_uuid);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_search_requests_mode_time' AND object_id = OBJECT_ID(N'dbo.search_requests'))
    CREATE INDEX idx_search_requests_mode_time ON dbo.search_requests(search_mode, created_at);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_search_impressions_request' AND object_id = OBJECT_ID(N'dbo.search_impressions'))
    CREATE INDEX idx_search_impressions_request ON dbo.search_impressions(request_id);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_search_interactions_request' AND object_id = OBJECT_ID(N'dbo.search_interactions'))
    CREATE INDEX idx_search_interactions_request ON dbo.search_interactions(request_id);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_search_interactions_type' AND object_id = OBJECT_ID(N'dbo.search_interactions'))
    CREATE INDEX idx_search_interactions_type ON dbo.search_interactions(interaction_type);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_search_feedback_request' AND object_id = OBJECT_ID(N'dbo.search_feedback'))
    CREATE INDEX idx_search_feedback_request ON dbo.search_feedback(request_id);
GO

/* ============================================================================
   Optional Full-Text Search Index (SQL Server FTS)
   ============================================================================ */

IF FULLTEXTSERVICEPROPERTY('IsFullTextInstalled') = 1
BEGIN
    IF NOT EXISTS (SELECT 1 FROM sys.fulltext_catalogs WHERE name = N'ft_video_semantic')
        CREATE FULLTEXT CATALOG ft_video_semantic;

    IF OBJECT_ID(N'dbo.transcript_segments', N'U') IS NOT NULL
       AND NOT EXISTS (SELECT 1 FROM sys.fulltext_indexes WHERE object_id = OBJECT_ID(N'dbo.transcript_segments'))
    BEGIN
        DECLARE @TranscriptPkIndex SYSNAME =
        (
            SELECT TOP (1) i.name
            FROM sys.indexes i
            WHERE i.object_id = OBJECT_ID(N'dbo.transcript_segments')
              AND i.is_primary_key = 1
        );

        IF @TranscriptPkIndex IS NOT NULL
        BEGIN
            DECLARE @FtsSql NVARCHAR(MAX) =
                N'CREATE FULLTEXT INDEX ON dbo.transcript_segments([text] LANGUAGE 1033) ' +
                N'KEY INDEX [' + REPLACE(@TranscriptPkIndex, ']', ']]') + N'] ON ft_video_semantic;';
            EXEC sp_executesql @FtsSql;
        END;
    END;
END;
GO

/* ============================================================================
   Triggers
   ============================================================================ */

CREATE OR ALTER TRIGGER dbo.trg_videos_updated_at
ON dbo.videos
AFTER UPDATE
AS
BEGIN
    SET NOCOUNT ON;

    IF UPDATE(updated_at)
        RETURN;

    UPDATE v
    SET updated_at = SYSUTCDATETIME()
    FROM dbo.videos v
    INNER JOIN inserted i ON i.id = v.id;
END;
GO

/* ============================================================================
   Utility Functions and Procedures
   ============================================================================ */

CREATE OR ALTER FUNCTION dbo.fn_cosine_similarity_json
(
    @vector_a NVARCHAR(MAX),
    @vector_b NVARCHAR(MAX)
)
RETURNS FLOAT
AS
BEGIN
    DECLARE @dot FLOAT = 0.0;
    DECLARE @norm_a FLOAT = 0.0;
    DECLARE @norm_b FLOAT = 0.0;

    IF @vector_a IS NULL OR @vector_b IS NULL
        RETURN NULL;

    IF ISJSON(@vector_a) <> 1 OR ISJSON(@vector_b) <> 1
        RETURN NULL;

    ;WITH a AS (
        SELECT TRY_CONVERT(INT, [key]) AS idx, TRY_CONVERT(FLOAT, [value]) AS val
        FROM OPENJSON(@vector_a)
    ),
    b AS (
        SELECT TRY_CONVERT(INT, [key]) AS idx, TRY_CONVERT(FLOAT, [value]) AS val
        FROM OPENJSON(@vector_b)
    )
    SELECT
        @dot = SUM(ISNULL(a.val, 0.0) * ISNULL(b.val, 0.0)),
        @norm_a = SQRT(SUM(ISNULL(a.val, 0.0) * ISNULL(a.val, 0.0))),
        @norm_b = SQRT(SUM(ISNULL(b.val, 0.0) * ISNULL(b.val, 0.0)))
    FROM a
    FULL OUTER JOIN b ON a.idx = b.idx;

    IF @norm_a = 0.0 OR @norm_b = 0.0
        RETURN NULL;

    RETURN @dot / (@norm_a * @norm_b);
END;
GO

CREATE OR ALTER PROCEDURE dbo.cleanup_stale_visual_embeddings
AS
BEGIN
    SET NOCOUNT ON;

    DELETE ve
    FROM dbo.visual_embeddings ve
    LEFT JOIN dbo.scenes s ON s.id = ve.scene_id
    WHERE s.id IS NULL;

    SELECT @@ROWCOUNT AS deleted_count;
END;
GO

CREATE OR ALTER PROCEDURE dbo.cleanup_stale_video_embeddings
AS
BEGIN
    SET NOCOUNT ON;

    DELETE ve
    FROM dbo.video_embeddings ve
    LEFT JOIN dbo.videos v ON v.id = ve.video_id
    WHERE v.id IS NULL;

    SELECT @@ROWCOUNT AS deleted_count;
END;
GO

CREATE OR ALTER PROCEDURE dbo.cleanup_stale_embeddings
AS
BEGIN
    SET NOCOUNT ON;

    DELETE e
    FROM dbo.embeddings e
    WHERE
        (
            e.segment_id IS NOT NULL
            AND NOT EXISTS (
                SELECT 1
                FROM dbo.transcript_segments ts
                WHERE ts.id = e.segment_id
            )
        )
        OR
        (
            e.segment_id IS NULL
            AND e.scene_id IS NOT NULL
            AND NOT EXISTS (
                SELECT 1
                FROM dbo.scenes s
                WHERE s.id = e.scene_id
            )
        );

    SELECT @@ROWCOUNT AS deleted_count;
END;
GO

CREATE OR ALTER PROCEDURE dbo.clean_query_cache
AS
BEGIN
    SET NOCOUNT ON;

    DELETE qc
    FROM dbo.query_cache qc
    WHERE qc.expires_at < SYSUTCDATETIME();

    SELECT @@ROWCOUNT AS deleted_count;
END;
GO

CREATE OR ALTER PROCEDURE dbo.update_cache_stats
    @cache_hash VARCHAR(64)
AS
BEGIN
    SET NOCOUNT ON;

    UPDATE dbo.query_cache
    SET hit_count = hit_count + 1,
        last_used = SYSUTCDATETIME()
    WHERE query_hash = @cache_hash;

    SELECT @@ROWCOUNT AS updated_count;
END;
GO

CREATE OR ALTER VIEW dbo.top_searched_segments
AS
SELECT
    ts.id,
    ts.video_id,
    v.filename,
    MAX(ts.[text]) AS [text],
    ts.start_time,
    ts.end_time,
    COUNT(sq.id) AS search_count,
    MAX(sq.search_timestamp) AS last_searched
FROM dbo.transcript_segments ts
JOIN dbo.videos v ON v.id = ts.video_id
LEFT JOIN dbo.search_queries sq ON sq.top_result_id = ts.id
GROUP BY
    ts.id,
    ts.video_id,
    v.filename,
    ts.start_time,
    ts.end_time;
GO

CREATE OR ALTER PROCEDURE dbo.refresh_search_analytics
AS
BEGIN
    SET NOCOUNT ON;
    SELECT COUNT(*) AS tracked_segments FROM dbo.top_searched_segments;
END;
GO

CREATE OR ALTER PROCEDURE dbo.hybrid_search
    @query_text NVARCHAR(MAX),
    @query_embedding NVARCHAR(MAX) = NULL,
    @embedding_model VARCHAR(100) = NULL,
    @text_weight FLOAT = 0.3,
    @semantic_weight FLOAT = 0.7,
    @limit_results INT = 10
AS
BEGIN
    SET NOCOUNT ON;

    IF @limit_results IS NULL OR @limit_results < 1
        SET @limit_results = 10;

    IF @text_weight IS NULL
        SET @text_weight = 0.3;

    IF @semantic_weight IS NULL
        SET @semantic_weight = 0.7;

    DECLARE @EmbeddingType SYSNAME =
    (
        SELECT TOP (1) t.name
        FROM sys.columns c
        JOIN sys.types t ON t.user_type_id = c.user_type_id
        WHERE c.object_id = OBJECT_ID(N'dbo.embeddings')
          AND c.name = N'embedding'
    );

    DECLARE @SemanticExpr NVARCHAR(MAX) = N'NULL';
    IF @EmbeddingType IN (N'nvarchar', N'varchar', N'ntext', N'text')
       AND @query_embedding IS NOT NULL
       AND ISJSON(@query_embedding) = 1
    BEGIN
        SET @SemanticExpr = N'dbo.fn_cosine_similarity_json(e.embedding, @query_embedding)';
    END;

    DECLARE @Sql NVARCHAR(MAX) = N'
;WITH text_scores AS (
    SELECT
        ts.id,
        CAST(CASE WHEN ts.[text] LIKE ''%'' + @query_text + ''%'' THEN 1.0 ELSE 0.0 END AS FLOAT) AS text_score
    FROM dbo.transcript_segments ts
    WHERE ts.[text] LIKE ''%'' + @query_text + ''%''
),
semantic_scores AS (
    SELECT
        e.segment_id,
        ' + @SemanticExpr + N' AS semantic_score
FROM dbo.embeddings e
WHERE e.segment_id IS NOT NULL
  AND (@embedding_model IS NULL OR e.embedding_model = @embedding_model)
)
SELECT TOP (@limit_results)
    ts.id AS segment_id,
    v.id AS video_id,
    v.filename AS video_filename,
    v.file_path AS video_path,
    ts.start_time,
    ts.end_time,
    ts.[text],
    ts.[language] AS result_language,
    (COALESCE(txt.text_score, 0.0) * @text_weight +
     COALESCE(sem.semantic_score, 0.0) * @semantic_weight) AS combined_score
FROM dbo.transcript_segments ts
JOIN dbo.videos v ON v.id = ts.video_id
LEFT JOIN text_scores txt ON txt.id = ts.id
LEFT JOIN semantic_scores sem ON sem.segment_id = ts.id
WHERE txt.text_score IS NOT NULL OR sem.semantic_score IS NOT NULL
ORDER BY combined_score DESC;';

    EXEC sp_executesql
        @Sql,
        N'@query_text NVARCHAR(MAX), @query_embedding NVARCHAR(MAX), @embedding_model VARCHAR(100), @text_weight FLOAT, @semantic_weight FLOAT, @limit_results INT',
        @query_text = @query_text,
        @query_embedding = @query_embedding,
        @embedding_model = @embedding_model,
        @text_weight = @text_weight,
        @semantic_weight = @semantic_weight,
        @limit_results = @limit_results;
END;
GO

/* ============================================================================
   Verification
   ============================================================================ */

SELECT
    DB_NAME() AS database_name,
    CASE WHEN EXISTS (SELECT 1 FROM sys.types WHERE name = N'vector') THEN 'VECTOR enabled' ELSE 'VECTOR not available (JSON fallback enabled)' END AS embedding_storage_mode;
GO
