USE VideoSemantic;
GO

/* ---------- videos ---------- */
CREATE TABLE dbo.videos (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    filename        VARCHAR(255) NOT NULL UNIQUE,
    file_path       NVARCHAR(MAX) NOT NULL,
    file_size_mb    FLOAT NULL,
    duration_seconds FLOAT NULL,
    whisper_model   VARCHAR(50) NULL,
    scene_threshold FLOAT NULL,
    processed_at    DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    video_fingerprint NVARCHAR(MAX) NULL, -- JSON
    created_at      DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    updated_at      DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    CONSTRAINT CK_videos_video_fingerprint_json
        CHECK (video_fingerprint IS NULL OR ISJSON(video_fingerprint) = 1)
);
GO

/* ---------- scenes ---------- */
CREATE TABLE dbo.scenes (
    id           INT IDENTITY(1,1) PRIMARY KEY,
    video_id     INT NOT NULL,
    scene_id     INT NOT NULL,
    start_time   FLOAT NOT NULL,
    end_time     FLOAT NOT NULL,
    duration     FLOAT NOT NULL,
    start_frame  INT NULL,
    end_frame    INT NULL,
    keyframe_path NVARCHAR(MAX) NULL,
    ocr_text      NVARCHAR(MAX) NULL,
    object_labels NVARCHAR(MAX) NULL, -- JSON
    caption       NVARCHAR(MAX) NULL,
    created_at   DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    CONSTRAINT CK_scenes_object_labels_json CHECK (object_labels IS NULL OR ISJSON(object_labels) = 1),
    CONSTRAINT FK_scenes_videos
        FOREIGN KEY (video_id) REFERENCES dbo.videos(id) ON DELETE CASCADE,
    CONSTRAINT UQ_scenes_video_scene UNIQUE (video_id, scene_id)
);
GO

/* ---------- transcript_segments ---------- */
CREATE TABLE dbo.transcript_segments (
    id            INT IDENTITY(1,1) PRIMARY KEY,
    video_id      INT NOT NULL,
    scene_id      INT NULL,
    segment_index INT NOT NULL,
    start_time    FLOAT NOT NULL,
    end_time      FLOAT NOT NULL,
    [text]        NVARCHAR(MAX) NOT NULL,
    [language]    VARCHAR(10) NOT NULL DEFAULT 'en',
    created_at    DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    CONSTRAINT FK_transcript_videos
        FOREIGN KEY (video_id) REFERENCES dbo.videos(id) ON DELETE CASCADE,
    CONSTRAINT FK_transcript_scenes
        FOREIGN KEY (scene_id) REFERENCES dbo.scenes(id) ON DELETE SET NULL,
    CONSTRAINT UQ_transcript_video_segment UNIQUE (video_id, segment_index)
);
GO

/* ---------- embeddings (TEXT) ---------- */
CREATE TABLE dbo.embeddings (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    segment_id      INT NULL,
    scene_id        INT NULL,
    embedding       VECTOR(1024) NULL,
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'Qwen/Qwen3-Embedding-0.6B',
    created_at      DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    CONSTRAINT FK_embeddings_segment
        FOREIGN KEY (segment_id) REFERENCES dbo.transcript_segments(id) ON DELETE CASCADE,
    CONSTRAINT FK_embeddings_scene
        FOREIGN KEY (scene_id) REFERENCES dbo.scenes(id) ON DELETE CASCADE,
    CONSTRAINT UQ_embeddings_source UNIQUE (segment_id, scene_id, embedding_model)
);
GO

/* ---------- visual_embeddings ---------- */
CREATE TABLE dbo.visual_embeddings (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    scene_id        INT NOT NULL,
    keyframe_path   NVARCHAR(MAX) NOT NULL,
    embedding       VECTOR(768) NULL,
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'google/siglip-base-patch16-224',
    created_at      DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    CONSTRAINT FK_visual_scene
        FOREIGN KEY (scene_id) REFERENCES dbo.scenes(id) ON DELETE CASCADE
);
GO

/* ---------- query_cache ---------- */
CREATE TABLE dbo.query_cache (
    id            INT IDENTITY(1,1) PRIMARY KEY,
    query_text    NVARCHAR(MAX) NOT NULL,
    query_hash    VARCHAR(64) NOT NULL UNIQUE,
    query_params  NVARCHAR(MAX) NULL,
    cached_results NVARCHAR(MAX) NULL,
    hit_count     INT NOT NULL DEFAULT 1,
    last_used     DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    expires_at    DATETIME2(3) NOT NULL,
    created_at    DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    CONSTRAINT CK_query_cache_params_json CHECK (query_params IS NULL OR ISJSON(query_params) = 1),
    CONSTRAINT CK_query_cache_results_json CHECK (cached_results IS NULL OR ISJSON(cached_results) = 1)
);
GO

/* ---------- search_queries ---------- */
CREATE TABLE dbo.search_queries (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    query_text       NVARCHAR(MAX) NOT NULL,
    query_embedding  VECTOR(1024) NULL,
    search_type      VARCHAR(20) NOT NULL DEFAULT 'text',
    results_count    INT NULL,
    top_result_id    INT NULL,
    search_timestamp DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    CONSTRAINT FK_search_top_result
        FOREIGN KEY (top_result_id) REFERENCES dbo.transcript_segments(id)
);
GO

/* ---------- search_image_cache ---------- */
CREATE TABLE dbo.search_image_cache (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    filename        VARCHAR(255) NULL,
    image_hash      VARCHAR(64) NULL UNIQUE,
    embedding       VECTOR(768) NULL,
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'google/siglip-base-patch16-224',
    search_count    INT NOT NULL DEFAULT 1,
    last_used       DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    created_at      DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME()
);
GO

/* ---------- indexes ---------- */
CREATE INDEX IX_scenes_video_id ON dbo.scenes(video_id);
CREATE INDEX IX_scenes_time_range ON dbo.scenes(video_id, start_time, end_time);

CREATE INDEX IX_transcript_video_id ON dbo.transcript_segments(video_id);
CREATE INDEX IX_transcript_time_range ON dbo.transcript_segments(video_id, start_time, end_time);

CREATE INDEX IX_embeddings_model ON dbo.embeddings(embedding_model);
CREATE INDEX IX_visual_embeddings_model ON dbo.visual_embeddings(embedding_model);
GO

/* ---------- updated_at trigger ---------- */
CREATE OR ALTER TRIGGER dbo.trg_videos_updated_at
ON dbo.videos
AFTER UPDATE
AS
BEGIN
  SET NOCOUNT ON;
  UPDATE v
    SET updated_at = SYSUTCDATETIME()
  FROM dbo.videos v
  INNER JOIN inserted i ON i.id = v.id;
END;
GO
