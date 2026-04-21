SET NOCOUNT ON;
USE [VideoSemanticDB];
GO

/* ============================================================================
   ATLAS Training Infrastructure (SQL Server)
   ============================================================================ */

IF OBJECT_ID(N'dbo.relevance_judgments', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.relevance_judgments (
        id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_relevance_judgments PRIMARY KEY,
        query_id VARCHAR(50) NOT NULL,
        query_text NVARCHAR(MAX) NOT NULL,
        query_type VARCHAR(30) NOT NULL CONSTRAINT DF_relevance_judgments_query_type DEFAULT 'natural_search',
        [language] VARCHAR(10) NOT NULL CONSTRAINT DF_relevance_judgments_language DEFAULT 'en',
        video_id INT NOT NULL,
        segment_id INT NULL,
        scene_id INT NULL,
        start_time FLOAT NOT NULL,
        end_time FLOAT NOT NULL,
        relevance INT NOT NULL CONSTRAINT DF_relevance_judgments_relevance DEFAULT 0,
        notes NVARCHAR(MAX) NULL,
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_relevance_judgments_created_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT FK_rj_video FOREIGN KEY (video_id) REFERENCES dbo.videos(id),
        CONSTRAINT FK_rj_segment FOREIGN KEY (segment_id) REFERENCES dbo.transcript_segments(id),
        CONSTRAINT FK_rj_scene FOREIGN KEY (scene_id) REFERENCES dbo.scenes(id)
    );
END;
GO

IF OBJECT_ID(N'dbo.model_runs', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.model_runs (
        id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_model_runs PRIMARY KEY,
        run_name VARCHAR(100) NOT NULL,
        embedding_model VARCHAR(100) NOT NULL,
        adapter_path NVARCHAR(MAX) NULL,
        asr_model VARCHAR(100) NULL,
        scene_config NVARCHAR(MAX) NULL,
        reranker_model VARCHAR(100) NULL,
        enrichment_mode VARCHAR(50) NULL,
        recall_at_1 FLOAT NULL,
        recall_at_5 FLOAT NULL,
        recall_at_10 FLOAT NULL,
        mrr FLOAT NULL,
        ndcg FLOAT NULL,
        timestamp_iou FLOAT NULL,
        hit_5s FLOAT NULL,
        hit_10s FLOAT NULL,
        median_abs_err_s FLOAT NULL,
        train_samples INT NULL,
        eval_samples INT NULL,
        notes NVARCHAR(MAX) NULL,
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_model_runs_created_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT CK_model_runs_scene_config_json CHECK (scene_config IS NULL OR ISJSON(scene_config) = 1)
    );
END;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_rj_query_id' AND object_id = OBJECT_ID(N'dbo.relevance_judgments'))
    CREATE INDEX IX_rj_query_id ON dbo.relevance_judgments(query_id);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_rj_video_id' AND object_id = OBJECT_ID(N'dbo.relevance_judgments'))
    CREATE INDEX IX_rj_video_id ON dbo.relevance_judgments(video_id);
GO
