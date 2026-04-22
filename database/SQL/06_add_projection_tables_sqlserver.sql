SET NOCOUNT ON;
USE [VideoSemanticDB];
GO

/* ============================================================================
   Dual Storage Projection Tables
   - Canonical embeddings remain in dbo.embeddings / dbo.document_embeddings
   - Low-dim projected vectors are stored here for faster vector-friendly search
   ============================================================================ */

IF OBJECT_ID(N'dbo.embedding_projections', N'U') IS NULL
BEGIN
    DECLARE @HasVectorProjection BIT = CASE WHEN EXISTS (SELECT 1 FROM sys.types WHERE name = N'vector') THEN 1 ELSE 0 END;
    DECLARE @ProjectionDim INT = 1024;
    DECLARE @UseVectorProjection BIT = CASE WHEN @HasVectorProjection = 1 AND @ProjectionDim <= 1998 THEN 1 ELSE 0 END;
    DECLARE @ProjectionColDef NVARCHAR(200) =
        CASE WHEN @UseVectorProjection = 1 THEN N'projection VECTOR(1024) NOT NULL,' ELSE N'projection NVARCHAR(MAX) NOT NULL,' END;
    DECLARE @ProjectionJsonConstraint NVARCHAR(MAX) =
        CASE WHEN @UseVectorProjection = 1 THEN N'' ELSE N'CONSTRAINT CK_embedding_projections_projection_json CHECK (ISJSON(projection) = 1),' END;

    DECLARE @ProjectionSql NVARCHAR(MAX) = N'
CREATE TABLE dbo.embedding_projections (
    id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_embedding_projections PRIMARY KEY,
    embedding_id INT NOT NULL,
    segment_id INT NULL,
    scene_id INT NULL,
    ' + @ProjectionColDef + N'
    projection_dim SMALLINT NOT NULL CONSTRAINT DF_embedding_projections_dim DEFAULT 1024,
    embedding_model VARCHAR(100) NOT NULL,
    projection_method VARCHAR(50) NOT NULL CONSTRAINT DF_embedding_projections_method DEFAULT ''head_l2_norm'',
    created_at DATETIME2(3) NOT NULL CONSTRAINT DF_embedding_projections_created_at DEFAULT SYSUTCDATETIME(),
    ' + @ProjectionJsonConstraint + N'
    CONSTRAINT FK_embedding_projections_embedding
        FOREIGN KEY (embedding_id) REFERENCES dbo.embeddings(id) ON DELETE CASCADE,
    CONSTRAINT FK_embedding_projections_segment
        FOREIGN KEY (segment_id) REFERENCES dbo.transcript_segments(id) ON DELETE CASCADE,
    CONSTRAINT FK_embedding_projections_scene
        FOREIGN KEY (scene_id) REFERENCES dbo.scenes(id) ON DELETE CASCADE,
    CONSTRAINT UQ_embedding_projections_embedding UNIQUE (embedding_id, projection_dim, projection_method)
);';

    EXEC sp_executesql @ProjectionSql;
END;
GO

IF OBJECT_ID(N'dbo.document_embedding_projections', N'U') IS NULL
BEGIN
    DECLARE @HasVectorDocProjection BIT = CASE WHEN EXISTS (SELECT 1 FROM sys.types WHERE name = N'vector') THEN 1 ELSE 0 END;
    DECLARE @DocProjectionDim INT = 1024;
    DECLARE @UseVectorDocProjection BIT = CASE WHEN @HasVectorDocProjection = 1 AND @DocProjectionDim <= 1998 THEN 1 ELSE 0 END;
    DECLARE @DocProjectionColDef NVARCHAR(200) =
        CASE WHEN @UseVectorDocProjection = 1 THEN N'projection VECTOR(1024) NOT NULL,' ELSE N'projection NVARCHAR(MAX) NOT NULL,' END;
    DECLARE @DocProjectionJsonConstraint NVARCHAR(MAX) =
        CASE WHEN @UseVectorDocProjection = 1 THEN N'' ELSE N'CONSTRAINT CK_doc_embedding_projections_projection_json CHECK (ISJSON(projection) = 1),' END;

    DECLARE @DocProjectionSql NVARCHAR(MAX) = N'
CREATE TABLE dbo.document_embedding_projections (
    id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_document_embedding_projections PRIMARY KEY,
    document_embedding_id INT NOT NULL,
    chunk_id INT NOT NULL,
    ' + @DocProjectionColDef + N'
    projection_dim SMALLINT NOT NULL CONSTRAINT DF_doc_embedding_projections_dim DEFAULT 1024,
    embedding_model VARCHAR(100) NOT NULL,
    projection_method VARCHAR(50) NOT NULL CONSTRAINT DF_doc_embedding_projections_method DEFAULT ''head_l2_norm'',
    created_at DATETIME2(3) NOT NULL CONSTRAINT DF_doc_embedding_projections_created_at DEFAULT SYSUTCDATETIME(),
    ' + @DocProjectionJsonConstraint + N'
    CONSTRAINT FK_doc_embedding_projections_embedding
        FOREIGN KEY (document_embedding_id) REFERENCES dbo.document_embeddings(id) ON DELETE CASCADE,
    CONSTRAINT FK_doc_embedding_projections_chunk
        FOREIGN KEY (chunk_id) REFERENCES dbo.document_chunks(id) ON DELETE CASCADE,
    CONSTRAINT UQ_doc_embedding_projections_embedding UNIQUE (document_embedding_id, projection_dim, projection_method)
);';

    EXEC sp_executesql @DocProjectionSql;
END;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_embedding_projections_segment_model' AND object_id = OBJECT_ID(N'dbo.embedding_projections'))
    CREATE INDEX idx_embedding_projections_segment_model
    ON dbo.embedding_projections(segment_id, embedding_model, projection_dim);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_embedding_projections_scene_model' AND object_id = OBJECT_ID(N'dbo.embedding_projections'))
    CREATE INDEX idx_embedding_projections_scene_model
    ON dbo.embedding_projections(scene_id, embedding_model, projection_dim);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_doc_embedding_projections_chunk_model' AND object_id = OBJECT_ID(N'dbo.document_embedding_projections'))
    CREATE INDEX idx_doc_embedding_projections_chunk_model
    ON dbo.document_embedding_projections(chunk_id, embedding_model, projection_dim);
GO

PRINT 'Projection tables ready.';
GO
