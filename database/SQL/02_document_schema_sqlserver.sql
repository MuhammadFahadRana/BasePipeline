SET NOCOUNT ON;
SET ANSI_NULLS ON;
SET QUOTED_IDENTIFIER ON;
SET ANSI_PADDING ON;
SET ANSI_WARNINGS ON;
SET ARITHABORT ON;
SET CONCAT_NULL_YIELDS_NULL ON;
SET NUMERIC_ROUNDABORT OFF;
USE [VideoSemanticDB];
GO

/* ============================================================================
   Document Pipeline Tables
   Mirrors database/document_schema.sql for SQL Server Express.
   ============================================================================ */

IF OBJECT_ID(N'dbo.documents', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.documents (
        id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_documents PRIMARY KEY,
        filename NVARCHAR(500) NOT NULL,
        file_path NVARCHAR(2000) NOT NULL,
        file_type VARCHAR(20) NOT NULL,
        file_size_mb FLOAT NULL,
        total_pages INT NULL,
        extraction_method VARCHAR(20) NULL,
        ocr_model VARCHAR(100) NULL,
        language VARCHAR(10) NOT NULL CONSTRAINT DF_documents_language DEFAULT 'en',
        category_id INT NULL,
        label NVARCHAR(255) NULL,
        file_identity_hash AS CONVERT(VARCHAR(64), HASHBYTES('SHA2_256', LOWER(CONCAT(filename, N'|', file_path))), 2) PERSISTED,
        processed_at DATETIME2(3) NOT NULL CONSTRAINT DF_documents_processed_at DEFAULT SYSUTCDATETIME(),
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_documents_created_at DEFAULT SYSUTCDATETIME(),
        updated_at DATETIME2(3) NOT NULL CONSTRAINT DF_documents_updated_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT UQ_document_file UNIQUE (file_identity_hash),
        CONSTRAINT FK_documents_category
            FOREIGN KEY (category_id) REFERENCES dbo.video_categories(id) ON DELETE SET NULL
    );
END;
GO

IF OBJECT_ID(N'dbo.document_chunks', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.document_chunks (
        id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_document_chunks PRIMARY KEY,
        document_id INT NOT NULL,
        chunk_index INT NOT NULL,
        page_number INT NULL,
        section_heading NVARCHAR(MAX) NULL,
        [text] NVARCHAR(MAX) NOT NULL,
        summary NVARCHAR(MAX) NULL,
        ocr_confidence FLOAT NULL,
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_document_chunks_created_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT FK_document_chunks_document
            FOREIGN KEY (document_id) REFERENCES dbo.documents(id) ON DELETE CASCADE,
        CONSTRAINT UQ_document_chunk UNIQUE (document_id, chunk_index)
    );
END;
GO

IF OBJECT_ID(N'dbo.document_embeddings', N'U') IS NULL
BEGIN
    DECLARE @HasVectorDoc BIT = CASE WHEN EXISTS (SELECT 1 FROM sys.types WHERE name = N'vector') THEN 1 ELSE 0 END;
    DECLARE @DocEmbeddingDim INT = 4096;
    DECLARE @UseVectorDoc BIT = CASE WHEN @HasVectorDoc = 1 AND @DocEmbeddingDim <= 1998 THEN 1 ELSE 0 END;
    DECLARE @DocEmbeddingColumnDef NVARCHAR(200) =
        CASE WHEN @UseVectorDoc = 1 THEN N'embedding VECTOR(4096) NULL,' ELSE N'embedding NVARCHAR(MAX) NULL,' END;
    DECLARE @DocEmbeddingJsonConstraint NVARCHAR(MAX) =
        CASE WHEN @HasVectorDoc = 1 THEN N'' ELSE N'CONSTRAINT CK_document_embeddings_embedding_json CHECK (embedding IS NULL OR ISJSON(embedding) = 1),' END;

    DECLARE @DocEmbSql NVARCHAR(MAX) = N'
CREATE TABLE dbo.document_embeddings (
    id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_document_embeddings PRIMARY KEY,
    chunk_id INT NOT NULL,
    ' + @DocEmbeddingColumnDef + N'
    embedding_model VARCHAR(100) NOT NULL CONSTRAINT DF_document_embeddings_model DEFAULT ''Qwen/Qwen3-Embedding-0.6B'',
    created_at DATETIME2(3) NOT NULL CONSTRAINT DF_document_embeddings_created_at DEFAULT SYSUTCDATETIME(),
    ' + @DocEmbeddingJsonConstraint + N'
    CONSTRAINT FK_document_embeddings_chunk
        FOREIGN KEY (chunk_id) REFERENCES dbo.document_chunks(id) ON DELETE CASCADE,
    CONSTRAINT UQ_doc_chunk_embedding UNIQUE (chunk_id, embedding_model)
);';

    EXEC sp_executesql @DocEmbSql;
END;
GO

/* ============================================================================
   Indexes
   ============================================================================ */

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_documents_filename' AND object_id = OBJECT_ID(N'dbo.documents'))
    CREATE INDEX idx_documents_filename ON dbo.documents(filename);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_doc_chunks_document_id' AND object_id = OBJECT_ID(N'dbo.document_chunks'))
    CREATE INDEX idx_doc_chunks_document_id ON dbo.document_chunks(document_id);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_doc_embeddings_model' AND object_id = OBJECT_ID(N'dbo.document_embeddings'))
    CREATE INDEX idx_doc_embeddings_model ON dbo.document_embeddings(embedding_model);
GO

/* ============================================================================
   Optional Full-Text Search Index
   ============================================================================ */

IF FULLTEXTSERVICEPROPERTY('IsFullTextInstalled') = 1
BEGIN
    IF NOT EXISTS (SELECT 1 FROM sys.fulltext_catalogs WHERE name = N'ft_video_semantic')
        CREATE FULLTEXT CATALOG ft_video_semantic;

    IF OBJECT_ID(N'dbo.document_chunks', N'U') IS NOT NULL
       AND NOT EXISTS (SELECT 1 FROM sys.fulltext_indexes WHERE object_id = OBJECT_ID(N'dbo.document_chunks'))
    BEGIN
        DECLARE @DocChunkPkIndex SYSNAME =
        (
            SELECT TOP (1) i.name
            FROM sys.indexes i
            WHERE i.object_id = OBJECT_ID(N'dbo.document_chunks')
              AND i.is_primary_key = 1
        );

        IF @DocChunkPkIndex IS NOT NULL
        BEGIN
            DECLARE @DocFtsSql NVARCHAR(MAX) =
                N'CREATE FULLTEXT INDEX ON dbo.document_chunks([text] LANGUAGE 1033) ' +
                N'KEY INDEX [' + REPLACE(@DocChunkPkIndex, ']', ']]') + N'] ON ft_video_semantic;';
            EXEC sp_executesql @DocFtsSql;
        END;
    END;
END;
GO

/* ============================================================================
   Trigger
   ============================================================================ */

CREATE OR ALTER TRIGGER dbo.trg_documents_updated_at
ON dbo.documents
AFTER UPDATE
AS
BEGIN
    SET NOCOUNT ON;

    IF UPDATE(updated_at)
        RETURN;

    UPDATE d
    SET updated_at = SYSUTCDATETIME()
    FROM dbo.documents d
    INNER JOIN inserted i ON i.id = d.id;
END;
GO
