-- ──────────────────────────────────────────────
-- ATLAS Training Infrastructure - Schema Additions (PostgreSQL)
-- Run against the video_semantic_search database
-- ──────────────────────────────────────────────

/* ---------- relevance_judgments ---------- */
-- Stores gold-standard query→segment pairs for training and evaluation.
-- Each row links a search query to a specific segment with a relevance grade.
CREATE TABLE IF NOT EXISTS relevance_judgments (
    id            SERIAL PRIMARY KEY,
    query_id      VARCHAR(50) NOT NULL,          -- e.g. "q_001"
    query_text    TEXT NOT NULL,                 -- the search query
    query_type    VARCHAR(30) NOT NULL DEFAULT 'natural_search',
                  -- exact_keyword | natural_search | paraphrase | cross_lingual | visual_text
    language      VARCHAR(10) NOT NULL DEFAULT 'en',
    video_id      INTEGER NOT NULL,
    segment_id    INTEGER NULL,                  -- FK to transcript_segments
    scene_id      INTEGER NULL,                  -- FK to scenes
    start_time    FLOAT NOT NULL,
    end_time      FLOAT NOT NULL,
    relevance     INTEGER NOT NULL DEFAULT 0,
                  -- 0 = irrelevant, 1 = marginal, 2 = relevant, 3 = exact answer
    notes         TEXT NULL,
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT FK_rj_video FOREIGN KEY (video_id) REFERENCES videos(id),
    CONSTRAINT FK_rj_segment FOREIGN KEY (segment_id) REFERENCES transcript_segments(id),
    CONSTRAINT FK_rj_scene FOREIGN KEY (scene_id) REFERENCES scenes(id)
);

CREATE INDEX IF NOT EXISTS IX_rj_query_id ON relevance_judgments(query_id);
CREATE INDEX IF NOT EXISTS IX_rj_video_id ON relevance_judgments(video_id);


/* ---------- model_runs ---------- */
-- Tracks experiment runs: which models, configs, and resulting metrics.
-- Invaluable for thesis evaluation tables.
CREATE TABLE IF NOT EXISTS model_runs (
    id                SERIAL PRIMARY KEY,
    run_name          VARCHAR(100) NOT NULL,       -- descriptive label, e.g. "baseline_v1"
    embedding_model   VARCHAR(100) NOT NULL,       -- e.g. "Qwen/Qwen3-Embedding-0.6B"
    adapter_path      TEXT NULL,                   -- path to LoRA adapter if fine-tuned
    asr_model         VARCHAR(100) NULL,           -- e.g. "Whisper-Large-v3"
    scene_config      JSONB NULL,                  -- JSON blob of scene detection config
    reranker_model    VARCHAR(100) NULL,           -- e.g. "BAAI/bge-reranker-v2-m3"
    enrichment_mode   VARCHAR(50) NULL,            -- transcript_only | +ocr | +caption+ocr
    recall_at_1       FLOAT NULL,
    recall_at_5       FLOAT NULL,
    recall_at_10      FLOAT NULL,
    mrr               FLOAT NULL,
    ndcg              FLOAT NULL,
    timestamp_iou     FLOAT NULL,
    hit_5s            FLOAT NULL,                  -- % queries hitting within ±5 seconds
    hit_10s           FLOAT NULL,                  -- % queries hitting within ±10 seconds
    median_abs_err_s  FLOAT NULL,                  -- median absolute timestamp error
    train_samples     INTEGER NULL,
    eval_samples      INTEGER NULL,
    notes             TEXT NULL,
    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
