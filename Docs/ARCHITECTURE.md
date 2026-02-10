# Video Semantic Search Pipeline - Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           VIDEO INPUT                                   │
│                     (videos/AkerBP 1.mp4, etc.)                         │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      PROCESSING PIPELINE                               │
│  ┌────────────────────────────────────────────────────────────────┐    │ 
│  │  basic_pipeline.py                                             │    │
│  │  ┌──────────────────┐  ┌─────────────────┐                     │    │
│  │  │  Transcription   │  │ Scene Detection │                     │    │
│  │  │ (transcribe_all) │  │ (scene_detector)│                     │    │
│  │  │                  │  │                 │                     │    │
│  │  │ • Whisper        │  │ • Content       │                     │    │
│  │  │ • WhisperX       │  │   Detection     │                     │    │
│  │  │ • Distil-Whisper │  │ • Keyframes     │                     │    │
│  │  │ • Wav2Vec2       │  │ • Timestamps    │                     │    │
│  │  │ • NeMo           │  │                 │                     │    │
│  │  └────────┬─────────┘  └────────┬────────┘                     │    │
│  │           │                     │                              │    │
│  │           └──────────┬──────────┘                              │    │
│  │                      │                                         │    │
│  │                      ▼                                         │    │
│  │             ┌─────────────────┐                                │    │
│  │             │ Alignment       │                                │    │
│  │             │ & Results       │                                │    │
│  │             └────────┬────────┘                                │    │
│  └─────────────────────┼──────────────────────────────────────────┘    │
└────────────────────────┼───────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT FILES                                   │
│  processed/                                                             │
│  ├── transcripts/{ModelName}/{VideoName}/                               │
│  │   ├── transcript.json                                                │
│  │   └── transcript.txt                                                 │
│  ├── scenes/{VideoName}/                                                │
│  │   ├── scenes.json                                                    │
│  │   └── keyframes/*.jpg                                                │
│  └── results/{VideoName}/                                               │
│      ├── results.json                                                   │
│      ├── manifest.json                                                  │
│      └── report.html                                                    │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA INGESTION                                     │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  database/ingest.py                                            │     │
│  │  ┌──────────────────┐  ┌─────────────────┐                     │     │
│  │  │  Load Results    │  │  Generate       │                     │     │
│  │  │  Files           │─▶│  Embeddings     │                    │     │
│  │  │                  │  │  (sentence-     │                     │     │
│  │  │                  │  │  transformers)  │                     │     │
│  │  └──────────────────┘  └────────┬────────┘                     │     │
│  └─────────────────────────────────┼──────────────────────────────┘     │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATABASE LAYER                                     │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │  PostgreSQL + pgvector                                        │      │
│  │  ┌───────────┐ ┌─────────┐ ┌──────────────┐ ┌────────────┐    │      │
│  │  │  videos   │ │ scenes  │ │ transcript_  │ │ embeddings │    │      │
│  │  │           │ │         │ │  segments    │ │            │    │      │
│  │  │ • id      │ │ • id    │ │ • id         │ │ • id       │    │      │
│  │  │ • filename│ │ • video │ │ • video_id   │ │ • segment  │    │      │
│  │  │ • model   │ │ • start │ │ • scene_id   │ │ • vector   │    │      │
│  │  │ • fingerp.│ │ • end   │ │ • start_time │ │   (384dim) │    │      │
│  │  └───────────┘ └─────────┘ │ • end_time   │ │ • model    │    │      │
│  │                             │ • text       │ └────────────┘   │      │
│  │                             └──────────────┘                  │      │
│  │                                                               │      │
│  │  Indexes:                                                     │      │
│  │  • HNSW vector index (fast similarity search)                 │      │
│  │  • GIN full-text search index                                 │      │
│  │  • B-tree indexes on foreign keys                             │      │
│  └───────────────────────────────────────────────────────────────┘      │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      SEARCH ENGINE                                     │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  search/semantic_search.py                                     │    │
│  │  ┌──────────────────┐  ┌─────────────────┐                     │    │
│  │  │  Typo Correction │  │  Query Embedding│                     │    │
│  │  │  "Umega" → "Omega"│─▶│  Generation     │                   │    │
│  │  └──────────────────┘  └────────┬────────┘                    │    │
│  │                                 │                             │    │
│  │                    ┌────────────┴────────────┐                │    │
│  │                    ▼                         ▼                │    │
│  │         ┌───────────────────┐    ┌───────────────────┐        │    │
│  │         │ Semantic Search   │    │ Fuzzy Text Search │        │    │
│  │         │ (vector similarity)│    │ (PostgreSQL FTS)  │       │    │
│  │         │                   │    │                   │        │    │
│  │         │ Weight: 70%       │    │ Weight: 30%       │        │    │
│  │         └──────────┬────────┘    └─────────┬─────────┘        │    │
│  │                    │                       │                  │    │
│  │                    └───────────┬───────────┘                  │    │
│  │                                ▼                              │    │
│  │                      ┌──────────────────┐                     │    │
│  │                      │ Hybrid Ranking   │                     │    │
│  │                      │ & Deduplication  │                     │    │
│  │                      └────────┬─────────┘                     │    │
│  └───────────────────────────────┼───────────────────────────────┘    │
└──────────────────────────────────┼────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          REST API                                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  api/app.py (FastAPI)                                           │    │
│  │  ┌───────────────────────────────────────────────────────────┐  │    │
│  │  │  Endpoints:                                               │  │    │
│  │  │  • GET  /                    - API info                   │  │    │
│  │  │  • GET  /health              - Health check               │  │    │
│  │  │  • GET  /videos              - List videos                │  │    │
│  │  │  • POST /search              - Semantic search (full)     │  │    │
│  │  │  • GET  /search/quick        - Quick search               │  │    │
│  │  │  • GET  /search/exact        - Exact phrase search        │  │    │
│  │  └───────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────────────────────┐
│                           CLIENTS                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐     │
│  │  Web UI      │  │  curl/wget   │  │  Python/JavaScript       │     │
│  │  (Browser)   │  │  (CLI)       │  │  Applications            │     │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘     │
└───────────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Video Input** → Videos placed in `videos/` folder
2. **Processing** → `basic_pipeline.py` transcribes and detects scenes
3. **Output** → Results saved to `processed/` directory structure
4. **Ingestion** → `database/ingest.py` loads into PostgreSQL
5. **Embedding** → Text segments converted to 384-dim vectors
6. **Storage** → Data stored in normalized database with vector index
7. **Search** → Queries processed with typo correction + hybrid search
8. **API** → FastAPI serves results via REST endpoints
9. **Client** → Users query via HTTP requests

## Key Technologies

| Layer | Technology |
|-------|------------|
| ASR Models | Whisper, WhisperX, Distil-Whisper, Wav2Vec2, NeMo |
| Scene Detection | PySceneDetect (Content Detector) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Database | PostgreSQL 14+ |
| Vector Search | pgvector (HNSW index) |
| ORM | SQLAlchemy 2.0 |
| API Framework | FastAPI |
| Web Server | Uvicorn |

## Performance Features

- **Smart Caching**: Avoid reprocessing videos
- **Batch Processing**: Process multiple files efficiently
- **Vector Indexing**: HNSW for O(log N) similarity search
- **Connection Pooling**: Reuse database connections
- **Lazy Loading**: Load embeddings model only when needed
- **Normalized Embeddings**: L2 normalization for faster cosine similarity

## Security Considerations

- Environment variables for sensitive config (`.env`)
- Prepared statements (SQLAlchemy) prevent SQL injection
- CORS middleware configurable for production
- Input validation via Pydantic models
- Connection limits and timeouts configured

## Scalability

Current setup handles:
- ✅ Thousands of videos
- ✅ Millions of transcript segments
- ✅ Concurrent search requests
- ✅ Real-time query response (<500ms typical)

For larger scale:
- Add Redis caching layer
- Use read replicas for database
- Deploy API behind load balancer
- Use Celery for async video processing
