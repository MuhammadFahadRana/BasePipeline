# Video Semantic Search Pipeline

A comprehensive video processing and semantic search system that transcribes videos, detects scenes, and enables natural language querying of video content with typo tolerance.

## Features

- **Multi-Model Transcription**: Support for Whisper, WhisperX, Distil-Whisper, Wav2Vec2, and NeMo
- **Scene Detection**: Automatic video segmentation and keyframe extraction
- **Semantic Search**: Query videos using natural language with typo/fuzzy matching
- **Database Storage**: PostgreSQL with pgvector for efficient similarity search
- **REST API**: FastAPI-based API for programmatic access
- **Smart Caching**: Avoid reprocessing already-transcribed videos

## Quick Start

> **📌 Database Already Set Up?** → See **[QUICK_START.md](QUICK_START.md)** for daily usage instructions!

This section is for **first-time setup only**. If your database is already configured, skip to [QUICK_START.md](QUICK_START.md).

---

### Option A: Using Docker (Recommended for Windows)

**Prerequisites:**
- Docker Desktop installed and running

```powershell
# 1. Start PostgreSQL with pgvector
docker-compose up -d

# 2. Wait for database to initialize (8 seconds)
timeout /t 8

# 3. Test connection
python -c "from database.config import test_connection; test_connection()"
# Expected: ✓ Connected to PostgreSQL: PostgreSQL 16.11...

# 4. Initialize database schema
python -c "from database.config import init_db; init_db()"
# Expected: ✓ Database initialized successfully

# 5. Verify tables created
docker exec video_search_db psql -U postgres -d video_semantic_search -c "\dt"
```

**✅ Database is now ready!** Proceed to Step 3 below.

---

### Option B: Using Local PostgreSQL (Linux/Mac)

**Prerequisites:**
- PostgreSQL 12+ with pgvector extension installed

```bash
# Install PostgreSQL with pgvector
# Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib

# Install pgvector extension
# See: https://github.com/pgvector/pgvector#installation

# Create database
createdb video_semantic_search

# Or using psql:
psql -U postgres
CREATE DATABASE video_semantic_search;
CREATE EXTENSION vector;
\q

# Apply schema
psql -U postgres -d video_semantic_search -f database/schema.sql
```

### 3. Configure Environment

The `.env` file should already be configured. Verify it contains:

```env
DB_HOST=localhost      # Important: Use 'localhost' for Docker on Windows
DB_PORT=5432
DB_NAME=video_semantic_search
DB_USER=postgres
DB_PASSWORD=postgres
```

### 4. Install Python Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt
```

### 5. Process Videos

```python
from basic_pipeline import BasicVideoPipeline

# Initialize with Whisper base model
pipeline = BasicVideoPipeline(
    backend="whisper",
    model_variant={"name": "base"},
    scene_threshold=20.0
)

# Process all videos in the videos folder
pipeline.batch_process(
    video_folder="videos",
    output_base="processed",
    force=False  # Skip already processed videos
)
```

### 6. Ingest into Database

```bash
# Ingest all processed videos and generate embeddings
python database/ingest.py
```

### 7. Start the API

```bash
# Start the FastAPI server
python api/app.py

# Or using uvicorn directly:
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

### 8. Search Videos

```bash
# Using curl
curl "http://localhost:8000/search/quick?q=where+Omega+Alpha+well+is+discussed&limit=5"

# Using Python
import requests

response = requests.post("http://localhost:8000/search", json={
    "query": "where Omega Alpha well is discussed",
    "top_k": 5
})

results = response.json()
for result in results["results"]:
    print(f"{result['video_filename']} at {result['timestamp']}: {result['text']}")
```

## Example Query

**Query:** "where Omega Alpha well is discussed"

**Result:**
```json
{
  "video_filename": "AkerBP 1.mp4",
  "timestamp": "00:00:37",
  "start_time": 37.0,
  "end_time": 41.0,
  "text": "We're currently drilling the fourth lateral on the Umega Alpha well in the Idrissel area...",
  "score": 0.8542,
  "match_type": "semantic"
}
```

**Note:** The system automatically handles typos like "Umega" → "Omega" and "Alfa" → "Alpha"

## Project Structure

```
BasePipeline/
├── api/
│   └── app.py                    # FastAPI application
├── database/
│   ├── schema.sql                # PostgreSQL schema
│   ├── models.py                 # SQLAlchemy models
│   ├── config.py                 # Database configuration
│   └── ingest.py                 # Data ingestion script
├── embeddings/
│   └── text_embeddings.py        # Embedding generation
├── search/
│   └── semantic_search.py        # Search engine with typo handling
├── videos/                       # Input videos
├── processed/                    # Processed outputs
│   ├── transcripts/              # Model-specific transcripts
│   ├── scenes/                   # Scene detection results
│   └── results/                  # Combined results
├── basic_pipeline.py             # Main video processing pipeline
├── transcribe_all.py             # Multi-model transcriber
├── scene_detector.py             # Scene detection
└── requirements.txt              # Python dependencies
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /videos` - List all videos
- `POST /search` - Semantic search (full options)
- `GET /search/quick` - Quick search (simple GET)
- `GET /search/exact` - Exact phrase search

## Models Supported

### ASR Models
- **Whisper** (OpenAI): tiny, base, small, medium, large, large-v2, large-v3
- **WhisperX**: Enhanced Whisper with better timestamps
- **Distil-Whisper**: 6x faster than standard Whisper
- **Wav2Vec2** (Meta): English and multilingual models
- **NeMo Canary** (NVIDIA): Enterprise-grade ASR

### Embedding Model
- **all-MiniLM-L6-v2** (default): 384-dim sentence embeddings

## Advanced Usage

### Search with Custom Weights

```python
from database.config import SessionLocal
from search.semantic_search import SemanticSearchEngine

db = SessionLocal()
search_engine = SemanticSearchEngine(db)

results = search_engine.search(
    query="drilling techniques",
    top_k=10,
    semantic_weight=0.7,  # Favor semantic understanding
    text_weight=0.3,      # Lower weight for exact text match
    min_score=0.5,        # Minimum threshold
    video_filter="AkerBP 1.mp4"  # Search only in this video
)

for r in results:
    print(f"{r.timestamp}: {r.text}")
```

### Process with Different Models

```python
# Use Distil-Whisper for faster processing
pipeline = BasicVideoPipeline(
    backend="distil-whisper",
    model_variant={
        "name": "distil-large-v3",
        "model_id": "distil-whisper/distil-large-v3"
    }
)

# Process single video
pipeline.process_video("videos/AkerBP 1.mp4")
```

## Troubleshooting

### Database Connection Errors

**If using Docker (Windows):**
```powershell
# 1. Ensure Docker Desktop is running
docker --version

# 2. Check if container is running
docker ps

# 3. Start container if stopped
docker-compose up -d

# 4. Check container logs
docker logs video_search_db --tail 30

# 5. Test connection
python -c "from database.config import test_connection; test_connection()"
```

**Common Issues:**
- **Error**: `password authentication failed`
  - **Fix**: Ensure `.env` has `DB_HOST=localhost` (NOT `127.0.0.1`)
  
- **Error**: `port 5432 already in use`
  - **Fix**: Stop local PostgreSQL service or change port in `docker-compose.yml`
  
- **Error**: `cannot connect to Docker daemon`
  - **Fix**: Start Docker Desktop

**If using Local PostgreSQL (Linux/Mac):**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Start PostgreSQL
sudo systemctl start postgresql

# Test connection
psql -U postgres -d video_semantic_search
```

### pgvector Not Found
```powershell
# For Docker - pgvector is pre-installed in the image
docker exec video_search_db psql -U postgres -d video_semantic_search -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Verify installation
docker exec video_search_db psql -U postgres -d video_semantic_search -c "\dx"
```

### CUDA Out of Memory
```python
# Use CPU instead
pipeline = BasicVideoPipeline(device="cpu")
```

## License

MIT License

## Contributors

Your Name
