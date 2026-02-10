# Setup Guide for Video Semantic Search Pipeline

This guide walks you through setting up the complete video semantic search system.

## Prerequisites

1. **Python 3.9+**
2. **PostgreSQL 14+** with pgvector extension
3. **CUDA** (optional, for GPU acceleration)

## Installation Steps

### Option 1: Using Docker (Recommended)

```bash
# 1. Start PostgreSQL with pgvector
docker-compose up -d

# 2. Verify database is running
docker-compose ps

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Copy and configure environment file
cp .env.example .env
# Edit .env if needed (defaults work with Docker setup)
```

### Option 2: Manual PostgreSQL Installation

#### Windows

```powershell
# 1. Download and install PostgreSQL from:
# https://www.postgresql.org/download/windows/

# 2. Install pgvector extension
# Download from: https://github.com/pgvector/pgvector/releases
# Follow installation instructions for Windows

# 3. Create database
psql -U postgres
CREATE DATABASE video_semantic_search;
\q

# 4. Apply schema
psql -U postgres -d video_semantic_search -f database/schema.sql

# 5. Install Python dependencies
pip install -r requirements.txt

# 6. Configure environment
cp .env.example .env
# Edit .env with your credentials
```

#### Linux (Ubuntu/Debian)

```bash
# 1. Install PostgreSQL
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib postgresql-server-dev-all

# 2. Install pgvector
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
cd ..

# 3. Create database
sudo -u postgres createdb video_semantic_search

# 4. Apply schema
sudo -u postgres psql -d video_semantic_search -f database/schema.sql

# 5. Install Python dependencies
pip install -r requirements.txt

# 6. Configure environment
cp .env.example .env
nano .env  # Edit with your credentials
```

## Quick Start

```bash
# Process videos, setup database, and ingest data
python quick_start.py

# Start the API server
python api/app.py
```

## Manual Workflow

### 1. Process Videos

```python
from basic_pipeline import BasicVideoPipeline

pipeline = BasicVideoPipeline(
    backend="whisper",
    model_variant={"name": "base"},
    scene_threshold=20.0
)

# Process all videos in 'videos' folder
pipeline.batch_process(
    video_folder="videos",
    force=False  # Skip already processed videos
)
```

### 2. Ingest into Database

```bash
python database/ingest.py
```

### 3. Start API Server

```bash
python api/app.py
# Or: uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test the API

Open browser: http://localhost:8000/docs

Or use curl:
```bash
curl "http://localhost:8000/search/quick?q=Omega+Alpha+well"
```

## Using Different Models

### Faster Processing (Distil-Whisper)

```python
pipeline = BasicVideoPipeline(
    backend="distil-whisper",
    model_variant={
        "name": "distil-large-v3",
        "model_id": "distil-whisper/distil-large-v3"
    }
)
```

### Better Accuracy (Whisper Large)

```python
pipeline = BasicVideoPipeline(
    backend="whisper",
    model_variant={"name": "large-v3"}
)
```

### Enhanced Timestamps (WhisperX)

```python
# Install WhisperX first: pip install whisperx
pipeline = BasicVideoPipeline(
    backend="whisperx",
    model_variant={"name": "large-v2"}
)
```

## Troubleshooting

### PostgreSQL Connection Errors

```bash
# Check if PostgreSQL is running
# Windows: Check Services
# Linux: sudo systemctl status postgresql

# Test connection
psql -U postgres -d video_semantic_search
```

### pgvector Extension Not Found

```sql
-- Connect to database
psql -U postgres -d video_semantic_search

-- Create extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify
\dx
```

### CUDA Out of Memory

```python
# Use CPU instead
pipeline = BasicVideoPipeline(device="cpu")
```

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or create new virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Common Issues

### "Database does not exist"

```bash
createdb -U postgres video_semantic_search
psql -U postgres -d video_semantic_search -f database/schema.sql
```

### "No module named 'pgvector'"

```bash
pip install pgvector
```

### "CUDA not available" but you have a GPU

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Videos not being skipped even though already processed

The pipeline checks:
1. Video fingerprint (size + modification time)
2. Model configuration
3. Existence of all output files

If any changed, it will reprocess. Use `force=False` to skip.

## Next Steps

- Read the [README.md](README.md) for detailed usage
- Check [API Documentation](http://localhost:8000/docs) after starting the server
- Explore the code in `search/semantic_search.py` for custom search logic
