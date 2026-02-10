# 🚀 Quick Start Guide - Video Semantic Search Pipeline

**Database Status:** ✅ Already created and configured!

This guide is for **starting the application** when the database has already been set up.

---

## Prerequisites Check

Before starting, ensure these are running:

### 1. Docker Desktop
- **Status**: Must be running
- **Check**: Look for Docker whale icon in system tray (Windows)
- **Start**: Open Docker Desktop application

### 2. PostgreSQL Container
- **Status**: Should be running in Docker
- **Check**: Run `docker ps` - you should see `video_search_db` container
- **Start**: See commands below

---

## 🏁 Starting the Application

### Step 1: Start Docker PostgreSQL Database

```powershell
# Navigate to project directory
cd "d:\Education\UiS\MS Computer Science\Spring 2026, 4th\Code\BasePipeline"

# Start the database container (if not already running)
docker-compose up -d

# Wait 5 seconds for database to be ready
timeout /t 5

# Verify database is running
docker ps
# Output: video_search_db running

# Test database connection
python -c "from database.config import test_connection; test_connection()"
# Expected: ✓ Connected to PostgreSQL: PostgreSQL 16.11...
```

**If the database container doesn't exist**, recreate it:
```powershell
docker-compose down -v
docker-compose up -d
timeout /t 8
python -c "from database.config import init_db; init_db()"
```

### Step 2: Verify Database Tables

```powershell
# List all tables in the database
docker exec video_search_db psql -U postgres -d video_semantic_search -c "\dt"

# Expected output: 5 tables
# - embeddings
# - scenes  
# - search_queries
# - transcript_segments
# - videos
```

### Step 3: Process Videos (Optional - if you have new videos)

```powershell
# Process all videos in the videos/ folder
python basic_pipeline.py

# Or process with a specific model
python transcribe_all.py --model whisper --variant base
```

### Step 4: Ingest Data into Database

```powershell
# Ingest processed videos and generate embeddings
python database/ingest.py

# This will:
# - Import video metadata
# - Import transcript segments
# - Generate and store embeddings for semantic search
```

### Step 5: Start the API Server

```powershell
# Start the FastAPI server
python api/app.py

# Alternative: Use uvicorn directly
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: **http://localhost:8000**
- **Frontend UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Step 6: Query Videos

Open a new terminal/PowerShell window and try:

```powershell
# Quick search using GET request
curl "http://localhost:8000/search/quick?q=drilling+techniques&limit=5"

# Or use Python
python -c "import requests; r = requests.get('http://localhost:8000/search/quick', params={'q': 'drilling techniques', 'limit': 5}); print(r.json())"
```

---

## 🔄 Daily Workflow (After Initial Setup)

When you want to work with the application on subsequent days:

```powershell
# 1. Ensure Docker Desktop is running
# 2. Start database (if not running)
docker-compose up -d

# 3. Test connection
python -c "from database.config import test_connection; test_connection()"

# 4. Start API
python api/app.py
```

---

## 🛑 Stopping the Application

```powershell
# Stop the API: Press Ctrl+C in the terminal running the API

# Stop the database container (keeps data)
docker-compose stop

# Stop and remove container (keeps data in volume)
docker-compose down

# Stop and DELETE all data (be careful!)
docker-compose down -v
```

---

## 📊 Useful Database Commands

All database commands must use `docker exec` since PostgreSQL is running in Docker:

```powershell
# Connect to database interactively
docker exec -it video_search_db psql -U postgres -d video_semantic_search

# List all tables
docker exec video_search_db psql -U postgres -d video_semantic_search -c "\dt"

# Count videos in database
docker exec video_search_db psql -U postgres -d video_semantic_search -c "SELECT COUNT(*) FROM videos;"

# Count transcript segments
docker exec video_search_db psql -U postgres -d video_semantic_search -c "SELECT COUNT(*) FROM transcript_segments;"

# Check pgvector extension
docker exec video_search_db psql -U postgres -d video_semantic_search -c "\dx"

# View recent searches
docker exec video_search_db psql -U postgres -d video_semantic_search -c "SELECT * FROM search_queries ORDER BY created_at DESC LIMIT 10;"
```

---

## ⚙️ Configuration Files

- **`.env`**: Database credentials and API settings
  ```env
  DB_HOST=localhost        # Use 'localhost' NOT '127.0.0.1' for Docker on Windows
  DB_PORT=5432
  DB_NAME=video_semantic_search
  DB_USER=postgres
  DB_PASSWORD=postgres
  ```

- **`docker-compose.yml`**: PostgreSQL container configuration
- **`database/config.py`**: Python database connection settings

---

## 🐛 Troubleshooting

### Database Won't Start

```powershell
# Check if Docker Desktop is running
docker --version

# Check container logs
docker logs video_search_db --tail 50

# Restart container
docker-compose restart

# If all else fails, recreate
docker-compose down -v
docker-compose up -d
```

### Connection Failed Error

**Error**: `password authentication failed for user "postgres"`

**Solution**: Make sure `.env` uses `DB_HOST=localhost` (not `127.0.0.1`)

### Port Already in Use

**Error**: `Bind for 0.0.0.0:5432 failed: port is already allocated`

**Solution**: Another PostgreSQL is running
```powershell
# Find what's using port 5432
netstat -ano | findstr :5432

# Option 1: Stop local PostgreSQL service
# Option 2: Change port in docker-compose.yml to 5433:5432
```

### No Videos Found

```powershell
# Check if videos exist
dir videos

# Check if videos have been processed
dir processed\transcripts

# Process videos if needed
python basic_pipeline.py
```

---

## 📁 Directory Structure Reference

```
BasePipeline/
├── videos/                      # Place your videos here
├── processed/                   # Processing outputs
│   ├── transcripts/            # Transcription results
│   ├── scenes/                 # Scene detection
│   └── results/                # Combined results
├── database/                    # Database code
├── api/                        # FastAPI application
├── .env                        # Configuration (already set up!)
└── docker-compose.yml          # Docker setup (already configured!)
```

---

## 🎯 What's Already Done

✅ Docker PostgreSQL container created  
✅ Database `video_semantic_search` created  
✅ Tables created (videos, transcript_segments, embeddings, etc.)  
✅ pgvector extension installed (v0.8.1)  
✅ Environment variables configured (`.env`)  
✅ Database connection tested and working  

---

## Next Steps for New Projects

1. Place video files in `videos/` folder
2. Run `python basic_pipeline.py` to process them
3. Run `python database/ingest.py` to load into database
4. Start API with `python api/app.py`
5. Query your videos!

---

**Last Updated**: 2026-02-04  
**PostgreSQL Version**: 16.11  
**pgvector Version**: 0.8.1
