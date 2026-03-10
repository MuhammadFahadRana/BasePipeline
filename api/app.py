"""FastAPI application for video semantic search."""

import sys
import subprocess
import asyncio
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Depends, HTTPException, Query, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Optional
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import os

from database.config import get_db, test_connection
from database.models import Video
from search.semantic_search import SemanticSearchEngine, SearchResult
from search.multi_modal_search import MultiModalSearchEngine, set_optimal_weights
import traceback
import time
from datetime import datetime

# Formats that browsers cannot play natively → must be transcoded
TRANSCODE_EXTENSIONS = {".ts", ".mp2t", ".m2ts", ".mts", ".avi", ".mkv", ".mov"}

# Lazy-loaded components
_video_qa = None
_search_engine = None


# Pydantic models for API
class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(10, description="Number of results to return", ge=1, le=50)
    semantic_weight: float = Field(
        0.7, description="Weight for semantic similarity", ge=0, le=1
    )
    text_weight: float = Field(0.3, description="Weight for text matching", ge=0, le=1)
    min_score: float = Field(0.1, description="Minimum score threshold", ge=0, le=1)
    video_filter: Optional[str] = Field(None, description="Filter by video filename")


class MultiModalSearchRequest(BaseModel):
    """Multi-modal search request model (text + vision)."""

    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(10, description="Number of results to return", ge=1, le=50)
    text_weight: float = Field(0.5, description="Weight for text similarity", ge=0, le=1)
    vision_weight: float = Field(0.5, description="Weight for vision similarity", ge=0, le=1)
    use_vision: bool = Field(True, description="Enable vision search")
    search_mode: Optional[str] = Field(
        "balanced",
        description="Search mode: balanced, text_heavy, vision_heavy, visual_only",
    )
    use_llm: bool = Field(True, description="Use LLM for intent parsing (disable for speed)")
    video_filter: Optional[str] = Field(None, description="Filter by video filename")


class SearchResponse(BaseModel):
    """Search response model."""

    query: str
    results_count: int
    results: List[dict]
    search_time_seconds: float = Field(..., description="Time taken to execute search in seconds")
    search_metadata: Optional[dict] = Field(None, description="Additional search metadata (strategies, LLM intent, etc)")


class QARequest(BaseModel):
    """Question Answering request model."""

    question: str = Field(..., description="The question to ask about the video", min_length=3)
    video_filter: Optional[str] = Field(None, description="Optional specific video to search in")
    top_k: int = Field(5, description="Number of context snippets to use", ge=1, le=10)


class QA_Citation(BaseModel):
    """Citation for a QA answer."""

    segment_id: int
    video_filename: str
    timestamp: str
    text: str
    score: float


class QA_Response(BaseModel):
    """Response from the QA system."""

    answer: str
    citations: List[dict]
    metadata: dict


class VideoInfo(BaseModel):
    """Video information model."""

    id: int
    filename: str
    duration_seconds: Optional[float]
    whisper_model: Optional[str]
    processed_at: Optional[str]


# Initialize FastAPI app
app = FastAPI(
    title="Video Semantic Search API",
    description="Search video transcripts using semantic understanding and fuzzy matching",
    version="1.0.0",
)

# CORS middleware (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Check database connection on startup."""
    if not test_connection():
        raise RuntimeError("Failed to connect to database. Check your .env configuration.")
    print("✓ API server started successfully")


def get_video_qa(db: Session = Depends(get_db)):
    """Lazy loader for VideoQA."""
    global _video_qa
    if _video_qa is None:
        from llm.video_qa import VideoQA

        print("Initializing Video QA system (this may take a moment)...")
        _video_qa = VideoQA(db)
    return _video_qa


def get_search_engine(db: Session = Depends(get_db)):
    """Lazy loader for SemanticSearchEngine."""
    global _search_engine
    if _search_engine is None:
        print("Initializing Semantic Search Engine (this may take a moment)...")
        _search_engine = SemanticSearchEngine(db)
    return _search_engine


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    db_ok = test_connection()
    return {"status": "healthy" if db_ok else "unhealthy", "database": "ok" if db_ok else "error"}


@app.get("/video/stream/{video_id}")
async def stream_video(video_id: int, request: Request, db: Session = Depends(get_db)):
    """
    Stream video file with support for range requests (seeking).
    
    This endpoint allows the frontend to play videos directly in the browser
    and seek to specific timestamps from search results.
    """
    # Get video from database
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = video.file_path
    
    # Path Mapping (FIX): If DB contains Linux absolute paths but we are on Windows,
    # resolve the filename to the local 'videos' directory.
    if not os.path.exists(video_path):
        local_filename = os.path.basename(video_path)
        # Check local 'videos' folder relative to project root
        project_root = Path(__file__).parent.parent
        resolved_path = project_root / "videos" / local_filename
        
        if resolved_path.exists():
            video_path = str(resolved_path)
        else:
            raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
    
    file_size = os.path.getsize(video_path)
    
    # For .ts and other browser-incompatible formats, redirect to the transcode endpoint
    ext = os.path.splitext(video_path)[1].lower()
    if ext in TRANSCODE_EXTENSIONS:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/video/transcode/{video_id}")
    
    # Determine content type based on extension
    content_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
    }
    content_type = content_types.get(ext, "video/mp4")
    
    # Handle range requests for seeking
    range_header = request.headers.get("Range")
    
    if range_header:
        # Parse range header (e.g., "bytes=0-1024")
        range_spec = range_header.replace("bytes=", "")
        start_str, end_str = range_spec.split("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        
        # Ensure valid range
        if start >= file_size:
            raise HTTPException(status_code=416, detail="Range not satisfiable")
        end = min(end, file_size - 1)
        
        chunk_size = end - start + 1
        
        def iterfile():
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    read_size = min(8192, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data
        
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
            "Content-Type": content_type,
        }
        
        return StreamingResponse(iterfile(), status_code=206, headers=headers)
    
    else:
        # Full file request
        def iterfile():
            with open(video_path, "rb") as f:
                while chunk := f.read(8192):
                    yield chunk
        
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": content_type,
        }
        
        return StreamingResponse(iterfile(), headers=headers)


@app.get("/video/transcode/{video_id}")
async def transcode_video(video_id: int, db: Session = Depends(get_db)):
    """
    Transcode a video to browser-compatible MP4 using FFmpeg.

    Used automatically for .ts and other container formats that browsers
    cannot play natively. Streams the transcoded output directly without
    writing a temporary file to disk.
    
    Requires ffmpeg to be installed and on the system PATH.
    """
    # Resolve the video file path (same logic as stream_video)
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = video.file_path
    if not os.path.exists(video_path):
        local_filename = os.path.basename(video_path)
        project_root = Path(__file__).parent.parent
        resolved_path = project_root / "videos" / local_filename
        if resolved_path.exists():
            video_path = str(resolved_path)
        else:
            raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")

    async def stream_ffmpeg():
        """Pipe ffmpeg stdout as a fragmented MP4 stream."""
        cmd = [
            "ffmpeg",
            "-loglevel", "error",      # suppress progress spam
            "-i", video_path,          # input file
            "-c:v", "copy",            # copy video stream (no re-encode → fast)
            "-c:a", "aac",             # re-encode audio to AAC for browser compat
            "-f", "mp4",               # output container
            "-movflags", "frag_keyframe+empty_moov+faststart",  # streaming-safe fragmented MP4
            "pipe:1",                  # write to stdout
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            while True:
                chunk = await proc.stdout.read(65536)
                if not chunk:
                    break
                yield chunk
            await proc.wait()
            if proc.returncode and proc.returncode != 0:
                stderr = await proc.stderr.read()
                print(f"FFmpeg error (video_id={video_id}): {stderr.decode()}")
        except FileNotFoundError:
            # ffmpeg not on PATH
            raise HTTPException(
                status_code=500,
                detail="ffmpeg is not installed or not on PATH. Install ffmpeg to play .ts files."
            )

    return StreamingResponse(
        stream_ffmpeg(),
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'inline; filename="{Path(video_path).stem}.mp4"',
            "X-Content-Type-Options": "nosniff",
        },
    )



@app.get("/videos", response_model=List[VideoInfo])
async def list_videos(db: Session = Depends(get_db)):
    """
    List all videos in the database.
    """
    videos = db.query(Video).all()
    return [
        VideoInfo(
            id=v.id,
            filename=v.filename,
            duration_seconds=v.duration_seconds,
            whisper_model=v.whisper_model,
            processed_at=v.processed_at.isoformat() if v.processed_at else None,
        )
        for v in videos
    ]


@app.post("/qa/ask", response_model=QA_Response)
async def ask_video_question(request: QARequest, qa_system=Depends(get_video_qa)):
    """
    Ask a natural language question about the available videos.
    Uses RAG (Retrieval-Augmented Generation) to answer based on transcripts and visual semantics.
    """
    try:
        result = qa_system.ask(
            question=request.question, video_filter=request.video_filter, top_k=request.top_k
        )
        return result
    except Exception as e:
        print(f"QA Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    videos = db.query(Video).all()

    return [
        VideoInfo(
            id=v.id,
            filename=v.filename,
            duration_seconds=v.duration_seconds,
            whisper_model=v.whisper_model,
            processed_at=v.processed_at.isoformat() if v.processed_at else None,
        )
        for v in videos
    ]
    
    
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, search_engine: SemanticSearchEngine = Depends(get_search_engine)):
    """
    Semantic search endpoint.

    Search video transcripts using hybrid semantic + fuzzy text matching.
    """
    start_time = time.time()

    try:
        results = search_engine.search(
            query=request.query,
            top_k=request.top_k,
            semantic_weight=request.semantic_weight,
            text_weight=request.text_weight,
            min_score=request.min_score,
            video_filter=request.video_filter,
            log_query=True,
        )
        
        search_time = time.time() - start_time

        return SearchResponse(
            query=request.query,
            results_count=len(results),
            results=[r.to_dict() for r in results],
            search_time_seconds=round(search_time, 3),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/search/quick")
async def quick_search(
    q: str = Query(..., description="Search query", min_length=1),
    limit: int = Query(10, description="Number of results", ge=1, le=50),
    video: Optional[str] = Query(None, description="Filter by video filename"),
    search_engine: SemanticSearchEngine = Depends(get_search_engine),
):
    """
    Quick search endpoint (GET request for easy testing).
    ```
    GET /search/quick?q=Omega+Alpha+well&limit=5
    ```
    """
    start_time = time.time()

    try:
        fallback_data = search_engine.search_with_fallback(
            query=q, top_k=limit, video_filter=video
        )
        
        results = fallback_data["results"]
        metadata = fallback_data["search_metadata"]
        search_time = time.time() - start_time

        return {
            "query": q,
            "results_count": len(results),
            "results": [r.to_dict() for r in results],
            "search_time_seconds": round(search_time, 3),
            "search_strategy": metadata.get("search_strategy"),
            "search_message": metadata.get("search_message"),
            "did_you_mean": metadata.get("did_you_mean"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/search/exact")
async def exact_search(
    phrase: str = Query(..., description="Exact phrase to search", min_length=1),
    video: Optional[str] = Query(None, description="Filter by video filename"),
    search_engine: SemanticSearchEngine = Depends(get_search_engine),
):
    """
    Exact phrase search (case-insensitive).
    """
    start_time = time.time()

    try:
        results = search_engine.search_exact_phrase(phrase=phrase, video_filter=video)
        
        search_time = time.time() - start_time

        return {
            "phrase": phrase,
            "results_count": len(results),
            "results": [r.to_dict() for r in results],
            "search_time_seconds": round(search_time, 3),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/search/multimodal", response_model=SearchResponse)
async def multimodal_search(request: MultiModalSearchRequest, db: Session = Depends(get_db)):
    """
    Multi-modal search endpoint combining text and vision.

    Search using **both transcript text and visual content** from keyframes.
    This provides more accurate results by matching both what was said and what was shown.

    **Search Modes:**
    - `balanced`: Equal weight to text and vision (50/50)
    - `text_heavy`: Prioritize transcript matches (70/30)
    - `vision_heavy`: Prioritize visual content (30/70)
    - `visual_only`: Only search by visual similarity (0/100)

    **Example queries:**
    - "drilling rig" → Finds both mentions AND visual appearances of drilling rigs
    - "safety equipment" → Finds helmets, vests even if not mentioned
    - "offshore platform" → Matches visual scenes + spoken references

    **Returns:**
    Results with both text_score and vision_score for transparency.
    """
    start_time = time.time()
    
    try:
        # Set weights based on search mode if provided
        if request.search_mode:
            text_w, vision_w = set_optimal_weights(request.search_mode)
            text_weight = text_w
            vision_weight = vision_w
        else:
            text_weight = request.text_weight
            vision_weight = request.vision_weight

        # Initialize multi-modal search engine
        mm_search = MultiModalSearchEngine(
            db=db,
            text_weight=text_weight,
            vision_weight=vision_weight
        )

        # Perform search with fallback (includes LLM intent parsing)
        fallback_data = mm_search.search_with_fallback(
            query=request.query,
            top_k=request.top_k,
            video_filter=request.video_filter,
            use_llm=request.use_llm,
        )
        
        results = fallback_data["results"]
        metadata = fallback_data["search_metadata"]

        search_time = time.time() - start_time

        return SearchResponse(
            query=request.query,
            results_count=len(results),
            results=[r.to_dict() if hasattr(r, 'to_dict') else r for r in results],
            search_time_seconds=round(search_time, 3),
            search_metadata=metadata,
        )

    except Exception as e:
        # Log the full error for debugging
        print(f"\nMulti-modal search error: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        
        # Fallback to text-only search if vision fails
        if any(w in str(e).lower() for w in ["vision", "clip", "siglip", "embedding"]):
            print(f"Vision search failed, falling back to text-only: {e}")
            search_engine = SemanticSearchEngine(db)
            results = search_engine.search(
                query=request.query,
                top_k=request.top_k,
                video_filter=request.video_filter
            )
            search_time = time.time() - start_time
            return SearchResponse(
                query=request.query,
                results_count=len(results),
                results=[r.to_dict() for r in results],
                search_time_seconds=round(search_time, 3),
            )
        else:
            raise HTTPException(status_code=500, detail=f"Multi-modal search failed: {str(e)}")


@app.get("/search/multimodal/quick")
async def quick_multimodal_search(
    q: str = Query(..., description="Search query", min_length=1),
    limit: int = Query(10, description="Number of results", ge=1, le=50),
    mode: str = Query("balanced", description="Search mode: balanced, text_heavy, vision_heavy, visual_only"),
    use_llm: bool = Query(True, description="Use LLM for intent parsing (disable for speed)"),
    video: Optional[str] = Query(None, description="Filter by video filename"),
    db: Session = Depends(get_db),
):
    """
    Quick multi-modal search (GET request for easy testing).

    **Example:**
    ```
    GET /search/multimodal/quick?q=drilling+techniques&limit=5&mode=balanced
    ```

    **Modes:**
    - `balanced` (default): 50% text, 50% vision
    - `text_heavy`: 70% text, 30% vision
    - `vision_heavy`: 30% text, 70% vision
    - `visual_only`: 0% text, 100% vision
    """
    try:
        text_weight, vision_weight = set_optimal_weights(mode)

        start_time = time.time()

        mm_search = MultiModalSearchEngine(
            db=db,
            text_weight=text_weight,
            vision_weight=vision_weight
        )

        fallback_data = mm_search.search_with_fallback(
            query=q,
            top_k=limit,
            video_filter=video,
            use_llm=use_llm,
        )

        results = fallback_data["results"]
        metadata = fallback_data["search_metadata"]
        search_time = time.time() - start_time

        return {
            "query": q,
            "mode": mode,
            "weights": {"text": text_weight, "vision": vision_weight},
            "results_count": len(results),
            "results": [r.to_dict() for r in results],
            "search_time_seconds": round(search_time, 3),
            "search_strategy": metadata.get("search_strategy"),
            "search_message": metadata.get("search_message"),
            "llm_intent": metadata.get("llm_intent"),
        }

    except Exception as e:
        # Log the full error for debugging
        print(f"\nQuick multi-modal search error: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        
        # Graceful fallback to text-only
        if any(w in str(e).lower() for w in ["vision", "clip", "siglip", "embedding"]):
            print(f"Vision search unavailable, using text-only: {e}")
            search_engine = SemanticSearchEngine(db)
            fallback_data = search_engine.search_with_fallback(
                query=q, top_k=limit, video_filter=video
            )
            results = fallback_data["results"]
            metadata = fallback_data["search_metadata"]
            search_time = time.time() - start_time
            return {
                "query": q,
                "mode": "text_only (vision unavailable)",
                "weights": {"text": 1.0, "vision": 0.0},
                "results_count": len(results),
                "results": [r.to_dict() for r in results],
                "search_time_seconds": round(search_time, 3),
                "search_strategy": metadata.get("search_strategy"),
                "search_message": metadata.get("search_message"),
            }
        else:
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/search/visual/image")
async def visual_image_search(
    file: UploadFile = File(...),
    limit: int = Query(10, description="Number of results", ge=1, le=50),
    video: Optional[str] = Query(None, description="Filter by video filename"),
    db: Session = Depends(get_db),
):
    """
    Reverse image search - upload an image to find similar moments in videos!
    
    **How it works:**
    - Uploads the image and generates a vision embedding (SigLIP)
    - Matches against all indexed keyframes
    - Returns timestamps of similar visual scenes
    """
    start_time = time.time()
    
    try:
        from search.visual_search import VisualSearchEngine
        from database.models import SearchQuery, SearchImageCache
        import hashlib
        
        # Read image bytes
        image_bytes = await file.read()
        
        visual_engine = VisualSearchEngine(db)
        results = visual_engine.search_by_image(
            image_input=image_bytes,
            top_k=limit,
            video_filter=video
        )
        
        search_time = time.time() - start_time
        
        # Cache image embedding for re-ranking / "find more like this"
        try:
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            existing_cache = db.query(SearchImageCache).filter_by(image_hash=image_hash).first()
            if existing_cache:
                existing_cache.search_count += 1
                existing_cache.last_used = datetime.utcnow()
            else:
                vision_embedding = visual_engine.vision_model.encode_image(image_bytes, normalize=True)
                cache_entry = SearchImageCache(
                    filename=file.filename or 'uploaded_image',
                    image_hash=image_hash,
                    embedding=vision_embedding.tolist(),
                )
                db.add(cache_entry)
            db.commit()
        except Exception as cache_err:
            print(f"Warning: Failed to cache image embedding: {cache_err}")
            db.rollback()
        
        # Log image search query for analytics/learning
        try:
            query_log = SearchQuery(
                query_text=f"[IMAGE] {file.filename or 'uploaded_image'}",
                search_type="image",
                results_count=len(results),
                top_result_id=results[0].segment_id if results else None,
            )
            db.add(query_log)
            db.commit()
        except Exception as log_err:
            print(f"Warning: Failed to log image query: {log_err}")
            db.rollback()
        
        return {
            "query": f"Image: {file.filename}",
            "search_type": "reverse_image_search",
            "results_count": len(results),
            "results": [r.to_dict() for r in results],
            "search_time_seconds": round(search_time, 3),
        }
        
    except Exception as e:
        print(f"Visual image search error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Visual image search failed: {str(e)}")


@app.post("/search/visual/combined")
async def visual_combined_search(
    file: UploadFile = File(...),
    text_query: str = Query("", description="Optional text to refine image search"),
    limit: int = Query(10, description="Number of results", ge=1, le=50),
    video: Optional[str] = Query(None, description="Filter by video filename"),
    image_weight: float = Query(0.7, description="Image embedding weight", ge=0, le=1),
    db: Session = Depends(get_db),
):
    """
    Combined image + text search.
    Upload an image AND provide text to get more precise visual matches.
    The image and text embeddings are fused with configurable weights.
    """
    start_time = time.time()
    
    try:
        from search.visual_search import VisualSearchEngine
        from database.models import SearchQuery
        
        image_bytes = await file.read()
        visual_engine = VisualSearchEngine(db)
        text_weight = 1.0 - image_weight
        
        if text_query.strip():
            results = visual_engine.search_by_image_and_text(
                image_input=image_bytes,
                text_query=text_query,
                top_k=limit,
                video_filter=video,
                image_weight=image_weight,
                text_weight=text_weight
            )
        else:
            results = visual_engine.search_by_image(
                image_input=image_bytes,
                top_k=limit,
                video_filter=video
            )
        
        search_time = time.time() - start_time
        
        # Log
        try:
            query_log = SearchQuery(
                query_text=f"[IMAGE+TEXT] {file.filename}: {text_query}" if text_query.strip() else f"[IMAGE] {file.filename}",
                search_type="image" if not text_query.strip() else "hybrid",
                results_count=len(results),
                top_result_id=results[0].segment_id if results else None,
            )
            db.add(query_log)
            db.commit()
        except Exception:
            db.rollback()
        
        return {
            "query": text_query or f"Image: {file.filename}",
            "search_type": "combined_image_text" if text_query.strip() else "reverse_image_search",
            "image_weight": image_weight,
            "text_weight": text_weight,
            "results_count": len(results),
            "results": [r.to_dict() for r in results],
            "search_time_seconds": round(search_time, 3),
        }
    except Exception as e:
        print(f"Combined search error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Combined search failed: {str(e)}")


@app.get("/search/visual")
async def visual_search(
    q: str = Query(..., description="Visual search query", min_length=1),
    limit: int = Query(10, description="Number of results", ge=1, le=50),
    video: Optional[str] = Query(None, description="Filter by video filename"),
    db: Session = Depends(get_db),
):
    """
    Pure visual search - finds images matching your description!
    
    **Perfect for:**
    - "picture of an oil rig"
    - "image of safety equipment"  
    - "show me drilling operations"
    - "ocean scenes"
    
    **How it works:**
    - Searches visual embeddings (SigLIP) directly
    - IGNORES transcript completely
    - Finds what's SHOWN, not what's SAID
    
    **Example:**
    ```
    GET /search/visual?q=oil+rig&limit=10
    ```
    """
    start_time = time.time()
    
    try:
        from search.visual_search import VisualSearchEngine
        
        visual_engine = VisualSearchEngine(db)
        results = visual_engine.search_visual(
            query=q,
            top_k=limit,
            video_filter=video
        )
        
        search_time = time.time() - start_time
        
        return {
            "query": q,
            "search_type": "visual_only",
            "results_count": len(results),
            "results": [r.to_dict() for r in results],
            "search_time_seconds": round(search_time, 3),
        }
        
    except Exception as e:
        print(f"Visual search error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Visual search failed: {str(e)}")


@app.get("/search/hybrid")
async def hybrid_search(
    q: str = Query(..., description="Search query", min_length=1),
    limit: int = Query(10, description="Number of results", ge=1, le=50),
    mode: str = Query("auto", description="Search mode: auto, visual, text, balanced"),
    video: Optional[str] = Query(None, description="Filter by video filename"),
    db: Session = Depends(get_db),
):
    """
    Truly hybrid search - combines text + semantic + visual!
    
    **Auto-detection:**
    - "picture of oil rig" → Visual-heavy (70% visual, 30% text)
    - "discussed drilling" → Text-heavy (70% text, 30% visual)
    - "oil rig" → Balanced (50% text, 50% visual)
    
    **Modes:**
    - `auto`: Automatically detects query type (recommended)
    - `visual`: Force visual-heavy search
    - `text`: Force text-heavy search
    - `balanced`: Equal weights
    
    **Example:**
    ```
    GET /search/hybrid?q=oil+rig&mode=auto&limit=10
    ```
    """
    start_time = time.time()
    
    try:
        from search.visual_search import HybridSearchEngine
        
        # Set weights based on mode
        if mode == "visual":
            hybrid_engine = HybridSearchEngine(db, text_weight=0.1, semantic_weight=0.2, visual_weight=0.7)
            auto_mode = False
        elif mode == "text":
            hybrid_engine = HybridSearchEngine(db, text_weight=0.4, semantic_weight=0.5, visual_weight=0.1)
            auto_mode = False
        elif mode == "balanced":
            hybrid_engine = HybridSearchEngine(db, text_weight=0.33, semantic_weight=0.33, visual_weight=0.34)
            auto_mode = False
        else:  # auto
            hybrid_engine = HybridSearchEngine(db)
            auto_mode = True
        
        results = hybrid_engine.search(
            query=q,
            top_k=limit,
            video_filter=video,
            auto_mode=auto_mode
        )
        
        search_time = time.time() - start_time
        
        return {
            "query": q,
            "search_type": "hybrid",
            "mode": mode,
            "results_count": len(results),
            "results": [r.to_dict() for r in results],
            "search_time_seconds": round(search_time, 3),
        }
        
    except Exception as e:
        print(f"Hybrid search error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  OpenAI-compatible endpoints — for Open WebUI / any OpenAI client       ║
# ║  Base URL to use in Open WebUI: http://host.docker.internal:8000/v1     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "video-rag"
    messages: List[ChatMessage]
    stream: bool = True
    max_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.1, ge=0.0, le=2.0)


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model list. Required by Open WebUI on startup."""
    return {
        "object": "list",
        "data": [
            {
                "id": "video-rag",
                "object": "model",
                "created": 1700000000,
                "owned_by": "local",
                "description": "Video semantic search RAG — answers questions from your video library.",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    db: Session = Depends(get_db),
):
    """
    OpenAI-compatible chat completions endpoint.

    Workflow:
    1. Extract the latest user message as the question
    2. Check system message for optional `video:<filename>` scoping directive
    3. Run VideoQA RAG: semantic search → Qwen2.5-1.5B generates grounded answer
    4. Stream back tokens via SSE (or return full JSON if stream=False)

    Connect Open WebUI to: http://host.docker.internal:8000/v1
    Select model: video-rag
    """
    from llm.video_qa_streaming import get_streaming_qa

    # ── Extract the user's question ──────────────────────────────────────
    user_question = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_question = msg.content.strip()
            break

    if not user_question:
        raise HTTPException(status_code=400, detail="No user message found in messages.")

    # ── Check system prompt for video scoping directive ──────────────────
    # e.g. system: "video:AkerBP_2.mp4" → restricts search to that file
    video_filter: Optional[str] = None
    for msg in request.messages:
        if msg.role == "system":
            import re
            m = re.search(r"video:\s*([^\s]+)", msg.content, re.IGNORECASE)
            if m:
                video_filter = m.group(1).strip()
                break

    # ── Load StreamingVideoQA (singleton, lazy) ──────────────────────────
    try:
        qa = get_streaming_qa(db=db)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"VideoQA system could not be initialised: {e}"
        )

    # ── Streaming response (Open WebUI default: stream=True) ─────────────
    if request.stream:
        async def sse_generator():
            try:
                for chunk in qa.stream_ask(
                    question=user_question,
                    video_filter=video_filter,
                    max_new_tokens=request.max_tokens,
                ):
                    yield chunk
                    # Small yield point so FastAPI can flush SSE chunks
                    await asyncio.sleep(0)
            except Exception as e:
                import traceback
                traceback.print_exc()
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # Disable nginx buffering
            },
        )

    # ── Non-streaming response ────────────────────────────────────────────
    try:
        result = qa.ask_sync(
            question=user_question,
            video_filter=video_filter,
            top_k=5,
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/keyframe")
async def serve_keyframe(path: str = Query(..., description="Path to keyframe image")):
    """Serve keyframe images for thumbnails in search results."""
    from pathlib import Path as FilePath
    
    keyframe_path = FilePath(path)
    if not keyframe_path.exists():
        raise HTTPException(status_code=404, detail="Keyframe not found")
    
    # Basic security: only serve image files
    if keyframe_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    return FileResponse(str(keyframe_path), media_type=f"image/{keyframe_path.suffix.lstrip('.').replace('jpg', 'jpeg')}")


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

@app.get("/api-info")
async def api_info():
    """API information endpoint (moved from root)."""
    return {
        "name": "Video Semantic Search API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "search": "/search",
            "videos": "/videos",
            "health": "/health",
            "analytics": "/analytics",
        },
    }


@app.get("/analytics")
async def search_analytics(db: Session = Depends(get_db)):
    """
    Search analytics dashboard.
    Returns stats: total queries, type breakdown, top queries, zero-result queries, daily trend.
    """
    from sqlalchemy import func, text as sa_text
    from database.models import SearchQuery
    
    try:
        # Total queries
        total = db.query(func.count(SearchQuery.id)).scalar() or 0
        
        # Breakdown by search_type
        type_rows = db.query(
            SearchQuery.search_type,
            func.count(SearchQuery.id)
        ).group_by(SearchQuery.search_type).all()
        type_breakdown = {t or "text": c for t, c in type_rows}
        
        # Average results count
        avg_results = db.query(func.avg(SearchQuery.results_count)).scalar()
        avg_results = round(float(avg_results), 1) if avg_results else 0
        
        # Top 10 most common queries
        top_queries_rows = db.query(
            SearchQuery.query_text,
            func.count(SearchQuery.id).label('count')
        ).group_by(SearchQuery.query_text).order_by(
            func.count(SearchQuery.id).desc()
        ).limit(10).all()
        top_queries = [{"query": q, "count": c} for q, c in top_queries_rows]
        
        # Queries with zero results
        zero_results = db.query(
            SearchQuery.query_text,
            SearchQuery.search_type,
            SearchQuery.search_timestamp
        ).filter(
            SearchQuery.results_count == 0
        ).order_by(SearchQuery.search_timestamp.desc()).limit(20).all()
        zero_result_queries = [
            {"query": q, "type": t or "text", "timestamp": str(ts)}
            for q, t, ts in zero_results
        ]
        
        # Daily trend (last 14 days)
        daily_rows = db.execute(sa_text("""
            SELECT DATE(search_timestamp) as day, COUNT(*) as count
            FROM search_queries
            WHERE search_timestamp >= CURRENT_DATE - INTERVAL '14 days'
            GROUP BY DATE(search_timestamp)
            ORDER BY day DESC
        """)).fetchall()
        daily_trend = [{"date": str(d), "count": c} for d, c in daily_rows]
        
        return {
            "total_queries": total,
            "type_breakdown": type_breakdown,
            "avg_results_count": avg_results,
            "top_queries": top_queries,
            "zero_result_queries": zero_result_queries,
            "daily_trend": daily_trend,
        }
    except Exception as e:
        print(f"Analytics error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")


@app.post("/admin/enrich-captions")
async def enrich_captions(
    video_id: Optional[int] = None,
    batch_size: int = 10,
    db: Session = Depends(get_db),
):
    """
    Re-enrich existing scenes with Qwen2-VL captions, object labels, and OCR text.

    Finds all scenes with NULL captions (and an existing keyframe), runs Qwen2-VL
    on each keyframe, saves the results back to the scenes table, and generates a
    text embedding from the caption so that vector (semantic) search can find these scenes.

    **This is a long-running operation.** For large libraries call it with a small
    batch_size and repeat until `scenes_remaining` returns 0.

    Args:
        video_id: Optional - restrict to a single video's scenes
        batch_size: How many scenes to process per call (default 10)
    """
    try:
        from scene_detector import SceneDetector, SceneConfig
        from database.models import Scene, Embedding
        from embeddings.text_embeddings import get_embedding_generator

        # Query scenes that need enrichment: caption IS NULL + keyframe exists
        query = db.query(Scene).filter(Scene.keyframe_path.isnot(None))
        if video_id:
            query = query.filter(Scene.video_id == video_id)
        
        # Split into: still need enrichment vs already done
        all_scenes = query.all()
        unenriched = [s for s in all_scenes if s.caption is None]
        total_remaining = len(unenriched)
        
        if total_remaining == 0:
            return {
                "status": "already_complete",
                "message": "All scenes with keyframes already have captions.",
                "scenes_enriched": 0,
                "scenes_remaining": 0,
            }

        batch = unenriched[:batch_size]
        
        # Load Qwen2-VL via SceneDetector (lazy-loads the model)
        cfg = SceneConfig(enable_visual_enrichment=True)
        detector = SceneDetector(config=cfg)
        qwen = detector._ensure_qwen_vl()
        if qwen is None:
            raise HTTPException(
                status_code=503,
                detail="Qwen2-VL model could not be loaded. Check server logs."
            )

        # Load text embedding generator for indexing captions
        emb_gen = get_embedding_generator()
        
        enriched_count = 0
        embedding_count = 0
        
        for scene in batch:
            kf_path = Path(scene.keyframe_path)
            # Resolve Windows / Linux path differences
            if not kf_path.exists():
                project_root = Path(__file__).parent.parent
                candidate = project_root / scene.keyframe_path
                if candidate.exists():
                    kf_path = candidate

            if not kf_path.exists():
                print(f"  Keyframe not found for scene {scene.id}: {scene.keyframe_path}")
                continue

            try:
                result = qwen.analyze_image(str(kf_path))
                caption = result.get("caption")
                object_labels = result.get("object_labels", [])
                ocr_text = result.get("ocr_text")

                # Update scene columns
                scene.caption = caption
                scene.object_labels = object_labels
                scene.ocr_text = ocr_text
                enriched_count += 1

                # Build the text to embed: caption + object labels + OCR
                parts = [caption] if caption else []
                if object_labels:
                    if isinstance(object_labels, list):
                        parts.append(" ".join(str(l) for l in object_labels))
                    else:
                        parts.append(str(object_labels))
                if ocr_text:
                    parts.append(ocr_text)

                if parts:
                    embed_text = " ".join(parts)
                    vec = emb_gen.encode_single(embed_text)

                    # Upsert: skip if an embedding for this scene already exists
                    existing_emb = db.query(Embedding).filter(
                        Embedding.scene_id == scene.id,
                        Embedding.segment_id == None,  # noqa: E711
                    ).first()

                    if existing_emb:
                        existing_emb.embedding = vec.tolist()
                    else:
                        new_emb = Embedding(
                            scene_id=scene.id,
                            segment_id=None,
                            embedding=vec.tolist(),
                            embedding_model=emb_gen.model_name,
                        )
                        db.add(new_emb)
                    embedding_count += 1

            except Exception as e:
                print(f"  Failed to enrich scene {scene.id}: {e}")
                continue

        db.commit()

        return {
            "status": "success",
            "scenes_enriched": enriched_count,
            "embeddings_created": embedding_count,
            "scenes_remaining": total_remaining - len(batch),
            "batch_size": batch_size,
            "message": (
                f"Enriched {enriched_count}/{len(batch)} scenes. "
                f"{total_remaining - len(batch)} scenes still need enrichment. "
                "Call this endpoint again to continue."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Enrichment failed: {str(e)}")


@app.get("/admin/caption-stats")
async def caption_stats(db: Session = Depends(get_db)):
    """Returns how many scenes have captions vs still need enrichment."""
    from database.models import Scene
    total = db.query(Scene).count()
    with_caption = db.query(Scene).filter(Scene.caption.isnot(None)).count()
    with_keyframe = db.query(Scene).filter(Scene.keyframe_path.isnot(None)).count()
    return {
        "total_scenes": total,
        "scenes_with_caption": with_caption,
        "scenes_needing_enrichment": with_keyframe - with_caption,
        "scenes_without_keyframe": total - with_keyframe,
        "caption_coverage_pct": round(with_caption / total * 100, 1) if total else 0,
    }


# Serve the frontend
@app.get("/")
async def read_root():
    return FileResponse('frontend/index.html')

# Mount static files (css, js) - Make sure this is AFTER all other routes
app.mount("/", StaticFiles(directory="frontend"), name="frontend")


if __name__ == "__main__":
    import uvicorn

    # Pass the app object directly when reload=False for better reliability
    uvicorn.run(app, host="localhost", port=8000)
