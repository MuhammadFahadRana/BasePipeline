"""FastAPI application for video semantic search."""

import sys
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

# Lazy-loaded components
_video_qa = None


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
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
    
    file_size = os.path.getsize(video_path)
    
    # Determine content type based on extension
    ext = os.path.splitext(video_path)[1].lower()
    content_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
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
async def search(request: SearchRequest, db: Session = Depends(get_db)):
    """
    Semantic search endpoint.

    Search video transcripts using hybrid semantic + fuzzy text matching.

    **Example queries:**
    - "where Omega Alpha well is discussed"
    - "drilling ultra-long reservoir sections"
    - "geostering techniques"

    **Typo tolerance:**
    - "Umega Alfa" → automatically corrected to "Omega Alpha"

    **Returns:**
    Video filename, timestamp, matching text segment, and search time.
    """
    start_time = time.time()
    search_engine = SemanticSearchEngine(db)

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
    db: Session = Depends(get_db),
):
    """
    Quick search endpoint (GET request for easy testing).
    ```
    GET /search/quick?q=Omega+Alpha+well&limit=5
    ```
    """
    start_time = time.time()
    search_engine = SemanticSearchEngine(db)

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
    db: Session = Depends(get_db),
):
    """
    Exact phrase search (case-insensitive).

    **Example:**
    ```
    GET /search/exact?phrase=Omega+Alpha+well
    ```
    """
    start_time = time.time()
    search_engine = SemanticSearchEngine(db)

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

# Serve the frontend
@app.get("/")
async def read_root():
    return FileResponse('frontend/index.html')

# Mount static files (css, js) - Make sure this is AFTER all other routes
app.mount("/", StaticFiles(directory="frontend"), name="frontend")


if __name__ == "__main__":
    import uvicorn

    # uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

    uvicorn.run("api.app:app", host="localhost", port=8000, reload=True)
