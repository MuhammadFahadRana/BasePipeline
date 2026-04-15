"""FastAPI application for video semantic search."""

import sys
import subprocess
import asyncio
import json
from pathlib import Path
from functools import lru_cache

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Depends, HTTPException, Query, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from typing import List, Optional
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import os

from database.config import get_db, test_connection
from database.models import (
    Video,
    VideoCategory,
    User,
    UserCategoryAccess,
    TranscriptSegment,
)
from search.semantic_search import SemanticSearchEngine, SearchResult
from search.multi_modal_search import MultiModalSearchEngine
from api.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    require_admin,
    get_video_category,
    user_can_access_video,
    get_user_allowed_categories,
)
import traceback
import time
import re
import urllib.parse
import urllib.request
from datetime import datetime
from contextlib import asynccontextmanager

# Formats that browsers cannot play natively → must be transcoded
TRANSCODE_EXTENSIONS = {".ts", ".mp2t", ".m2ts", ".mts", ".avi", ".mkv", ".mov"}

# Lazy-loaded components
_video_qa = None
_search_engine = None
_mm_search_engine = None

PROJECT_ROOT = Path(__file__).parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
INDEX_HTML_PATH = FRONTEND_DIR / "index.html"
FRONTEND_ASSETS = ("styles.css", "chat_styles.css", "app.js", "chat.js")


def _server_capabilities() -> dict:
    import torch

    has_cuda = torch.cuda.is_available()
    return {
        "compute_device": "cuda" if has_cuda else "cpu",
        "has_cuda": has_cuda,
        "default_search_mode": "text_only",
        "deep_search_policy": "disabled",
        "multimodal_enabled": True,
    }


@lru_cache(maxsize=len(FRONTEND_ASSETS))
def _frontend_asset_version(asset_name: str) -> str:
    asset_path = FRONTEND_DIR / asset_name
    return str(asset_path.stat().st_mtime_ns)


def _render_frontend_index() -> str:
    html = INDEX_HTML_PATH.read_text(encoding="utf-8")
    for asset_name in FRONTEND_ASSETS:
        versioned_name = f'{asset_name}?v={_frontend_asset_version(asset_name)}'
        html = html.replace(f'"{asset_name}"', f'"{versioned_name}"')
    return html


def _resolve_video_file_path(raw_path: Optional[str]) -> Optional[Path]:
    if not raw_path:
        return None

    candidate = Path(raw_path)
    if candidate.exists():
        return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)

    local_candidate = PROJECT_ROOT / "videos" / os.path.basename(raw_path)
    if local_candidate.exists():
        return local_candidate

    if not candidate.is_absolute():
        relative_candidate = PROJECT_ROOT / candidate
        if relative_candidate.exists():
            return relative_candidate

    return None


# ── Category-intent detection from natural-language queries ──────────────
_CATEGORY_PATTERNS = [
    # English patterns
    re.compile(
        r"(?:show|list|find|get|display|give)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:videos?|results?)\s+(?:in|from|for|under|of)\s+(?:the\s+)?(?:category\s+)?[\"']?(.+?)[\"']?\s*(?:category)?\s*$",
        re.I,
    ),
    re.compile(
        r"(?:all\s+)?(?:videos?|results?)\s+(?:in|from|for|under|of)\s+(?:the\s+)?(?:category\s+)?[\"']?(.+?)[\"']?\s*(?:category)?\s*$",
        re.I,
    ),
    # Norwegian patterns
    re.compile(
        r"(?:vis|finn|hent|søk)\s+(?:meg\s+)?(?:alle\s+)?(?:videoer?|resultater?)\s+(?:i|fra|for|under)\s+(?:kategorien?\s+)?[\"']?(.+?)[\"']?\s*(?:kategori(?:en)?)?\s*$",
        re.I,
    ),
]

# ── Site (label) intent detection from natural-language queries ───────────


def _detect_category_intent(query: str, known_categories: set) -> Optional[str]:
    """If the query is a natural-language request to browse a category, return the category name."""
    for pattern in _CATEGORY_PATTERNS:
        m = pattern.search(query.strip())
        if m:
            candidate = m.group(1).strip().strip("\"'")
            # Fuzzy match against known categories (case-insensitive)
            for cat in known_categories:
                if cat.lower() == candidate.lower():
                    return cat
                # Partial match: "akerbp" matches "AkerBP"
                if candidate.lower() in cat.lower() or cat.lower() in candidate.lower():
                    return cat
    return None


def _detect_site_intent(query: str, known_sites: set) -> Optional[str]:
    """Detect if a query references a known site/label.

    Uses direct substring matching rather than fragile regex patterns.
    Handles: 'Yggdrasil', 'show me Yggdrasil', 'tell about Yggdrasil',
    'Yggdrasil videos', 'vis meg Yggdrasil', etc.
    """
    q_lower = query.strip().lower()
    if not q_lower:
        return None

    # 1) Exact match: query IS the site name
    for site in known_sites:
        if q_lower == site.lower():
            return site

    # 2) Site name appears as a substring of the query
    #    Pick the longest match to avoid e.g. "Troll" matching inside "Trollfjord"
    best_match = None
    best_len = 0
    for site in known_sites:
        if site.lower() in q_lower and len(site) > best_len:
            best_match = site
            best_len = len(site)

    return best_match


def _get_allowed_filenames(user: User, db: Session) -> Optional[set]:
    """Return the set of video filenames the user may see, or None for admins (=all)."""
    allowed_cats = get_user_allowed_categories(user)
    if allowed_cats is None:  # admin
        return None
    all_videos = db.query(Video).all()
    return {
        v.filename for v in all_videos if get_video_category(v.filename) in allowed_cats
    }


def _get_accessible_available_videos(
    user: User, db: Session
) -> List[tuple[Video, Path]]:
    allowed_cats = get_user_allowed_categories(user)
    visible_videos: List[tuple[Video, Path]] = []

    for video in db.query(Video).all():
        if allowed_cats is not None and get_video_category(video.filename) not in allowed_cats:
            continue

        resolved_path = _resolve_video_file_path(video.file_path)
        if resolved_path is None:
            continue

        visible_videos.append((video, resolved_path))

    return visible_videos


def _serialize_video_info(video: Video) -> "VideoInfo":
    return VideoInfo(
        id=video.id,
        filename=video.filename,
        duration_seconds=video.duration_seconds,
        whisper_model=video.whisper_model,
        processed_at=video.processed_at.isoformat() if video.processed_at else None,
        label=video.label,
        category=video.category_rel.name if video.category_rel else None,
        category_id=video.category_id,
    )


def _is_result_playable(result) -> bool:
    return _resolve_video_file_path(getattr(result, "video_path", None)) is not None


def _filter_results(results, allowed_filenames, limit=None):
    """Filter search results to only include videos the user may access."""
    filtered = []
    for result in results:
        if (
            allowed_filenames is not None
            and getattr(result, "video_filename", None) not in allowed_filenames
        ):
            continue
        if not _is_result_playable(result):
            continue

        filtered.append(result)
        if limit and len(filtered) >= limit:
            break
    return filtered


def _filter_result_dicts(result_dicts, allowed_filenames, limit=None):
    """Filter dict-form search results to only include videos the user may access."""
    filtered = []
    for result in result_dicts:
        if (
            allowed_filenames is not None
            and result.get("video_filename") not in allowed_filenames
        ):
            continue
        if not _resolve_video_file_path(result.get("video_path")):
            continue
        filtered.append(result)
        if limit and len(filtered) >= limit:
            break
    return filtered[:limit] if limit else filtered


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
    language: Optional[str] = Field(
        None,
        description="Language hint for search (e.g. 'en', 'no'). Auto-detected if not set.",
    )


class MultiModalSearchRequest(BaseModel):
    """Multi-modal search request model (text + vision)."""

    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(10, description="Number of results to return", ge=1, le=50)
    text_weight: float = Field(
        0.5, description="Weight for text similarity", ge=0, le=1
    )
    vision_weight: float = Field(
        0.5, description="Weight for vision similarity", ge=0, le=1
    )
    use_vision: bool = Field(True, description="Enable vision search")
    use_llm: bool = Field(
        True, description="Use LLM for intent parsing (disable for speed)"
    )
    video_filter: Optional[str] = Field(None, description="Filter by video filename")
    language: Optional[str] = Field(
        None,
        description="Language hint for search (e.g. 'en', 'no'). Auto-detected if not set.",
    )


class SearchResponse(BaseModel):
    """Search response model."""

    query: str
    results_count: int
    results: List[dict]
    search_time_seconds: float = Field(
        ..., description="Time taken to execute search in seconds"
    )
    search_metadata: Optional[dict] = Field(
        None, description="Additional search metadata (strategies, LLM intent, etc)"
    )


class QARequest(BaseModel):
    """Question Answering request model."""

    question: str = Field(
        ..., description="The question to ask about the video", min_length=3
    )
    video_filter: Optional[str] = Field(
        None, description="Optional specific video to search in"
    )
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
    label: Optional[str] = None
    category: Optional[str] = None
    category_id: Optional[int] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for app startup and shutdown."""
    # ── Startup ──────────────────────────────────────────────────────────
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ATLAS API starting up...")
    
    if not test_connection():
        raise RuntimeError(
            "Failed to connect to database. Check your .env configuration."
        )
        
    # Create auth tables if they don't exist yet
    from database.config import engine
    from database.models import Base
    Base.metadata.create_all(bind=engine)

    # Seed a default admin user if no users exist at all
    from database.config import SessionLocal
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            admin = User(
                username="admin",
                password_hash=hash_password("admin"),
                role="admin",
            )
            db.add(admin)
            db.commit()
            print("[auth] Created default admin user (admin / admin)")
        else:
            print("[auth] Users table OK")

        # Seed default video categories
        _DEFAULT_CATEGORIES = ["Oil & Gas", "Maintenance", "Installation", "Operations"]
        for cat_name in _DEFAULT_CATEGORIES:
            if not db.query(VideoCategory).filter(VideoCategory.name == cat_name).first():
                db.add(VideoCategory(name=cat_name))
        db.commit()
    finally:
        db.close()

    # ── Pre-warm models (Optimized) ──────────────────────────────────────
    # We only pre-warm the fast semantic engine by default.
    # Heavy models (LLM) are pre-warmed ONLY if CUDA is available to avoid CPU lag.
    capabilities = _server_capabilities()
    has_cuda = capabilities["has_cuda"]
    device_label = "GPU (CUDA)" if has_cuda else "CPU"
    print(f"[device] Detected hardware: {device_label}")
    
    print("[warmup] Pre-loading search engines...")
    warmup_db = SessionLocal()
    try:
        global _search_engine, _mm_search_engine
        if _search_engine is None:
            _search_engine = SemanticSearchEngine(warmup_db)
        if has_cuda and _mm_search_engine is None:
            _mm_search_engine = MultiModalSearchEngine(db=warmup_db, text_search=_search_engine)
            
        if has_cuda:
            # Only preload heavy stuff on GPU
            from search.reranker import get_reranker
            from llm.query_parser import get_query_parser
            from embeddings.vision_embeddings import get_vision_embedding_generator
            get_reranker(enabled=True)
            get_query_parser(enabled=True)
            get_vision_embedding_generator()
            print("[warmup] GPU models pre-loaded")
        else:
            print("[warmup] CPU mode: heavy models will be lazy-loaded on demand to speed up startup")
    except Exception as e:
        print(f"[warmup] Non-critical warmup error: {e}")
    finally:
        warmup_db.close()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] ATLAS API ready")
    
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ATLAS API shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Video Semantic Search API",
    description="Search video transcripts using semantic understanding and fuzzy matching",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




def get_video_qa(db: Session = Depends(get_db)):
    """Lazy loader for VideoQA."""
    global _video_qa
    if _video_qa is None:
        from llm.video_qa import VideoQA

        print("Initializing Video QA system (this may take a moment)...")
        _video_qa = VideoQA(db)
    else:
        _video_qa.update_db(db)
    return _video_qa


def get_search_engine(db: Session = Depends(get_db)):
    """Lazy loader for SemanticSearchEngine (singleton)."""
    global _search_engine
    if _search_engine is None:
        print("Initializing Semantic Search Engine (this may take a moment)...")
        _search_engine = SemanticSearchEngine(db)
    else:
        _search_engine.db = db
    return _search_engine


def get_mm_search_engine(db: Session = Depends(get_db)):
    """Lazy loader for MultiModalSearchEngine (singleton, reuses text search engine)."""
    global _mm_search_engine, _search_engine
    if _mm_search_engine is None:
        # Ensure the text search singleton exists first
        if _search_engine is None:
            _search_engine = SemanticSearchEngine(db)
        print("Initializing Multi-Modal Search Engine (this may take a moment)...")
        _mm_search_engine = MultiModalSearchEngine(db=db, text_search=_search_engine)
    else:
        _mm_search_engine.update_db(db)
    return _mm_search_engine


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    db_ok = test_connection()
    return {
        "status": "healthy" if db_ok else "unhealthy",
        "database": "ok" if db_ok else "error",
        **_server_capabilities(),
    }


# ══════════════════════════════════════════════════════════════════════════
#  AUTH ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=4)
    role: str = Field("viewer", pattern="^(admin|viewer)$")
    categories: List[str] = Field(default_factory=list)


class UpdateUserRequest(BaseModel):
    role: Optional[str] = Field(None, pattern="^(admin|viewer)$")
    categories: Optional[List[str]] = None
    password: Optional[str] = Field(None, min_length=4)


class UserInfoResponse(BaseModel):
    id: int
    username: str
    role: str
    categories: List[str]


@app.post("/auth/login")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate and return a JWT token."""
    user = db.query(User).filter(User.username == req.username).first()
    if not user or not verify_password(req.password, str(user.password_hash)):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token(user.id, str(user.username), str(user.role))
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.role,
            "categories": [a.category for a in user.category_access],
        },
    }


@app.get("/auth/me", response_model=UserInfoResponse)
async def auth_me(user: User = Depends(get_current_user)):
    """Return profile of the currently authenticated user."""
    return UserInfoResponse(
        id=user.id,
        username=user.username,
        role=user.role,
        categories=[a.category for a in user.category_access],
    )


@app.get("/auth/categories")
async def list_all_categories(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Return categories visible to the current user (admins see all, viewers see only allowed)."""
    # Only DB-backed categories (managed via the admin Category dropdown)
    db_cats = db.query(VideoCategory.name).order_by(VideoCategory.name).all()
    cats = {c.name for c in db_cats}
    # Restrict to user's allowed categories (admins get None → all)
    allowed = get_user_allowed_categories(user)
    if allowed is not None:
        cats = cats & allowed
    return sorted(cats)


@app.get("/auth/sites")
async def list_all_sites(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Return distinct non-null video labels (sites) visible to the current user."""
    q_videos = db.query(Video).filter(Video.label.isnot(None), Video.label != "")
    acl_filenames = _get_allowed_filenames(user, db)
    videos = q_videos.all()
    if acl_filenames is not None:
        videos = [v for v in videos if v.filename in acl_filenames]
    sites = sorted({v.label for v in videos})
    return sites


@app.get("/search/intent")
async def detect_search_intent(
    q: str = Query(..., description="Raw user query"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Detect whether a natural-language query is a category-browse or site-browse request.
    Returns {\"type\": \"category_browse\", \"category\": \"AkerBP\"}
    or     {\"type\": \"site_browse\",     \"site\": \"Yggdrasil\"}
    or     {\"type\": \"search\"}.
    """
    # Build set of known categories visible to this user (DB-backed only)
    db_cats = db.query(VideoCategory.name).all()
    cats = {c.name for c in db_cats}
    allowed = get_user_allowed_categories(user)
    if allowed is not None:
        cats = cats & allowed

    matched = _detect_category_intent(q, cats)
    if matched:
        return {"type": "category_browse", "category": matched}

    # Check for site (label) intent
    all_videos = db.query(Video).all()
    acl_filenames = _get_allowed_filenames(user, db)
    label_videos = [v for v in all_videos if v.label]
    if acl_filenames is not None:
        label_videos = [v for v in label_videos if v.filename in acl_filenames]
    known_sites = {v.label for v in label_videos}
    matched_site = _detect_site_intent(q, known_sites)
    if matched_site:
        return {"type": "site_browse", "site": matched_site}

    return {"type": "search"}


@app.post("/translate")
async def translate_text(request: Request):
    """
    Translate text using MyMemory free API.
    Body: {\"text\": \"...\", \"source\": \"en\", \"target\": \"no\"}
    """
    body = await request.json()
    text = body.get("text", "")
    source = body.get("source", "en")
    target = body.get("target", "no")
    if not text:
        return {"translated": ""}
    # MyMemory uses ISO language pairs as \"en|no\"
    lang_pair = f"{source}|{target}"
    api_url = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(text[:500])}&langpair={lang_pair}"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "ATLAS/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            translated = data.get("responseData", {}).get("translatedText", text)
            return {"translated": translated}
    except Exception:
        return {"translated": text}


# ── Admin: user management ────────────────────────────────────────────────


@app.get("/admin/users", response_model=List[UserInfoResponse])
async def list_users(
    admin: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """List all users (admin only)."""
    users = db.query(User).order_by(User.id).all()
    return [
        UserInfoResponse(
            id=u.id,
            username=u.username,
            role=u.role,
            categories=[a.category for a in u.category_access],
        )
        for u in users
    ]


@app.post("/admin/users", response_model=UserInfoResponse)
async def create_user(
    req: RegisterRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create a new user (admin only)."""
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status_code=409, detail="Username already exists")
    new_user = User(
        username=req.username,
        password_hash=hash_password(req.password),
        role=req.role,
    )
    db.add(new_user)
    db.flush()  # get new_user.id
    for cat in req.categories:
        db.add(UserCategoryAccess(user_id=new_user.id, category=cat))
    db.commit()
    db.refresh(new_user)
    return UserInfoResponse(
        id=new_user.id,
        username=new_user.username,
        role=new_user.role,
        categories=[a.category for a in new_user.category_access],
    )


@app.put("/admin/users/{user_id}", response_model=UserInfoResponse)
async def update_user(
    user_id: int,
    req: UpdateUserRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Update an existing user's role, password, or category access (admin only)."""
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if req.role is not None:
        target.role = req.role
    if req.password is not None:
        target.password_hash = hash_password(req.password)
    if req.categories is not None:
        # Replace category list
        db.query(UserCategoryAccess).filter(
            UserCategoryAccess.user_id == user_id
        ).delete()
        for cat in req.categories:
            db.add(UserCategoryAccess(user_id=user_id, category=cat))
    db.commit()
    db.refresh(target)
    return UserInfoResponse(
        id=target.id,
        username=target.username,
        role=target.role,
        categories=[a.category for a in target.category_access],
    )


@app.delete("/admin/users/{user_id}")
async def delete_user(
    user_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Delete a user (admin only). Cannot delete yourself."""
    if admin.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(target)
    db.commit()
    return {"detail": f"User '{target.username}' deleted"}


# ── Admin: video upload, categories, ground truth, pipeline config ─────

# Allowed video extensions for upload
_ALLOWED_VIDEO_EXT = {
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".webm",
    ".flv",
    ".wmv",
    ".ts",
    ".m2ts",
}

# Available pipeline models (used for frontend dropdowns)
TRANSCRIPTION_MODELS = [
    {
        "id": "whisper-tiny",
        "label": "Whisper Tiny",
        "backend": "whisper",
        "variant": {"name": "tiny"},
    },
    {
        "id": "whisper-base",
        "label": "Whisper Base",
        "backend": "whisper",
        "variant": {"name": "base"},
    },
    {
        "id": "whisper-small",
        "label": "Whisper Small",
        "backend": "whisper",
        "variant": {"name": "small"},
    },
    {
        "id": "whisper-medium",
        "label": "Whisper Medium",
        "backend": "whisper",
        "variant": {"name": "medium"},
    },
    {
        "id": "whisper-large",
        "label": "Whisper Large",
        "backend": "whisper",
        "variant": {"name": "large"},
    },
    {
        "id": "whisperx-base",
        "label": "WhisperX Base",
        "backend": "whisperx",
        "variant": {"name": "base"},
    },
    {
        "id": "whisperx-large",
        "label": "WhisperX Large",
        "backend": "whisperx",
        "variant": {"name": "large"},
    },
    {
        "id": "distil-whisper",
        "label": "Distil-Whisper Large v3",
        "backend": "distil-whisper",
        "variant": {},
    },
    {"id": "vosk-en", "label": "Vosk English", "backend": "vosk", "variant": {}},
    {
        "id": "canary-1b",
        "label": "NVIDIA Canary 1B",
        "backend": "canary",
        "variant": {},
    },
    {
        "id": "parakeet-ctc-1.1b",
        "label": "NVIDIA Parakeet CTC 1.1B",
        "backend": "parakeet",
        "variant": {"name": "parakeet-ctc-1.1b"},
    },
    {
        "id": "parakeet-ctc-0.6b",
        "label": "NVIDIA Parakeet CTC 0.6B",
        "backend": "parakeet",
        "variant": {"name": "parakeet-ctc-0.6b"},
    },
    {
        "id": "google-medasr",
        "label": "Google MedASR",
        "backend": "medasr",
        "variant": {},
    },
    {
        "id": "speecht5-asr",
        "label": "Microsoft SpeechT5 ASR",
        "backend": "speecht5",
        "variant": {},
    },
    {
        "id": "wav2vec2",
        "label": "Facebook Wav2Vec2",
        "backend": "wav2vec",
        "variant": {},
    },
    {
        "id": "qwen3-asr-1.7b",
        "label": "Qwen3 ASR 1.7B",
        "backend": "qwen",
        "variant": {"name": "qwen3-asr-1.7b"},
    },
    {
        "id": "qwen3-asr-0.6b",
        "label": "Qwen3 ASR 0.6B",
        "backend": "qwen",
        "variant": {"name": "qwen3-asr-0.6b"},
    },
    {
        "id": "vibevoice",
        "label": "Microsoft VibeVoice ASR",
        "backend": "vibevoice",
        "variant": {},
    },
    {
        "id": "voxtral-mini",
        "label": "Mistral Voxtral Mini 4B",
        "backend": "voxtral",
        "variant": {},
    },
]

SCENE_DETECTION_MODELS = [
    {"id": "pyscenedetect", "label": "PySceneDetect (ContentDetector)"},
    {"id": "transnetv2", "label": "TransNetV2 (Shot Boundary)"},
]

EMBEDDING_MODELS = [
    {"id": "bge-m3", "label": "BAAI/bge-m3 (1024-dim)"},
    {"id": "qwen3-embedding-0.6b", "label": "Qwen3-Embedding-0.6B"},
]

VISION_MODELS = [
    {"id": "siglip-base", "label": "google/siglip-base-patch16-224"},
    {"id": "clip-vit-b32", "label": "CLIP ViT-B/32 (OpenAI)"},
]


@app.get("/admin/pipeline-models")
async def get_pipeline_models(admin: User = Depends(require_admin)):
    """Return all available model choices for the pipeline configuration dropdowns."""
    return {
        "transcription": TRANSCRIPTION_MODELS,
        "scene_detection": SCENE_DETECTION_MODELS,
        "embedding": EMBEDDING_MODELS,
        "vision": VISION_MODELS,
    }


@app.post("/admin/categories")
async def create_category(
    req: dict,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create a new video category. Body: {"name": "My Category"}"""
    name = (req.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Category name is required")
    existing = db.query(VideoCategory).filter(VideoCategory.name == name).first()
    if not existing:
        cat = VideoCategory(name=name)
        db.add(cat)
        db.commit()
        db.refresh(cat)
        return {"id": cat.id, "category": cat.name}
    return {"id": existing.id, "category": existing.name}


@app.get("/admin/video-categories")
async def list_video_categories(
    admin: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """List all video categories."""
    cats = db.query(VideoCategory).order_by(VideoCategory.name).all()
    return [{"id": c.id, "name": c.name} for c in cats]


@app.delete("/admin/video-categories/{cat_id}")
async def delete_video_category(
    cat_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """Delete a video category. Videos in this category will have category_id set to NULL."""
    cat = db.query(VideoCategory).filter(VideoCategory.id == cat_id).first()
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")
    db.delete(cat)
    db.commit()
    return {"detail": f"Category '{cat.name}' deleted"}


class UpdateVideoRequest(BaseModel):
    label: Optional[str] = None
    category_id: Optional[int] = None


@app.put("/videos/{video_id}")
async def update_video_metadata(
    video_id: int,
    req: UpdateVideoRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Set label and/or category on a video (admin only)."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if req.label is not None:
        video.label = req.label.strip() or None
    if req.category_id is not None:
        if req.category_id == 0:
            video.category_id = None
        else:
            cat = (
                db.query(VideoCategory)
                .filter(VideoCategory.id == req.category_id)
                .first()
            )
            if not cat:
                raise HTTPException(status_code=404, detail="Category not found")
            video.category_id = cat.id
    db.commit()
    db.refresh(video)
    return {
        "id": video.id,
        "filename": video.filename,
        "label": video.label,
        "category": video.category_rel.name if video.category_rel else None,
        "category_id": video.category_id,
    }


@app.post("/admin/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    category: str = Query("Other", description="Category for the video"),
    admin: User = Depends(require_admin),
):
    """Upload a video file to the videos/ directory."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in _ALLOWED_VIDEO_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_VIDEO_EXT))}",
        )

    project_root = Path(__file__).parent.parent
    videos_dir = project_root / "videos"
    videos_dir.mkdir(exist_ok=True)

    # Sanitise filename: keep only safe characters
    import re as _re

    safe_name = _re.sub(r"[^\w\s\-\.\(\)]", "", file.filename)
    if not safe_name:
        safe_name = f"upload_{int(time.time())}{ext}"

    dest = videos_dir / safe_name
    if dest.exists():
        raise HTTPException(
            status_code=409, detail=f"File '{safe_name}' already exists"
        )

    # Stream file to disk (avoid loading entire file into memory)
    total = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            f.write(chunk)
            total += len(chunk)

    return {
        "filename": safe_name,
        "size_mb": round(total / (1024 * 1024), 2),
        "category": category,
        "path": str(dest),
    }


@app.post("/admin/upload-ground-truth")
async def upload_ground_truth(
    file: UploadFile = File(...),
    admin: User = Depends(require_admin),
):
    """Upload a ground-truth JSON file to the ground_truth/ directory."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if not file.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are accepted")

    project_root = Path(__file__).parent.parent
    gt_dir = project_root / "ground_truth"
    gt_dir.mkdir(exist_ok=True)

    import re as _re

    safe_name = _re.sub(r"[^\w\s\-\.\(\)]", "", file.filename)
    if not safe_name:
        safe_name = f"gt_{int(time.time())}.json"

    dest = gt_dir / safe_name

    content = await file.read()
    # Validate JSON
    try:
        json.loads(content)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON content")

    with open(dest, "wb") as f:
        f.write(content)

    return {"filename": safe_name, "size_bytes": len(content), "path": str(dest)}


@app.get("/admin/ground-truths")
async def list_ground_truths(admin: User = Depends(require_admin)):
    """List all ground-truth files."""
    project_root = Path(__file__).parent.parent
    gt_dir = project_root / "ground_truth"
    if not gt_dir.exists():
        return []
    files = []
    for f in sorted(gt_dir.glob("*.json")):
        files.append({"filename": f.name, "size_bytes": f.stat().st_size})
    return files


@app.post("/admin/run-pipeline")
async def run_pipeline(
    req: dict,
    admin: User = Depends(require_admin),
):
    """
    Trigger the pipeline for a specific video with selected models.
    Body: {
        "filename": "video.mp4",
        "transcription_model": "whisper-base",
        "scene_detection": "pyscenedetect",
        "scene_threshold": 30.0,
        "device": "auto"
    }
    Returns immediately with status; pipeline runs synchronously on the server.
    """
    filename = (req.get("filename") or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="filename is required")

    project_root = Path(__file__).parent.parent
    video_path = project_root / "videos" / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {filename}")

    # Resolve transcription model
    model_id = req.get("transcription_model", "whisper-base")
    model_entry = next((m for m in TRANSCRIPTION_MODELS if m["id"] == model_id), None)
    if not model_entry:
        raise HTTPException(
            status_code=400, detail=f"Unknown transcription model: {model_id}"
        )

    scene_threshold = float(req.get("scene_threshold", 30.0))
    device = req.get("device", "auto")

    # Run pipeline in a background thread to avoid blocking the event loop
    import concurrent.futures

    def _run():
        from basic_pipeline import BasicVideoPipeline

        pipe = BasicVideoPipeline(
            backend=model_entry["backend"],
            model_variant=model_entry["variant"] or None,
            scene_threshold=scene_threshold,
            device=device,
        )
        return pipe.process_video(str(video_path))

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, _run)

    return {
        "status": "completed",
        "filename": filename,
        "model": model_entry["label"],
        "result_summary": {
            "segments": result.get("num_segments", 0)
            if isinstance(result, dict)
            else 0,
            "scenes": result.get("num_scenes", 0) if isinstance(result, dict) else 0,
        },
    }


@app.get("/video/stream/{video_id}")
async def stream_video(
    video_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Stream video file with support for range requests (seeking).

    This endpoint allows the frontend to play videos directly in the browser
    and seek to specific timestamps from search results.
    """
    # Get video from database
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Access control: check category
    if not user_can_access_video(user, video.filename):
        raise HTTPException(status_code=403, detail="Access denied to this video")

    resolved_path = _resolve_video_file_path(video.file_path)
    if resolved_path is None:
        raise HTTPException(
            status_code=404, detail=f"Video file not found: {video.file_path}"
        )
    video_path = str(resolved_path)

    file_size = os.path.getsize(video_path)

    # For .ts and other browser-incompatible formats, redirect to the transcode endpoint
    ext = os.path.splitext(video_path)[1].lower()
    if ext in TRANSCODE_EXTENSIONS:
        from fastapi.responses import RedirectResponse

        # Preserve token query param so the redirected request stays authenticated
        qs = request.url.query
        redirect_url = f"/video/transcode/{video_id}"
        if qs:
            redirect_url += f"?{qs}"
        return RedirectResponse(url=redirect_url)

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
                    read_size = min(262144, remaining)
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
                while chunk := f.read(262144):
                    yield chunk

        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": content_type,
        }

        return StreamingResponse(iterfile(), headers=headers)


@app.get("/video/transcode/{video_id}")
async def transcode_video(
    video_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
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

    # Security check: can user access this video's category?
    if not user_can_access_video(user, video.filename):
        raise HTTPException(
            status_code=403, detail="You do not have permission to access this video"
        )

    resolved_path = _resolve_video_file_path(video.file_path)
    if resolved_path is None:
        raise HTTPException(
            status_code=404, detail=f"Video file not found: {video.file_path}"
        )
    video_path = str(resolved_path)

    async def stream_ffmpeg():
        """Pipe ffmpeg stdout as a fragmented MP4 stream."""
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",  # suppress progress spam
            "-i",
            video_path,  # input file
            "-c:v",
            "copy",  # copy video stream (no re-encode → fast)
            "-c:a",
            "aac",  # re-encode audio to AAC for browser compat
            "-f",
            "mp4",  # output container
            "-movflags",
            "frag_keyframe+empty_moov+faststart",  # streaming-safe fragmented MP4
            "pipe:1",  # write to stdout
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
                detail="ffmpeg is not installed or not on PATH. Install ffmpeg to play .ts files.",
            )

    return StreamingResponse(
        stream_ffmpeg(),
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'inline; filename="{Path(video_path).stem}.mp4"',
            "X-Content-Type-Options": "nosniff",
        },
    )


def format_vtt_timestamp(seconds: float) -> str:
    """Format a float seconds value to WebVTT timestamp format (HH:MM:SS.mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


@app.get("/video/subtitles/{video_id}", response_class=PlainTextResponse)
async def get_video_subtitles(
    video_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    """
    Serve video subtitles in WebVTT format directly from the database.
    """
    from database.models import TranscriptSegment

    # Check if video exists
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    segments = (
        db.query(TranscriptSegment)
        .filter(TranscriptSegment.video_id == video_id)
        .order_by(TranscriptSegment.start_time)
        .all()
    )

    if not segments:
        return "WEBVTT\n\n"

    # Build WebVTT string
    vtt = ["WEBVTT\n"]
    for i, seg in enumerate(segments, 1):
        start = format_vtt_timestamp(seg.start_time)
        end = format_vtt_timestamp(seg.end_time)
        vtt.append(f"\n{i}")
        vtt.append(f"{start} --> {end}")
        vtt.append(seg.text.strip())

    return "\n".join(vtt)


@app.get("/videos", response_model=List[VideoInfo])
async def list_videos(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    """
    List videos the current user is allowed to see.
    Admins see all; viewers see only their assigned categories.
    """
    return [
        _serialize_video_info(video)
        for video, _ in _get_accessible_available_videos(user, db)
    ]


@app.get("/videos/count")
async def count_videos(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    """Return the accessible count of currently available videos."""
    return {"count": len(_get_accessible_available_videos(user, db))}


@app.post("/qa/ask", response_model=QA_Response)
async def ask_video_question(
    request: QARequest,
    qa_system=Depends(get_video_qa),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Ask a natural language question about the available videos.
    Uses RAG (Retrieval-Augmented Generation) to answer based on transcripts and visual semantics.
    """
    try:
        # Run the blocking model inference in a thread pool to avoid
        # freezing the event loop (which blocks all other requests).
        result = await asyncio.to_thread(
            qa_system.ask,
            question=request.question,
            video_filter=request.video_filter,
            top_k=request.top_k,
        )
        # Filter citations to only include videos the user can access
        allowed_filenames = _get_allowed_filenames(user, db)
        if allowed_filenames is not None and result.get("citations"):
            result["citations"] = [
                c
                for c in result["citations"]
                if c.get("video_filename") in allowed_filenames
            ]
        return result
    except Exception as e:
        print(f"QA Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    search_engine: SemanticSearchEngine = Depends(get_search_engine),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Semantic search endpoint.

    Search video transcripts using hybrid semantic + fuzzy text matching.
    """
    start_time = time.time()

    try:
        allowed_filenames = _get_allowed_filenames(user, db)
        results = search_engine.search(
            query=request.query,
            top_k=request.top_k * 3 if allowed_filenames is not None else request.top_k,
            semantic_weight=request.semantic_weight,
            text_weight=request.text_weight,
            min_score=request.min_score,
            video_filter=request.video_filter,
            log_query=True,
        )
        results = _filter_results(results, allowed_filenames, limit=request.top_k)

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
    category: Optional[List[str]] = Query(
        None, description="Filter by video category name(s)"
    ),
    label: Optional[str] = Query(
        None, description="Filter by video label (substring match)"
    ),
    site: Optional[List[str]] = Query(
        None, description="Filter by site name (exact label match)"
    ),
    facet: str = Query(
        "auto",
        description="Optional meaning facet: auto, oil_gas, tools, analytics",
    ),
    search_engine: SemanticSearchEngine = Depends(get_search_engine),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Quick search endpoint (GET request for easy testing).
    Supports filtering by category, label, and site.
    ```
    GET /search/quick?q=Omega+Alpha+well&limit=5&category=Installation&site=Yggdrasil
    ```
    """
    start_time = time.time()

    # Build set of allowed filenames: enforce user category access + optional category/label/site UI filter
    acl_filenames = _get_allowed_filenames(user, db)
    extra_filter = None
    cats = [c for c in (category or []) if c]  # flatten empty strings
    sites = [s for s in (site or []) if s]
    if cats or label or sites:
        q_videos = db.query(Video)
        if cats:
            q_videos = q_videos.join(
                VideoCategory, Video.category_id == VideoCategory.id
            ).filter(VideoCategory.name.in_(cats))
        if label:
            q_videos = q_videos.filter(Video.label.ilike(f"%{label}%"))
        if sites:
            q_videos = q_videos.filter(Video.label.in_(sites))
        extra_filter = {v.filename for v in q_videos.all()}

    # Merge: if both ACL and UI filter exist, intersect them
    if acl_filenames is not None and extra_filter is not None:
        allowed_filenames = acl_filenames & extra_filter
    elif acl_filenames is not None:
        allowed_filenames = acl_filenames
    else:
        allowed_filenames = extra_filter

    try:
        fallback_data = search_engine.search_with_fallback(
            query=q,
            top_k=limit * 3 if allowed_filenames else limit,
            video_filter=video,
            facet=facet or "auto",
        )

        results = _filter_results(
            fallback_data["results"], allowed_filenames, limit=limit
        )
        metadata = fallback_data["search_metadata"]
        search_time = time.time() - start_time

        # Build grouped-by-video structure
        result_dicts = [r.to_dict() for r in results]
        grouped_by_video = {}
        for rd in result_dicts:
            vid = rd["video_id"]
            if vid not in grouped_by_video:
                grouped_by_video[vid] = {
                    "video_id": vid,
                    "video_filename": rd["video_filename"],
                    "occurrences": [],
                }
            grouped_by_video[vid]["occurrences"].append(rd)
        grouped_results = list(grouped_by_video.values())

        return {
            "query": q,
            "results_count": len(result_dicts),
            "results": result_dicts,
            "grouped_results": grouped_results,
            "search_time_seconds": round(search_time, 3),
            "search_strategy": metadata.get("search_strategy"),
            "search_message": metadata.get("search_message"),
            "did_you_mean": metadata.get("did_you_mean"),
            "sense_suggestions": metadata.get("sense_suggestions") or [],
            "facets": metadata.get("facets") or [],
            "facet_applied": metadata.get("facet_applied") or (facet or "auto"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/search/browse")
async def browse_by_category(
    category: Optional[List[str]] = Query(
        None, description="Category name(s) to browse"
    ),
    site: Optional[List[str]] = Query(None, description="Site/label name(s) to browse"),
    limit: int = Query(10, description="Max videos to return", ge=1, le=50),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Browse videos (with their transcript snippets) by category and/or site.
    No text query needed — just returns the latest segments from matching videos.
    """
    start_time = time.time()

    cats = [c for c in (category or []) if c]
    sites = [s for s in (site or []) if s]
    if not cats and not sites:
        raise HTTPException(
            status_code=400, detail="At least one category or site is required"
        )

    # Build query based on category and/or site filters
    q_videos = db.query(Video)
    if cats:
        q_videos = q_videos.join(
            VideoCategory, Video.category_id == VideoCategory.id
        ).filter(VideoCategory.name.in_(cats))
    if sites:
        q_videos = q_videos.filter(Video.label.in_(sites))

    # Enforce ACL
    acl_filenames = _get_allowed_filenames(user, db)
    matched_videos = q_videos.all()
    if acl_filenames is not None:
        matched_videos = [v for v in matched_videos if v.filename in acl_filenames]
    matched_videos = [
        v for v in matched_videos if _resolve_video_file_path(v.file_path) is not None
    ]

    matched_videos = matched_videos[:limit]

    # Build grouped results with a few transcript snippets per video
    grouped_results = []
    result_dicts = []
    for v in matched_videos:
        segments = (
            db.query(TranscriptSegment)
            .filter(TranscriptSegment.video_id == v.id)
            .order_by(TranscriptSegment.start_time)
            .limit(5)
            .all()
        )
        occurrences = []
        for seg in segments:
            rd = {
                "video_id": v.id,
                "video_filename": v.filename,
                "segment_index": seg.segment_index,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "text": seg.text,
                "score": 1.0,
            }
            occurrences.append(rd)
            result_dicts.append(rd)
        grouped_results.append(
            {
                "video_id": v.id,
                "video_filename": v.filename,
                "occurrences": occurrences,
            }
        )

    search_time = time.time() - start_time
    browse_parts = []
    if cats:
        browse_parts.append(", ".join(cats))
    if sites:
        browse_parts.append(", ".join(sites))
    return {
        "query": f"[Browse: {' / '.join(browse_parts)}]",
        "results_count": len(result_dicts),
        "results": result_dicts,
        "grouped_results": grouped_results,
        "search_time_seconds": round(search_time, 3),
        "search_strategy": "category_browse",
        "sense_suggestions": [],
        "facets": [],
    }


@app.get("/search/exact")
async def exact_search(
    phrase: str = Query(..., description="Exact phrase to search", min_length=1),
    video: Optional[str] = Query(None, description="Filter by video filename"),
    search_engine: SemanticSearchEngine = Depends(get_search_engine),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Exact phrase search (case-insensitive).
    """
    start_time = time.time()

    try:
        results = search_engine.search_exact_phrase(phrase=phrase, video_filter=video)
        allowed_filenames = _get_allowed_filenames(user, db)
        results = _filter_results(results, allowed_filenames)

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
async def multimodal_search(
    request: MultiModalSearchRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Multi-modal search endpoint combining text and vision.

    Search using **both transcript text and visual content** from keyframes.
    This provides more accurate results by matching both what was said and what was shown.

    **Example queries:**
    - "drilling rig" → Finds both mentions AND visual appearances of drilling rigs
    - "safety equipment" → Finds helmets, vests even if not mentioned
    - "offshore platform" → Matches visual scenes + spoken references

    **Returns:**
    Results with both text_score and vision_score for transparency.
    """
    start_time = time.time()

    try:
        text_weight = request.text_weight
        vision_weight = request.vision_weight

        # Reuse singleton multi-modal search engine
        global _mm_search_engine, _search_engine
        if _mm_search_engine is None:
            if _search_engine is None:
                _search_engine = SemanticSearchEngine(db)
            _mm_search_engine = MultiModalSearchEngine(
                db=db, text_search=_search_engine
            )
        _mm_search_engine.update_db(db)
        _mm_search_engine.text_weight = text_weight
        _mm_search_engine.vision_weight = vision_weight
        mm_search = _mm_search_engine

        allowed_filenames = _get_allowed_filenames(user, db)

        # Perform search with fallback (includes LLM intent parsing)
        fallback_data = mm_search.search_with_fallback(
            query=request.query,
            top_k=request.top_k * 3 if allowed_filenames is not None else request.top_k,
            video_filter=request.video_filter,
            use_llm=request.use_llm,
        )

        results = _filter_results(
            fallback_data["results"], allowed_filenames, limit=request.top_k
        )
        metadata = fallback_data["search_metadata"]

        search_time = time.time() - start_time

        return SearchResponse(
            query=request.query,
            results_count=len(results),
            results=[r.to_dict() if hasattr(r, "to_dict") else r for r in results],
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
            if _search_engine is None:
                _search_engine = SemanticSearchEngine(db)
            _search_engine.db = db
            allowed_filenames = _get_allowed_filenames(user, db)
            results = _search_engine.search(
                query=request.query,
                top_k=request.top_k * 3
                if allowed_filenames is not None
                else request.top_k,
                video_filter=request.video_filter,
            )
            results = _filter_results(results, allowed_filenames, limit=request.top_k)
            search_time = time.time() - start_time
            return SearchResponse(
                query=request.query,
                results_count=len(results),
                results=[r.to_dict() for r in results],
                search_time_seconds=round(search_time, 3),
            )
        else:
            raise HTTPException(
                status_code=500, detail=f"Multi-modal search failed: {str(e)}"
            )


@app.get("/search/multimodal/quick")
async def quick_multimodal_search(
    q: str = Query(..., description="Search query", min_length=1),
    limit: int = Query(10, description="Number of results", ge=1, le=50),
    use_llm: bool = Query(
        False, description="Use LLM for intent parsing (disable for speed)"
    ),
    facet: str = Query(
        "auto",
        description="Optional meaning facet: auto, oil_gas, tools, analytics",
    ),
    video: Optional[str] = Query(None, description="Filter by video filename"),
    category: Optional[List[str]] = Query(
        None, description="Filter by video category name(s)"
    ),
    label: Optional[str] = Query(
        None, description="Filter by video label (substring match)"
    ),
    site: Optional[List[str]] = Query(
        None, description="Filter by site name (exact label match)"
    ),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Quick multi-modal search (GET request for easy testing).
    Uses the default balanced profile for the main app experience.
    Supports filtering by category, label, and site.

    **Example:**
    ```
    GET /search/multimodal/quick?q=drilling+techniques&limit=5&site=Yggdrasil
    ```
    """
    # Build set of allowed filenames: enforce user category access + optional category/label/site UI filter
    acl_filenames = _get_allowed_filenames(user, db)
    extra_filter = None
    cats = [c for c in (category or []) if c]  # flatten empty strings
    sites = [s for s in (site or []) if s]
    if cats or label or sites:
        q_videos = db.query(Video)
        if cats:
            q_videos = q_videos.join(
                VideoCategory, Video.category_id == VideoCategory.id
            ).filter(VideoCategory.name.in_(cats))
        if label:
            q_videos = q_videos.filter(Video.label.ilike(f"%{label}%"))
        if sites:
            q_videos = q_videos.filter(Video.label.in_(sites))
        extra_filter = {v.filename for v in q_videos.all()}

    # Merge: if both ACL and UI filter exist, intersect them
    if acl_filenames is not None and extra_filter is not None:
        allowed_filenames = acl_filenames & extra_filter
    elif acl_filenames is not None:
        allowed_filenames = acl_filenames
    else:
        allowed_filenames = extra_filter

    try:
        text_weight, vision_weight = 0.5, 0.5

        start_time = time.time()

        # Reuse singleton multi-modal search engine
        global _mm_search_engine, _search_engine
        if _mm_search_engine is None:
            if _search_engine is None:
                _search_engine = SemanticSearchEngine(db)
            _mm_search_engine = MultiModalSearchEngine(
                db=db, text_search=_search_engine
            )
        _mm_search_engine.update_db(db)
        _mm_search_engine.text_weight = text_weight
        _mm_search_engine.vision_weight = vision_weight
        mm_search = _mm_search_engine

        fallback_data = mm_search.search_with_fallback(
            query=q,
            top_k=limit * 3 if allowed_filenames else limit,
            video_filter=video,
            use_llm=use_llm,
            facet=facet or "auto",
        )

        results = _filter_results(
            fallback_data["results"], allowed_filenames, limit=limit
        )
        metadata = fallback_data["search_metadata"]
        search_time = time.time() - start_time

        # Build grouped-by-video structure so the frontend can show all occurrences
        result_dicts = [r.to_dict() for r in results]
        grouped_by_video = {}
        for rd in result_dicts:
            vid = rd["video_id"]
            if vid not in grouped_by_video:
                grouped_by_video[vid] = {
                    "video_id": vid,
                    "video_filename": rd["video_filename"],
                    "occurrences": [],
                }
            grouped_by_video[vid]["occurrences"].append(rd)
        grouped_results = list(grouped_by_video.values())

        return {
            "query": q,
            "weights": {"text": text_weight, "vision": vision_weight},
            "results_count": len(result_dicts),
            "results": result_dicts,
            "grouped_results": grouped_results,
            "search_time_seconds": round(search_time, 3),
            "search_strategy": metadata.get("search_strategy"),
            "search_message": metadata.get("search_message"),
            "did_you_mean": metadata.get("did_you_mean"),
            "sense_suggestions": metadata.get("sense_suggestions") or [],
            "llm_intent": metadata.get("llm_intent"),
            "facets": metadata.get("facets") or [],
            "facet_applied": metadata.get("facet_applied") or (facet or "auto"),
        }

    except Exception as e:
        # Log the full error for debugging
        print(f"\nQuick multi-modal search error: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()

        # Graceful fallback to text-only
        if any(w in str(e).lower() for w in ["vision", "clip", "siglip", "embedding"]):
            print(f"Vision search unavailable, using text-only: {e}")
            if _search_engine is None:
                _search_engine = SemanticSearchEngine(db)
            _search_engine.db = db
            fallback_data = _search_engine.search_with_fallback(
                query=q,
                top_k=limit * 3 if allowed_filenames else limit,
                video_filter=video,
                facet=facet or "auto",
            )
            results = _filter_results(
                fallback_data["results"], allowed_filenames, limit=limit
            )
            metadata = fallback_data["search_metadata"]
            search_time = time.time() - start_time

            result_dicts = [r.to_dict() for r in results]
            grouped_by_video = {}
            for rd in result_dicts:
                vid = rd["video_id"]
                if vid not in grouped_by_video:
                    grouped_by_video[vid] = {
                        "video_id": vid,
                        "video_filename": rd["video_filename"],
                        "occurrences": [],
                    }
                grouped_by_video[vid]["occurrences"].append(rd)

            return {
                "query": q,
                "weights": {"text": 1.0, "vision": 0.0},
                "results_count": len(result_dicts),
                "results": result_dicts,
                "grouped_results": list(grouped_by_video.values()),
                "search_time_seconds": round(search_time, 3),
                "search_strategy": metadata.get("search_strategy"),
                "search_message": metadata.get("search_message"),
                "did_you_mean": metadata.get("did_you_mean"),
                "sense_suggestions": metadata.get("sense_suggestions") or [],
            }
        else:
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/search/visual/image")
async def visual_image_search(
    file: UploadFile = File(...),
    limit: int = Query(10, description="Number of results", ge=1, le=50),
    video: Optional[str] = Query(None, description="Filter by video filename"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
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
            image_input=image_bytes, top_k=limit, video_filter=video
        )
        allowed_filenames = _get_allowed_filenames(user, db)
        results = _filter_results(results, allowed_filenames)

        search_time = time.time() - start_time

        # Cache image embedding for re-ranking / "find more like this"
        try:
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            existing_cache = (
                db.query(SearchImageCache).filter_by(image_hash=image_hash).first()
            )
            if existing_cache:
                existing_cache.search_count += 1
                existing_cache.last_used = datetime.utcnow()
            else:
                vision_embedding = visual_engine.vision_model.encode_image(
                    image_bytes, normalize=True
                )
                cache_entry = SearchImageCache(
                    filename=file.filename or "uploaded_image",
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
        raise HTTPException(
            status_code=500, detail=f"Visual image search failed: {str(e)}"
        )


@app.post("/search/visual/combined")
async def visual_combined_search(
    file: UploadFile = File(...),
    text_query: str = Query("", description="Optional text to refine image search"),
    limit: int = Query(10, description="Number of results", ge=1, le=50),
    video: Optional[str] = Query(None, description="Filter by video filename"),
    image_weight: float = Query(0.7, description="Image embedding weight", ge=0, le=1),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
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
                text_weight=text_weight,
            )
        else:
            results = visual_engine.search_by_image(
                image_input=image_bytes, top_k=limit, video_filter=video
            )

        allowed_filenames = _get_allowed_filenames(user, db)
        results = _filter_results(results, allowed_filenames)

        search_time = time.time() - start_time

        # Log
        try:
            query_log = SearchQuery(
                query_text=f"[IMAGE+TEXT] {file.filename}: {text_query}"
                if text_query.strip()
                else f"[IMAGE] {file.filename}",
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
            "search_type": "combined_image_text"
            if text_query.strip()
            else "reverse_image_search",
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
    user: User = Depends(get_current_user),
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
        results = visual_engine.search_visual(query=q, top_k=limit, video_filter=video)
        allowed_filenames = _get_allowed_filenames(user, db)
        results = _filter_results(results, allowed_filenames)

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
async def hybrid_search():
    """
    Truly hybrid search - combines text + semantic + visual!

    **Auto-detection:**
    - "picture of oil rig" → Visual-heavy (70% visual, 30% text)
    - "discussed drilling" → Text-heavy (70% text, 30% visual)
    - "oil rig" → Balanced (50% text, 50% visual)

    Deprecated endpoint retained only to return a migration hint.
    """
    raise HTTPException(
        status_code=410,
        detail="`/search/hybrid` is deprecated. Use `/search/quick` for main search or `/search/multimodal/quick` explicitly.",
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  OpenAI-compatible chat completions endpoints                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "ATLAS"
    messages: List[ChatMessage]
    stream: bool = True
    max_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    language: Optional[str] = (
        None  # e.g. "Norwegian", "English", or None for auto-detect
    )


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model list."""
    return {
        "object": "list",
        "data": [
            {
                "id": "ATLAS",
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
    user: User = Depends(get_current_user),
):
    """
    OpenAI-compatible chat completions endpoint.

    Workflow:
    1. Extract the latest user message as the question
    2. Check system message for optional `video:<filename>` scoping directive
    3. Run VideoQA RAG: semantic search → Qwen2.5-1.5B generates grounded answer
    4. Stream back tokens via SSE (or return full JSON if stream=False)
    """
    from llm.video_qa_streaming import get_streaming_qa

    # ── Extract the user's question ──────────────────────────────────────
    user_question = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_question = msg.content.strip()
            break

    if not user_question:
        raise HTTPException(
            status_code=400, detail="No user message found in messages."
        )

    # ── Check system prompt for video scoping directive ──────────────────
    # e.g. system: "video:AkerBP_2.mp4" → restricts search to that file
    video_filter: Optional[str] = None
    language: Optional[str] = request.language
    for msg in request.messages:
        if msg.role == "system":
            import re

            m = re.search(r"video:\s*([^\s]+)", msg.content, re.IGNORECASE)
            if m:
                video_filter = m.group(1).strip()
            # Also check for language directive in system message
            if not language:
                lang_m = re.search(r"language:\s*([^\n]+)", msg.content, re.IGNORECASE)
                if lang_m:
                    language = lang_m.group(1).strip()
            break

    # ── Load StreamingVideoQA (singleton, lazy) ──────────────────────────
    try:
        qa = get_streaming_qa(db=db)
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"VideoQA system could not be initialised: {e}"
        )

    # ── Streaming response (default: stream=True) ───────────────────────
    if request.stream:

        async def sse_generator():
            try:
                for chunk in qa.stream_ask(
                    question=user_question,
                    video_filter=video_filter,
                    max_new_tokens=request.max_tokens,
                    language=language,
                ):
                    yield chunk
                    # Small yield point so FastAPI can flush SSE chunks
                    await asyncio.sleep(0)
            except Exception as e:
                import traceback

                traceback.print_exc()
                yield f'data: {{"error": "{str(e)}"}}\n\n'

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    # ── Non-streaming response ────────────────────────────────────────────
    try:
        result = qa.ask_sync(
            question=user_question,
            video_filter=video_filter,
            top_k=5,
            language=language,
        )
        return result
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/video/thumbnail/{video_id}")
async def get_video_thumbnail(
    video_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    """Return the first keyframe of a video as a thumbnail image."""
    from database.models import Scene as SceneModel
    from pathlib import Path as FilePath

    # Get the first scene for this video (smallest start_time)
    scene = (
        db.query(SceneModel)
        .filter(
            SceneModel.video_id == video_id,
            SceneModel.keyframe_path.isnot(None),
        )
        .order_by(SceneModel.start_time)
        .first()
    )

    if not scene or not scene.keyframe_path:
        raise HTTPException(
            status_code=404, detail="No thumbnail available for this video"
        )

    # Resolve path — try absolute first, then relative to project root
    kf = FilePath(scene.keyframe_path)
    if not kf.exists():
        project_root = FilePath(__file__).parent.parent
        kf = project_root / scene.keyframe_path
    if not kf.exists():
        raise HTTPException(status_code=404, detail="Keyframe file not found on disk")

    suffix = kf.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(status_code=400, detail="Invalid keyframe file type")

    media_type = f"image/{suffix.lstrip('.').replace('jpg', 'jpeg')}"
    return FileResponse(str(kf), media_type=media_type)


@app.get("/keyframe")
async def serve_keyframe(
    path: str = Query(..., description="Path to keyframe image"),
    user: User = Depends(get_current_user),
):
    """Serve keyframe images for thumbnails in search results."""
    from pathlib import Path as FilePath

    keyframe_path = FilePath(path)
    if not keyframe_path.exists():
        raise HTTPException(status_code=404, detail="Keyframe not found")

    # Basic security: only serve image files
    if keyframe_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(status_code=400, detail="Invalid file type")

    return FileResponse(
        str(keyframe_path),
        media_type=f"image/{keyframe_path.suffix.lstrip('.').replace('jpg', 'jpeg')}",
    )


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
async def search_analytics(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
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
        type_rows = (
            db.query(SearchQuery.search_type, func.count(SearchQuery.id))
            .group_by(SearchQuery.search_type)
            .all()
        )
        type_breakdown = {t or "text": c for t, c in type_rows}

        # Average results count
        avg_results = db.query(func.avg(SearchQuery.results_count)).scalar()
        avg_results = round(float(avg_results), 1) if avg_results else 0

        # Top 10 most common queries
        top_queries_rows = (
            db.query(SearchQuery.query_text, func.count(SearchQuery.id).label("count"))
            .group_by(SearchQuery.query_text)
            .order_by(func.count(SearchQuery.id).desc())
            .limit(10)
            .all()
        )
        top_queries = [{"query": q, "count": c} for q, c in top_queries_rows]

        # Queries with zero results
        zero_results = (
            db.query(
                SearchQuery.query_text,
                SearchQuery.search_type,
                SearchQuery.search_timestamp,
            )
            .filter(SearchQuery.results_count == 0)
            .order_by(SearchQuery.search_timestamp.desc())
            .limit(20)
            .all()
        )
        zero_result_queries = [
            {"query": q, "type": t or "text", "timestamp": str(ts)}
            for q, t, ts in zero_results
        ]

        # Daily trend (last 14 days)
        daily_rows = db.execute(
            sa_text("""
            SELECT DATE(search_timestamp) as day, COUNT(*) as count
            FROM search_queries
            WHERE search_timestamp >= CURRENT_DATE - INTERVAL '14 days'
            GROUP BY DATE(search_timestamp)
            ORDER BY day DESC
        """)
        ).fetchall()
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
    admin: User = Depends(require_admin),
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
                detail="Qwen2-VL model could not be loaded. Check server logs.",
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
                print(
                    f"  Keyframe not found for scene {scene.id}: {scene.keyframe_path}"
                )
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
                    existing_emb = (
                        db.query(Embedding)
                        .filter(
                            Embedding.scene_id == scene.id,
                            Embedding.segment_id == None,  # noqa: E711
                        )
                        .first()
                    )

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
async def caption_stats(
    db: Session = Depends(get_db), admin: User = Depends(require_admin)
):
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
@app.get("/index.html")
async def read_root():
    return HTMLResponse(
        _render_frontend_index(),
        headers={"Cache-Control": "no-store, max-age=0, must-revalidate"},
    )


# Mount static files (css, js) - Make sure this is AFTER all other routes
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


if __name__ == "__main__":
    import uvicorn

    # Pass the app object directly when reload=False for better reliability
    uvicorn.run(app, host="localhost", port=8000)
