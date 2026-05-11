"""FastAPI application for video semantic search."""

import sys
import subprocess
import asyncio
import json
import mimetypes
from pathlib import Path
from functools import lru_cache

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Depends, HTTPException, Query, Request, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import os
import uuid
import threading

from database.config import get_db, test_connection
from database.models import (
    Video,
    VideoCategory,
    User,
    UserCategoryAccess,
    TranscriptSegment,
    SearchRequestLog,
    SearchImpression,
    SearchInteraction,
    SearchFeedback,
)
from search.semantic_search import SemanticSearchEngine, SearchResult
from search.multi_modal_search import MultiModalSearchEngine
from api.auth import (
    AUTH_COOKIE_NAME,
    JWT_EXPIRE_HOURS,
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

PROJECT_ROOT = Path(__file__).parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
INDEX_HTML_PATH = FRONTEND_DIR / "index.html"
FRONTEND_ASSETS = ("styles.css", "app.js", "chat.js")
VIDEOS_DIR = PROJECT_ROOT / "videos"
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
PROCESSED_DIR = PROJECT_ROOT / "processed"
_PIPELINE_JOB_LOCK = threading.Lock()
_PIPELINE_JOBS: Dict[str, Dict[str, Any]] = {}
_PIPELINE_MAX_JOBS = 200
_HIDDEN_VIDEO_CATEGORY_NAMES = {"comedy", "installation", "science", "johan sverdrup"}


def _is_hidden_video_category_name(name: Optional[str]) -> bool:
    return (name or "").strip().lower() in _HIDDEN_VIDEO_CATEGORY_NAMES


def _cors_origins() -> List[str]:
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _auth_cookie_secure() -> bool:
    return os.getenv("AUTH_COOKIE_SECURE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _normalize_db_query_mode(raw_mode: Optional[str], default: str = "postgres") -> str:
    mode = (raw_mode or default).strip().lower()
    aliases = {
        "pg": "postgres",
        "postgresql": "postgres",
        "mssql": "sqlserver",
        "sql": "sqlserver",
        "dual": "both",
        "all": "both",
    }
    normalized = aliases.get(mode, mode)
    if normalized not in {"postgres", "sqlserver", "both"}:
        return default
    return normalized


def _get_db_query_mode() -> str:
    return _normalize_db_query_mode(os.getenv("DB_QUERY_MODE", "postgres"))


def _get_request_db_query_mode(raw_mode: Optional[str]) -> str:
    if raw_mode is None:
        return _get_db_query_mode()

    normalized = _normalize_db_query_mode(raw_mode, default="")
    if normalized not in {"postgres", "sqlserver", "both"}:
        raise HTTPException(
            status_code=400,
            detail="db_source must be one of: postgres, sqlserver, both",
        )
    return normalized


def _test_sqlserver_connection() -> bool:
    try:
        from database.SQL.mssql_connection import test_connection as test_sql

        return bool(test_sql())
    except Exception as exc:
        print(f"[search] SQL Server health check unavailable: {exc}")
        return False


def _build_search_engine(db: Session, query_mode: Optional[str] = None):
    """Build search engine based on DB_QUERY_MODE env."""
    query_mode = _normalize_db_query_mode(query_mode or _get_db_query_mode())
    print(f"[search] DB_QUERY_MODE={query_mode}")

    if query_mode == "postgres":
        return SemanticSearchEngine(db)

    if query_mode == "sqlserver":
        try:
            from search.sqlserver_search import SqlServerSemanticSearchEngine

            return SqlServerSemanticSearchEngine()
        except Exception as exc:
            print(f"[search] SQL Server mode unavailable ({exc}); falling back to postgres")
            return SemanticSearchEngine(db)

    # both
    try:
        from search.dual_search import DualSemanticSearchEngine
        from search.sqlserver_search import SqlServerSemanticSearchEngine

        return DualSemanticSearchEngine(
            postgres_engine=SemanticSearchEngine(db),
            sqlserver_engine=SqlServerSemanticSearchEngine(),
            mode="both",
        )
    except Exception as exc:
        print(f"[search] Dual mode unavailable ({exc}); falling back to postgres")
        return SemanticSearchEngine(db)


def _close_engine(engine: Any) -> None:
    if engine is None:
        return
    try:
        close = getattr(engine, "close", None)
        if callable(close):
            close()
    except Exception:
        pass

    for child_name in ("postgres_engine", "sqlserver_engine"):
        child = getattr(engine, child_name, None)
        if child is not None:
            _close_engine(child)


def _get_search_engine_for_mode(db: Session, raw_mode: Optional[str] = None):
    requested_mode = _get_request_db_query_mode(raw_mode)
    return _build_search_engine(db, query_mode=requested_mode), requested_mode


def _get_multimodal_text_engine(db: Session) -> SemanticSearchEngine:
    """
    Multi-modal search requires postgres-native text/vision joins.
    Keep this path on SemanticSearchEngine even when DB_QUERY_MODE=both/sqlserver.
    """
    return SemanticSearchEngine(db)


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


def _set_pipeline_job(job_id: str, **fields) -> None:
    with _PIPELINE_JOB_LOCK:
        job = _PIPELINE_JOBS.get(job_id)
        if not job:
            return
        if "progress" in fields:
            fields["progress"] = max(0, min(100, int(fields["progress"])))
        job.update(fields)
        job["updated_at"] = datetime.utcnow().isoformat()


def _get_pipeline_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _PIPELINE_JOB_LOCK:
        job = _PIPELINE_JOBS.get(job_id)
        return dict(job) if job else None


def _prune_pipeline_jobs() -> None:
    with _PIPELINE_JOB_LOCK:
        if len(_PIPELINE_JOBS) <= _PIPELINE_MAX_JOBS:
            return
        done_statuses = {"completed", "failed"}
        done_jobs = [
            (jid, data.get("updated_at", ""))
            for jid, data in _PIPELINE_JOBS.items()
            if data.get("status") in done_statuses
        ]
        done_jobs.sort(key=lambda x: x[1])
        to_remove = len(_PIPELINE_JOBS) - _PIPELINE_MAX_JOBS
        for jid, _ in done_jobs[:to_remove]:
            _PIPELINE_JOBS.pop(jid, None)


def _is_within_directory(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _resolve_local_file_path(
    raw_path: Optional[str],
    allowed_roots: Tuple[Path, ...],
    basename_root: Optional[Path] = None,
) -> Optional[Path]:
    if not raw_path:
        return None

    candidates: List[Path] = []
    candidate = Path(raw_path)
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append(PROJECT_ROOT / candidate)
    if basename_root is not None:
        candidates.append(basename_root / os.path.basename(raw_path))

    for item in candidates:
        try:
            resolved = item.resolve()
        except (OSError, RuntimeError):
            continue
        if not resolved.exists() or not resolved.is_file():
            continue
        if any(_is_within_directory(resolved, root) for root in allowed_roots):
            return resolved
    return None


def _resolve_video_file_path(raw_path: Optional[str]) -> Optional[Path]:
    return _resolve_local_file_path(
        raw_path,
        allowed_roots=(VIDEOS_DIR,),
        basename_root=VIDEOS_DIR,
    )


def _resolve_document_file_path(raw_path: Optional[str]) -> Optional[Path]:
    return _resolve_local_file_path(
        raw_path,
        allowed_roots=(DOCUMENTS_DIR,),
        basename_root=DOCUMENTS_DIR,
    )


def _resolve_keyframe_file_path(raw_path: Optional[str]) -> Optional[Path]:
    return _resolve_local_file_path(
        raw_path,
        allowed_roots=(PROCESSED_DIR,),
    )


def _safe_upload_filename(filename: str, fallback_prefix: str, suffix: str = "") -> str:
    raw_name = os.path.basename(filename or "")
    requested_suffix = suffix or Path(raw_name).suffix.lower()
    safe_stem = re.sub(r"[^\w\s\-\(\)]", "", Path(raw_name).stem).strip(" ._-")
    if not safe_stem:
        safe_stem = f"{fallback_prefix}_{int(time.time())}"
    return f"{safe_stem[:120]}{requested_suffix.lower()}"


def _media_type_for_image(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".").replace("jpg", "jpeg")
    return f"image/{suffix}"


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

_TRANSLATE_PATTERNS = [
    re.compile(r"^\s*translate(?:\s+this)?(?:\s+to\s+.+)?\s*[\?!.]?\s*$", re.I),
    re.compile(
        r"^\s*(?:can you|could you|please)\s+translate(?:\s+this)?(?:\s+to\s+.+)?\s*[\?!.]?\s*$",
        re.I,
    ),
    re.compile(
        r"^\s*(?:kan du|kan dere|vennligst)\s+oversett(?:\s+dette)?(?:\s+til\s+.+)?\s*[\?!.]?\s*$",
        re.I,
    ),
]

_LANGUAGE_TO_CODE = {
    "english": "en",
    "en": "en",
    "norwegian": "no",
    "norsk": "no",
    "no": "no",
    "spanish": "es",
    "es": "es",
    "french": "fr",
    "fr": "fr",
    "german": "de",
    "de": "de",
    "arabic": "ar",
    "ar": "ar",
}


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


def _is_translate_request(text: str) -> bool:
    q = (text or "").strip()
    if not q:
        return False
    return any(pattern.match(q) for pattern in _TRANSLATE_PATTERNS)


def _extract_target_language_code(question: str, language_hint: Optional[str]) -> Optional[str]:
    hint = (language_hint or "").strip().lower()
    if hint and hint != "auto":
        return _LANGUAGE_TO_CODE.get(hint)

    q = (question or "").strip().lower()
    if not q:
        return None

    # Keep this simple and deterministic for common phrasing.
    for phrase, code in (
        ("to english", "en"),
        ("in english", "en"),
        ("til engelsk", "en"),
        ("to norwegian", "no"),
        ("in norwegian", "no"),
        ("til norsk", "no"),
        ("to spanish", "es"),
        ("in spanish", "es"),
        ("to french", "fr"),
        ("in french", "fr"),
        ("to german", "de"),
        ("in german", "de"),
        ("to arabic", "ar"),
        ("in arabic", "ar"),
    ):
        if phrase in q:
            return code

    return None


def _translate_text_via_mymemory(text: str, target: str, source: str = "auto") -> str:
    if not text:
        return ""
    lang_pair = f"{source}|{target}"
    api_url = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(text[:500])}&langpair={lang_pair}"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "ATLAS/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data.get("responseData", {}).get("translatedText", text)
    except Exception:
        return text


def _get_allowed_filenames(user: User, db: Session) -> Optional[set]:
    """Return the set of video filenames the user may see, or None for admins (=all)."""
    allowed_cats = get_user_allowed_categories(user)
    if allowed_cats is None:  # admin
        return None
    all_videos = db.query(Video).all()
    return {
        v.filename for v in all_videos if get_video_category(v.filename) in allowed_cats
    }


def _get_document_acl_category(document_obj: Any) -> str:
    """Return effective ACL category name for a document."""
    category_rel = getattr(document_obj, "category_rel", None)
    if category_rel and getattr(category_rel, "name", None):
        return category_rel.name
    return "Other"


def _document_acl_clause(user: User):
    """Build a SQL clause for document category ACL (None means no ACL filter)."""
    allowed_cats = get_user_allowed_categories(user)
    if allowed_cats is None:  # admin
        return None

    allowed = {c.strip() for c in allowed_cats if c and c.strip()}
    if not allowed:
        from sqlalchemy import false as sa_false

        return sa_false()

    from sqlalchemy import or_ as sa_or

    predicates = []
    named_categories = sorted(cat for cat in allowed if cat != "Other")
    if named_categories:
        predicates.append(VideoCategory.name.in_(named_categories))

    if "Other" in allowed:
        predicates.append(VideoCategory.name == "Other")
        predicates.append(VideoCategory.name.is_(None))

    if not predicates:
        from sqlalchemy import false as sa_false

        return sa_false()

    return sa_or(*predicates)


def _get_allowed_document_ids(user: User, db: Session) -> Optional[set]:
    """Return document IDs the user may access, or None for admins (=all)."""
    try:
        from database.document_models import Document as DocumentModel
    except Exception:
        return set()

    acl_clause = _document_acl_clause(user)
    if acl_clause is None:  # admin
        return None

    rows = (
        db.query(DocumentModel.id)
        .outerjoin(VideoCategory, DocumentModel.category_id == VideoCategory.id)
        .filter(acl_clause)
        .all()
    )
    return {row.id for row in rows}


def _get_document_ids_for_filters(
    user: User,
    db: Session,
    categories: Optional[List[str]] = None,
    sites: Optional[List[str]] = None,
    label: Optional[str] = None,
) -> Optional[set]:
    """Return document IDs visible to the user and matching optional UI filters."""
    allowed_ids = _get_allowed_document_ids(user, db)
    cats = [c for c in (categories or []) if c]
    site_labels = [s for s in (sites or []) if s]
    label_filter = (label or "").strip()

    if not cats and not site_labels and not label_filter:
        return allowed_ids

    try:
        from database.document_models import Document as DocumentModel
    except Exception:
        return set()

    query = (
        db.query(DocumentModel.id)
        .outerjoin(VideoCategory, DocumentModel.category_id == VideoCategory.id)
    )
    if cats:
        query = query.filter(VideoCategory.name.in_(cats))
    if label_filter:
        query = query.filter(DocumentModel.label.ilike(f"%{label_filter}%"))
    if site_labels:
        query = query.filter(DocumentModel.label.in_(site_labels))

    filtered_ids = {row.id for row in query.all()}
    if allowed_ids is None:
        return filtered_ids
    return allowed_ids & filtered_ids


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


def _upsert_video_row_from_file(
    db: Session,
    file_path: Path,
    category_name: Optional[str] = None,
    label: Optional[str] = None,
) -> Video:
    """Ensure a video file on disk has a corresponding videos-table row."""
    filename = file_path.name
    stat = file_path.stat()

    video = db.query(Video).filter(Video.filename == filename).first()
    if not video:
        video = Video(filename=filename, file_path=str(file_path))
        db.add(video)

    video.file_path = str(file_path)
    video.file_size_mb = round(stat.st_size / (1024 * 1024), 2)

    if label is not None:
        video.label = label.strip() or None

    if category_name:
        category_name = category_name.strip() or "Other"
        if _is_hidden_video_category_name(category_name):
            category_name = "Other"
        category_obj = (
            db.query(VideoCategory).filter(VideoCategory.name == category_name).first()
        )
        if not category_obj:
            category_obj = VideoCategory(name=category_name)
            db.add(category_obj)
            db.flush()
        video.category_id = category_obj.id

    return video


def _sync_videos_table_with_disk(db: Session) -> int:
    """
    Auto-register files in PROJECT_ROOT/videos that are missing from DB.
    Returns number of new rows added.
    """
    videos_dir = PROJECT_ROOT / "videos"
    if not videos_dir.exists():
        return 0

    created = 0
    for f in videos_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in _ALLOWED_VIDEO_EXT:
            continue
        existing = db.query(Video.id).filter(Video.filename == f.name).first()
        if existing:
            continue
        _upsert_video_row_from_file(db, f)
        created += 1

    if created:
        db.commit()
    return created


def _sync_documents_table_with_disk(db: Session) -> int:
    """
    Auto-register files in PROJECT_ROOT/documents that are missing from DB.
    Returns number of new rows added.
    """
    documents_dir = PROJECT_ROOT / "documents"
    if not documents_dir.exists():
        return 0

    try:
        from database.document_models import Document as DocumentModel
    except Exception:
        return 0

    created = 0
    for f in documents_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in _ALLOWED_DOCUMENT_EXT:
            continue

        normalized_path = str(f.resolve())
        existing = (
            db.query(DocumentModel.id)
            .filter(DocumentModel.file_path == normalized_path)
            .first()
        )
        if existing:
            continue

        file_size_mb = round(f.stat().st_size / (1024 * 1024), 2)
        db.add(
            DocumentModel(
                filename=f.name,
                file_path=normalized_path,
                file_type=f.suffix.lower().lstrip("."),
                file_size_mb=file_size_mb,
                extraction_method="synced",
            )
        )
        created += 1

    if created:
        db.commit()
    return created


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


def _extract_document_id_from_result(result: Any) -> Optional[int]:
    doc_id = getattr(result, "document_id", None)
    if doc_id is None and getattr(result, "source_type", "video") == "document":
        doc_id = getattr(result, "video_id", None)
    try:
        return int(doc_id) if doc_id is not None else None
    except (TypeError, ValueError):
        return None


def _extract_document_id_from_result_dict(result: Dict[str, Any]) -> Optional[int]:
    doc_id = result.get("document_id")
    if doc_id is None and result.get("source_type") == "document":
        doc_id = result.get("video_id")
    try:
        return int(doc_id) if doc_id is not None else None
    except (TypeError, ValueError):
        return None


def _filter_results(
    results,
    allowed_filenames,
    limit=None,
    allowed_document_ids: Optional[set] = None,
):
    """Filter search results to only include videos the user may access."""
    filtered = []
    for result in results:
        # Document results are filtered by document ACL.
        if getattr(result, "source_type", "video") == "document":
            doc_id = _extract_document_id_from_result(result)
            if (
                allowed_document_ids is not None
                and (doc_id is None or doc_id not in allowed_document_ids)
            ):
                continue
            filtered.append(result)
            if limit and len(filtered) >= limit:
                break
            continue
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
    return filtered[:limit] if limit else filtered


def _filter_result_dicts(
    result_dicts,
    allowed_filenames,
    limit=None,
    allowed_document_ids: Optional[set] = None,
):
    """Filter dict-form search results to only include videos the user may access."""
    filtered = []
    for result in result_dicts:
        # Document results are filtered by document ACL.
        if result.get("source_type") == "document":
            doc_id = _extract_document_id_from_result_dict(result)
            if (
                allowed_document_ids is not None
                and (doc_id is None or doc_id not in allowed_document_ids)
            ):
                continue
            filtered.append(result)
            if limit and len(filtered) >= limit:
                break
            continue
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


def _to_result_dict(item: Any) -> Dict[str, Any]:
    """Convert result object/dict to a serializable dict without mutating the source."""
    if item is None:
        return {}
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "to_dict"):
        try:
            return dict(item.to_dict())
        except Exception:
            pass
    data = getattr(item, "__dict__", None)
    if isinstance(data, dict):
        return dict(data)
    return {}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _log_search_request_with_impressions(
    db: Session,
    user: Optional[User],
    query_text: str,
    search_mode: str,
    results: List[Any],
    latency_seconds: float,
    facet: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Persist Phase-1 telemetry:
      - one search request row
      - one impression row per returned result
    Returns (request_uuid, serialized_results_with_impression_ids).
    """
    serialized_results = [_to_result_dict(r) for r in (results or [])]

    try:
        request_row = SearchRequestLog(
            request_uuid=uuid.uuid4().hex,
            user_id=getattr(user, "id", None),
            query_text=query_text or "",
            search_mode=search_mode,
            facet=facet,
            filters=filters or {},
            results_count=len(serialized_results),
            latency_ms=round(max(0.0, latency_seconds) * 1000.0, 2),
        )
        db.add(request_row)
        db.flush()

        impression_rows: List[SearchImpression] = []
        for rank, item in enumerate(serialized_results, start=1):
            score_val = item.get("combined_score", item.get("score"))
            imp = SearchImpression(
                request_id=request_row.id,
                impression_rank=rank,
                source_type=item.get("source_type") or "video",
                result_segment_id=item.get("segment_id"),
                result_video_id=item.get("video_id"),
                result_video_filename=item.get("video_filename"),
                result_start_time=_safe_float(item.get("start_time")),
                result_end_time=_safe_float(item.get("end_time")),
                result_score=_safe_float(score_val),
                result_payload=item,
            )
            db.add(imp)
            impression_rows.append(imp)

        db.flush()

        for item, imp in zip(serialized_results, impression_rows):
            item["request_id"] = request_row.request_uuid
            item["impression_id"] = imp.id

        db.commit()
        return request_row.request_uuid, serialized_results

    except Exception as e:
        # Never fail search for telemetry issues.
        print(f"[feedback] Warning: failed to log request telemetry: {e}")
        db.rollback()
        return None, serialized_results


def _resolve_search_request_for_user(
    db: Session, user: User, request_uuid: str
) -> Optional[SearchRequestLog]:
    request_row = (
        db.query(SearchRequestLog)
        .filter(SearchRequestLog.request_uuid == request_uuid)
        .first()
    )
    if request_row is None:
        return None

    # Admin can inspect/label anything; viewers can only write to their own request rows.
    if user.role != "admin" and request_row.user_id and request_row.user_id != user.id:
        return None

    return request_row


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
    db_source: Optional[str] = Field(
        None,
        description="Optional database source override: postgres, sqlserver, or both. Defaults to DB_QUERY_MODE.",
    )
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
    request_id: Optional[str] = Field(
        None,
        description="Telemetry id used by the frontend to log click/dwell/feedback signals.",
    )
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


class FeedbackEventRequest(BaseModel):
    """Implicit interaction event payload."""

    request_id: str = Field(..., min_length=8, max_length=64)
    impression_id: Optional[int] = None
    interaction_type: str = Field(..., min_length=2, max_length=40)
    dwell_ms: Optional[int] = Field(None, ge=0)
    metadata: Optional[Dict[str, Any]] = None


class FeedbackRatingRequest(BaseModel):
    """Explicit relevance label payload."""

    request_id: str = Field(..., min_length=8, max_length=64)
    impression_id: Optional[int] = None
    feedback_value: int = Field(..., ge=-1, le=1)
    comment: Optional[str] = Field(None, max_length=2000)
    metadata: Optional[Dict[str, Any]] = None


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

    # Seed an initial admin only when an explicit bootstrap password is provided.
    from database.config import SessionLocal
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            admin_username = os.getenv("ATLAS_BOOTSTRAP_ADMIN_USER", "admin").strip()
            admin_password = os.getenv("ATLAS_BOOTSTRAP_ADMIN_PASSWORD")
            if not admin_password:
                print(
                    "[auth] No users exist. Set ATLAS_BOOTSTRAP_ADMIN_PASSWORD "
                    "to create the initial admin account."
                )
            else:
                admin = User(
                    username=admin_username,
                    password_hash=hash_password(admin_password),
                    role="admin",
                )
                db.add(admin)
                db.commit()
                print(f"[auth] Created bootstrap admin user ({admin_username})")
        else:
            print("[auth] Users table OK")

        # Seed default video categories
        _DEFAULT_CATEGORIES = ["Oil & Gas", "Maintenance", "Operations"]
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
        warmup_search_engine = _build_search_engine(warmup_db)
        if has_cuda:
            MultiModalSearchEngine(
                db=warmup_db,
                text_search=_get_multimodal_text_engine(warmup_db),
            )
        _close_engine(warmup_search_engine)
            
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




def get_video_qa(db: Session = Depends(get_db)):
    """Create a request-bound VideoQA wrapper around shared model resources."""
    from llm.video_qa import VideoQA

    return VideoQA(db)


def get_search_engine(db: Session = Depends(get_db)):
    """Create a request-bound configured search engine."""
    return _build_search_engine(db)


def get_mm_search_engine(db: Session = Depends(get_db)):
    """Create a request-bound MultiModalSearchEngine."""
    return MultiModalSearchEngine(
        db=db,
        text_search=_get_multimodal_text_engine(db),
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    db_ok = test_connection()
    payload = {
        "status": "healthy" if db_ok else "unhealthy",
        "database": "ok" if db_ok else "error",
        "db_query_mode": _get_db_query_mode(),
        **_server_capabilities(),
    }
    query_mode = _get_db_query_mode()
    if query_mode in {"sqlserver", "both"}:
        sql_ok = _test_sqlserver_connection()
        payload["sqlserver_database"] = "ok" if sql_ok else "error"
        if db_ok and not sql_ok and query_mode == "both":
            payload["status"] = "degraded"
    return payload


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
async def login(
    req: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    """Authenticate and return a JWT token."""
    user = db.query(User).filter(User.username == req.username).first()
    if not user or not verify_password(req.password, str(user.password_hash)):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token(user.id, str(user.username), str(user.role))
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        max_age=JWT_EXPIRE_HOURS * 3600,
        httponly=True,
        secure=_auth_cookie_secure(),
        samesite="lax",
        path="/",
    )
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


@app.post("/auth/logout")
async def logout(response: Response):
    """Clear the browser auth cookie."""
    response.delete_cookie(
        key=AUTH_COOKIE_NAME,
        httponly=True,
        secure=_auth_cookie_secure(),
        samesite="lax",
        path="/",
    )
    return {"status": "ok"}


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
    cats = {c.name for c in db_cats if not _is_hidden_video_category_name(c.name)}
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
    cats = {c.name for c in db_cats if not _is_hidden_video_category_name(c.name)}
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
    return {
        "translated": _translate_text_via_mymemory(
            text=text,
            target=target,
            source=source,
        )
    }


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

# Allowed document extensions for sync/upload
_ALLOWED_DOCUMENT_EXT = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
}

# Available pipeline models (used for frontend dropdowns)
TRANSCRIPTION_MODELS = [
    {
        "id": "whisper-large-v3",
        "label": "Whisper Large v3",
        "backend": "whisper",
        "variant": {"name": "large-v3"},
    }
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
    {"id": "siglip2-base", "label": "google/siglip2-base-patch16-224"},
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
    if _is_hidden_video_category_name(name):
        raise HTTPException(
            status_code=400,
            detail=f"'{name}' is no longer available as a category. Use the installation label field instead.",
        )
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
    cats = [c for c in cats if not _is_hidden_video_category_name(c.name)]
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
    """Set site label and/or category on a video (admin only)."""
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


class UpdateDocumentRequest(BaseModel):
    label: Optional[str] = None
    category_id: Optional[int] = None


@app.put("/documents/{doc_id}")
async def update_document_metadata(
    doc_id: int,
    req: UpdateDocumentRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Set site label and/or category on a document (admin only)."""
    try:
        from database.document_models import Document as DocumentModel
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document models unavailable: {e}")

    doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if req.label is not None:
        doc.label = req.label.strip() or None

    if req.category_id is not None:
        if req.category_id == 0:
            doc.category_id = None
        else:
            cat = db.query(VideoCategory).filter(VideoCategory.id == req.category_id).first()
            if not cat:
                raise HTTPException(status_code=404, detail="Category not found")
            doc.category_id = cat.id

    db.commit()
    db.refresh(doc)
    return {
        "id": doc.id,
        "filename": doc.filename,
        "label": doc.label,
        "category": doc.category_rel.name if doc.category_rel else None,
        "category_id": doc.category_id,
    }


@app.post("/admin/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    category: str = Query("Other", description="Category for the video"),
    label: Optional[str] = Query(None, description="Installation label for the video"),
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
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

    category_name = (category or "Other").strip() or "Other"
    if _is_hidden_video_category_name(category_name):
        category_name = "Other"
    installation_label = label.strip() if label is not None else None
    if dest.exists():
        # If file already exists on disk, treat as an idempotent register operation.
        video_row = _upsert_video_row_from_file(
            db,
            dest,
            category_name=category_name,
            label=installation_label,
        )
        db.commit()
        db.refresh(video_row)
        return {
            "filename": safe_name,
            "size_mb": video_row.file_size_mb,
            "category": category_name,
            "label": video_row.label,
            "path": str(dest),
            "video_id": video_row.id,
            "already_existed": True,
            "detail": f"File '{safe_name}' already existed and is now registered.",
        }

    # Stream file to disk (avoid loading entire file into memory)
    total = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            f.write(chunk)
            total += len(chunk)

    video_row = _upsert_video_row_from_file(
        db,
        dest,
        category_name=category_name,
        label=installation_label,
    )
    db.commit()
    db.refresh(video_row)

    return {
        "filename": safe_name,
        "size_mb": video_row.file_size_mb,
        "category": category_name,
        "label": video_row.label,
        "path": str(dest),
        "video_id": video_row.id,
        "already_existed": False,
    }


@app.post("/admin/sync-videos")
async def sync_videos(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Force-sync files in videos/ into the videos table."""
    created = _sync_videos_table_with_disk(db)
    total = db.query(Video).count()
    return {
        "status": "ok",
        "synced_new_rows": created,
        "total_rows": total,
    }


@app.post("/admin/sync-documents")
async def sync_documents(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Force-sync files in documents/ into the documents table."""
    created = _sync_documents_table_with_disk(db)
    try:
        from database.document_models import Document as DocumentModel

        total = db.query(DocumentModel).count()
    except Exception:
        total = 0
    return {
        "status": "ok",
        "synced_new_rows": created,
        "total_rows": total,
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
        "transcription_model": "whisper-large-v3",
        "scene_detection": "pyscenedetect",
        "scene_threshold": 30.0,
        "device": "auto"   # ignored; runtime always resolves automatically
    }
    Returns immediately with a job id; poll /admin/run-pipeline/{job_id}.
    """
    filename = (req.get("filename") or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="filename is required")

    project_root = Path(__file__).parent.parent
    video_path = project_root / "videos" / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {filename}")

    # Resolve transcription model
    model_id = req.get("transcription_model", "whisper-large-v3")
    model_entry = next((m for m in TRANSCRIPTION_MODELS if m["id"] == model_id), None)
    if not model_entry:
        raise HTTPException(
            status_code=400, detail=f"Unknown transcription model: {model_id}"
        )

    scene_threshold = float(req.get("scene_threshold", 30.0))
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with _PIPELINE_JOB_LOCK:
        _PIPELINE_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "stage": "queued",
            "message": "Queued",
            "filename": filename,
            "model": model_entry["label"],
            "device": "auto",
            "scene_threshold": scene_threshold,
            "result_summary": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
    _prune_pipeline_jobs()

    async def _runner() -> None:
        def _on_progress(percent: int, stage: str, message: str) -> None:
            _set_pipeline_job(
                job_id,
                status="running",
                progress=percent,
                stage=stage,
                message=message,
            )

        def _run_sync():
            from basic_pipeline import BasicVideoPipeline

            pipe = BasicVideoPipeline(
                scene_threshold=scene_threshold,
                device="auto",
            )
            return pipe.process_video(str(video_path), progress_callback=_on_progress)

        _set_pipeline_job(
            job_id,
            status="running",
            progress=1,
            stage="init",
            message="Starting pipeline",
        )
        try:
            result = await asyncio.to_thread(_run_sync)
            _set_pipeline_job(
                job_id,
                status="completed",
                progress=100,
                stage="done",
                message="Pipeline completed",
                result_summary={
                    "segments": (
                        (result.get("transcription") or {}).get("num_segments", 0)
                        if isinstance(result, dict)
                        else 0
                    ),
                    "scenes": (
                        (result.get("scene_analysis") or {}).get("num_scenes", 0)
                        if isinstance(result, dict)
                        else 0
                    ),
                },
            )
        except Exception as e:
            _set_pipeline_job(
                job_id,
                status="failed",
                progress=100,
                stage="error",
                message=f"Pipeline failed: {e}",
                error=str(e),
            )

    asyncio.create_task(_runner())

    return {
        "status": "started",
        "job_id": job_id,
        "filename": filename,
        "model": model_entry["label"],
        "device": "auto",
    }


@app.get("/admin/run-pipeline/{job_id}")
async def get_pipeline_job_status(
    job_id: str,
    admin: User = Depends(require_admin),
):
    """Return progress/status for a pipeline job started via /admin/run-pipeline."""
    job = _get_pipeline_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Pipeline job not found")
    return job


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
    _sync_videos_table_with_disk(db)
    return [
        _serialize_video_info(video)
        for video, _ in _get_accessible_available_videos(user, db)
    ]


@app.get("/videos/count")
async def count_videos(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    """Return the accessible count of currently available videos."""
    _sync_videos_table_with_disk(db)
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
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Semantic search endpoint.

    Search video transcripts using hybrid semantic + fuzzy text matching.
    """
    start_time = time.time()
    search_engine = None

    try:
        search_engine, active_db_mode = _get_search_engine_for_mode(db, request.db_source)
        allowed_filenames = _get_allowed_filenames(user, db)
        allowed_document_ids = _get_allowed_document_ids(user, db)
        results = search_engine.search(
            query=request.query,
            top_k=request.top_k * 3 if allowed_filenames is not None else request.top_k,
            semantic_weight=request.semantic_weight,
            text_weight=request.text_weight,
            min_score=request.min_score,
            video_filter=request.video_filter,
            log_query=True,
        )
        results = _filter_results(
            results,
            allowed_filenames,
            limit=request.top_k,
            allowed_document_ids=allowed_document_ids,
        )

        search_time = time.time() - start_time
        request_id, result_dicts = _log_search_request_with_impressions(
            db=db,
            user=user,
            query_text=request.query,
            search_mode="search_api",
            results=results,
            latency_seconds=search_time,
            facet=None,
            filters={
                "video_filter": request.video_filter,
                "db_source": active_db_mode,
                "semantic_weight": request.semantic_weight,
                "text_weight": request.text_weight,
                "min_score": request.min_score,
                "top_k": request.top_k,
            },
        )

        return SearchResponse(
            query=request.query,
            request_id=request_id,
            results_count=len(result_dicts),
            results=result_dicts,
            search_time_seconds=round(search_time, 3),
            search_metadata={"db_query_mode": active_db_mode},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    finally:
        _close_engine(search_engine)


@app.get("/search/quick")
async def quick_search(
    q: str = Query(..., description="Search query", min_length=1),
    limit: int = Query(10, description="Number of results", ge=1, le=50),
    video: Optional[str] = Query(None, description="Filter by video filename"),
    db_source: Optional[str] = Query(
        None,
        description="Optional database source override: postgres, sqlserver, or both",
    ),
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
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Quick search endpoint (GET request for easy testing).
    Supports filtering by category, label, and site.
    ```
    GET /search/quick?q=Omega+Alpha+well&limit=5&category=Maintenance&site=Yggdrasil
    ```
    """
    start_time = time.time()
    search_engine = None

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
    allowed_document_ids = _get_document_ids_for_filters(
        user,
        db,
        categories=cats,
        sites=sites,
        label=label,
    )

    try:
        search_engine, active_db_mode = _get_search_engine_for_mode(db, db_source)
        fallback_data = search_engine.search_with_fallback(
            query=q,
            top_k=limit * 3 if allowed_filenames else limit,
            video_filter=video,
            facet=facet or "auto",
        )

        results = _filter_results(
            fallback_data["results"],
            allowed_filenames,
            limit=limit,
            allowed_document_ids=allowed_document_ids,
        )
        metadata = fallback_data["search_metadata"]
        search_time = time.time() - start_time

        request_id, result_dicts = _log_search_request_with_impressions(
            db=db,
            user=user,
            query_text=q,
            search_mode="search_quick",
            results=results,
            latency_seconds=search_time,
            facet=metadata.get("facet_applied") or (facet or "auto"),
            filters={
                "limit": limit,
                "video_filter": video,
                "db_source": active_db_mode,
                "category": cats,
                "site": sites,
                "label": label,
                "tiers_tried": metadata.get("tiers_tried") or [],
            },
        )

        # Build grouped-by-video structure
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
            "request_id": request_id,
            "results_count": len(result_dicts),
            "results": result_dicts,
            "grouped_results": grouped_results,
            "db_query_mode": active_db_mode,
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
    finally:
        _close_engine(search_engine)


@app.get("/search/browse")
async def browse_by_category(
    category: Optional[List[str]] = Query(
        None, description="Category name(s) to browse"
    ),
    site: Optional[List[str]] = Query(
        None, description="Installation/site label name(s) to browse"
    ),
    limit: int = Query(
        10, description="Max videos and documents to return per source", ge=1, le=50
    ),
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
            status_code=400, detail="At least one category or installation is required"
        )

    # Build video query based on category and/or installation filters.
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

    matched_documents = []
    DocumentModel = None
    DocumentChunkModel = None
    try:
        from database.document_models import (
            Document as DocumentModel,
            DocumentChunk as DocumentChunkModel,
        )
    except Exception:
        DocumentModel = None
        DocumentChunkModel = None

    if DocumentModel is not None and DocumentChunkModel is not None:
        q_docs = db.query(DocumentModel)
        if cats:
            q_docs = q_docs.join(
                VideoCategory, DocumentModel.category_id == VideoCategory.id
            ).filter(VideoCategory.name.in_(cats))
        if sites:
            q_docs = q_docs.filter(DocumentModel.label.in_(sites))

        acl_clause = _document_acl_clause(user)
        if acl_clause is not None:
            if not cats:
                q_docs = q_docs.outerjoin(
                    VideoCategory, DocumentModel.category_id == VideoCategory.id
                )
            q_docs = q_docs.filter(acl_clause)

        matched_documents = q_docs.order_by(DocumentModel.id.desc()).limit(limit).all()

    # Build grouped results with a few transcript snippets/passages per source item.
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
                "source_type": "video",
                "video_id": v.id,
                "video_filename": v.filename,
                "video_path": v.file_path,
                "segment_index": seg.segment_index,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "text": seg.text,
                "score": 1.0,
            }
            occurrences.append(rd)
            result_dicts.append(rd)
        if occurrences:
            grouped_results.append(
                {
                    "source_type": "video",
                    "video_id": v.id,
                    "video_filename": v.filename,
                    "occurrences": occurrences,
                }
            )

    for doc in matched_documents:
        chunks = (
            db.query(DocumentChunkModel)
            .filter(DocumentChunkModel.document_id == doc.id)
            .order_by(DocumentChunkModel.chunk_index)
            .limit(5)
            .all()
        )
        if not chunks:
            chunks = [None]

        occurrences = []
        for idx, chunk in enumerate(chunks):
            page_number = getattr(chunk, "page_number", None)
            chunk_index = (
                getattr(chunk, "chunk_index", None)
                if chunk is not None
                else idx
            )
            section_heading = (
                getattr(chunk, "section_heading", None)
                if chunk is not None
                else None
            )
            text = getattr(chunk, "text", None) or f"Document: {doc.filename}"
            if section_heading:
                text = f"[{section_heading}] {text}"

            if page_number:
                location = f"Page {page_number}"
            elif chunk_index is not None:
                location = f"Chunk {int(chunk_index) + 1}"
            else:
                location = "Document"

            rd = {
                "source_type": "document",
                "segment_id": getattr(chunk, "id", None),
                "video_id": doc.id,
                "video_filename": doc.filename,
                "video_path": doc.file_path,
                "document_id": doc.id,
                "document_filename": doc.filename,
                "document_path": doc.file_path,
                "document_page": page_number,
                "document_chunk_index": chunk_index,
                "document_section_heading": section_heading,
                "document_file_type": doc.file_type,
                "document_location": location,
                "timestamp": location,
                "start_time": 0.0,
                "end_time": 0.0,
                "text": text,
                "score": 1.0,
                "match_type": "browse",
                "result_id": getattr(chunk, "id", None),
            }
            occurrences.append(rd)
            result_dicts.append(rd)

        grouped_results.append(
            {
                "source_type": "document",
                "video_id": doc.id,
                "video_filename": doc.filename,
                "display_name": doc.filename,
                "occurrences": occurrences,
            }
        )

    search_time = time.time() - start_time
    request_id, result_dicts = _log_search_request_with_impressions(
        db=db,
        user=user,
        query_text=f"[browse] categories={cats or []}; sites={sites or []}",
        search_mode="search_browse",
        results=result_dicts,
        latency_seconds=search_time,
        facet=None,
        filters={"category": cats, "site": sites, "limit": limit},
    )

    # Rebuild grouped payload from telemetry-enriched result rows (has impression ids).
    grouped_results = []
    grouped_map: Dict[str, Dict[str, Any]] = {}
    for row in result_dicts:
        source_type = row.get("source_type") or "video"
        entity_id = (
            row.get("document_id")
            if source_type == "document"
            else row.get("video_id")
        )
        group_key = f"{source_type}:{entity_id}"
        if group_key not in grouped_map:
            grouped_map[group_key] = {
                "source_type": source_type,
                "video_id": row["video_id"],
                "video_filename": row["video_filename"],
                "display_name": row.get("document_filename") or row["video_filename"],
                "occurrences": [],
            }
        grouped_map[group_key]["occurrences"].append(row)
    grouped_results = list(grouped_map.values())

    title_parts = []
    if cats:
        title_parts.append(f"Category: {', '.join(cats)}")
    if sites:
        title_parts.append(f"Installation: {', '.join(sites)}")
    display_query = " / ".join(title_parts)
    return {
        "query": display_query,
        "display_title": f"Results for {display_query}",
        "browse_filters": {"category": cats, "site": sites},
        "request_id": request_id,
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
    db_source: Optional[str] = Query(
        None,
        description="Optional database source override: postgres, sqlserver, or both",
    ),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Exact phrase search (case-insensitive).
    """
    start_time = time.time()
    search_engine = None

    try:
        search_engine, active_db_mode = _get_search_engine_for_mode(db, db_source)
        results = search_engine.search_exact_phrase(phrase=phrase, video_filter=video)
        allowed_filenames = _get_allowed_filenames(user, db)
        allowed_document_ids = _get_allowed_document_ids(user, db)
        results = _filter_results(
            results,
            allowed_filenames,
            allowed_document_ids=allowed_document_ids,
        )

        search_time = time.time() - start_time
        request_id, result_dicts = _log_search_request_with_impressions(
            db=db,
            user=user,
            query_text=phrase,
            search_mode="search_exact",
            results=results,
            latency_seconds=search_time,
            filters={"video_filter": video, "db_source": active_db_mode},
        )

        return {
            "phrase": phrase,
            "request_id": request_id,
            "results_count": len(result_dicts),
            "results": result_dicts,
            "db_query_mode": active_db_mode,
            "search_time_seconds": round(search_time, 3),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    finally:
        _close_engine(search_engine)


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

        mm_search = MultiModalSearchEngine(
            db=db,
            text_weight=text_weight,
            vision_weight=vision_weight,
            text_search=_get_multimodal_text_engine(db),
        )

        allowed_filenames = _get_allowed_filenames(user, db)
        allowed_document_ids = _get_allowed_document_ids(user, db)

        # Perform search with fallback (includes LLM intent parsing)
        fallback_data = mm_search.search_with_fallback(
            query=request.query,
            top_k=request.top_k * 3 if allowed_filenames is not None else request.top_k,
            video_filter=request.video_filter,
            use_llm=request.use_llm,
        )

        results = _filter_results(
            fallback_data["results"],
            allowed_filenames,
            limit=request.top_k,
            allowed_document_ids=allowed_document_ids,
        )
        metadata = fallback_data["search_metadata"]

        search_time = time.time() - start_time
        request_id, result_dicts = _log_search_request_with_impressions(
            db=db,
            user=user,
            query_text=request.query,
            search_mode="search_multimodal",
            results=results,
            latency_seconds=search_time,
            facet=metadata.get("facet_applied"),
            filters={
                "top_k": request.top_k,
                "video_filter": request.video_filter,
                "text_weight": request.text_weight,
                "vision_weight": request.vision_weight,
                "use_llm": request.use_llm,
            },
        )

        return SearchResponse(
            query=request.query,
            request_id=request_id,
            results_count=len(result_dicts),
            results=result_dicts,
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
            text_engine = _get_multimodal_text_engine(db)
            allowed_filenames = _get_allowed_filenames(user, db)
            allowed_document_ids = _get_allowed_document_ids(user, db)
            results = text_engine.search(
                query=request.query,
                top_k=request.top_k * 3
                if allowed_filenames is not None
                else request.top_k,
                video_filter=request.video_filter,
            )
            results = _filter_results(
                results,
                allowed_filenames,
                limit=request.top_k,
                allowed_document_ids=allowed_document_ids,
            )
            search_time = time.time() - start_time
            request_id, result_dicts = _log_search_request_with_impressions(
                db=db,
                user=user,
                query_text=request.query,
                search_mode="search_multimodal_fallback_text",
                results=results,
                latency_seconds=search_time,
                filters={
                    "top_k": request.top_k,
                    "video_filter": request.video_filter,
                },
            )
            return SearchResponse(
                query=request.query,
                request_id=request_id,
                results_count=len(result_dicts),
                results=result_dicts,
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
    allowed_document_ids = _get_document_ids_for_filters(
        user,
        db,
        categories=cats,
        sites=sites,
        label=label,
    )

    try:
        text_weight, vision_weight = 0.5, 0.5

        start_time = time.time()

        mm_search = MultiModalSearchEngine(
            db=db,
            text_weight=text_weight,
            vision_weight=vision_weight,
            text_search=_get_multimodal_text_engine(db),
        )

        fallback_data = mm_search.search_with_fallback(
            query=q,
            top_k=limit * 3 if allowed_filenames else limit,
            video_filter=video,
            use_llm=use_llm,
            facet=facet or "auto",
        )

        results = _filter_results(
            fallback_data["results"],
            allowed_filenames,
            limit=limit,
            allowed_document_ids=allowed_document_ids,
        )
        metadata = fallback_data["search_metadata"]
        search_time = time.time() - start_time

        request_id, result_dicts = _log_search_request_with_impressions(
            db=db,
            user=user,
            query_text=q,
            search_mode="search_multimodal_quick",
            results=results,
            latency_seconds=search_time,
            facet=metadata.get("facet_applied") or (facet or "auto"),
            filters={
                "limit": limit,
                "video_filter": video,
                "category": cats,
                "site": sites,
                "label": label,
                "use_llm": use_llm,
                "text_weight": text_weight,
                "vision_weight": vision_weight,
            },
        )

        # Build grouped-by-video structure so the frontend can show all occurrences
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
            "request_id": request_id,
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
            text_engine = _get_multimodal_text_engine(db)
            fallback_data = text_engine.search_with_fallback(
                query=q,
                top_k=limit * 3 if allowed_filenames else limit,
                video_filter=video,
                facet=facet or "auto",
            )
            results = _filter_results(
                fallback_data["results"],
                allowed_filenames,
                limit=limit,
                allowed_document_ids=allowed_document_ids,
            )
            metadata = fallback_data["search_metadata"]
            search_time = time.time() - start_time

            request_id, result_dicts = _log_search_request_with_impressions(
                db=db,
                user=user,
                query_text=q,
                search_mode="search_multimodal_quick_fallback_text",
                results=results,
                latency_seconds=search_time,
                facet=metadata.get("facet_applied") or (facet or "auto"),
                filters={
                    "limit": limit,
                    "video_filter": video,
                    "category": cats,
                    "site": sites,
                    "label": label,
                },
            )
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
                "request_id": request_id,
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
    - Uploads the image and generates a vision embedding (SigLIP 2)
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
        allowed_document_ids = _get_allowed_document_ids(user, db)
        results = _filter_results(
            results,
            allowed_filenames,
            allowed_document_ids=allowed_document_ids,
        )

        search_time = time.time() - start_time
        request_id, result_dicts = _log_search_request_with_impressions(
            db=db,
            user=user,
            query_text=f"[image] {file.filename or 'uploaded_image'}",
            search_mode="search_visual_image",
            results=results,
            latency_seconds=search_time,
            filters={"limit": limit, "video_filter": video},
        )

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
            "request_id": request_id,
            "search_type": "reverse_image_search",
            "results_count": len(result_dicts),
            "results": result_dicts,
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
        allowed_document_ids = _get_allowed_document_ids(user, db)
        results = _filter_results(
            results,
            allowed_filenames,
            allowed_document_ids=allowed_document_ids,
        )

        search_time = time.time() - start_time
        request_id, result_dicts = _log_search_request_with_impressions(
            db=db,
            user=user,
            query_text=text_query or f"[image] {file.filename or 'uploaded_image'}",
            search_mode="search_visual_combined",
            results=results,
            latency_seconds=search_time,
            filters={
                "limit": limit,
                "video_filter": video,
                "image_weight": image_weight,
                "text_weight": text_weight,
            },
        )

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
            "request_id": request_id,
            "search_type": "combined_image_text"
            if text_query.strip()
            else "reverse_image_search",
            "image_weight": image_weight,
            "text_weight": text_weight,
            "results_count": len(result_dicts),
            "results": result_dicts,
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
    - Searches visual embeddings (SigLIP 2) directly
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
        allowed_document_ids = _get_allowed_document_ids(user, db)
        results = _filter_results(
            results,
            allowed_filenames,
            allowed_document_ids=allowed_document_ids,
        )

        search_time = time.time() - start_time
        request_id, result_dicts = _log_search_request_with_impressions(
            db=db,
            user=user,
            query_text=q,
            search_mode="search_visual_only",
            results=results,
            latency_seconds=search_time,
            filters={"limit": limit, "video_filter": video},
        )

        return {
            "query": q,
            "request_id": request_id,
            "search_type": "visual_only",
            "results_count": len(result_dicts),
            "results": result_dicts,
            "search_time_seconds": round(search_time, 3),
        }

    except Exception as e:
        print(f"Visual search error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Visual search failed: {str(e)}")


@app.get("/search/visual/videos")
async def visual_video_search(
    q: str = Query(..., description="Whole-video visual search query", min_length=1),
    limit: int = Query(10, description="Number of videos", ge=1, le=50),
    video: Optional[str] = Query(None, description="Filter by video filename"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Search video-level visual embeddings for whole-video discovery."""
    start_time = time.time()
    try:
        from search.visual_search import VisualSearchEngine

        visual_engine = VisualSearchEngine(db)
        rows = visual_engine.search_video_level(q, top_k=limit * 3, video_filter=video)
        allowed_filenames = _get_allowed_filenames(user, db)
        if allowed_filenames is not None:
            rows = [r for r in rows if r.get("video_filename") in allowed_filenames]
        rows = rows[:limit]
        return {
            "query": q,
            "search_type": "video_level_visual",
            "results_count": len(rows),
            "results": rows,
            "search_time_seconds": round(time.time() - start_time, 3),
        }
    except Exception as e:
        print(f"Video-level visual search error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Video-level visual search failed: {str(e)}",
        )


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


@app.post("/feedback/event")
async def log_feedback_event(
    payload: FeedbackEventRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Log an implicit interaction signal (click, open, dwell, copy, etc.)."""
    if not re.match(r"^[a-z0-9_:\-]{2,40}$", payload.interaction_type):
        raise HTTPException(
            status_code=400,
            detail="interaction_type must match ^[a-z0-9_:\\-]{2,40}$",
        )

    request_row = _resolve_search_request_for_user(db, user, payload.request_id)
    if request_row is None:
        raise HTTPException(status_code=404, detail="Search request not found")

    impression_row = None
    if payload.impression_id is not None:
        impression_row = (
            db.query(SearchImpression)
            .filter(
                SearchImpression.id == payload.impression_id,
                SearchImpression.request_id == request_row.id,
            )
            .first()
        )
        if impression_row is None:
            raise HTTPException(status_code=404, detail="Impression not found")

    try:
        event = SearchInteraction(
            request_id=request_row.id,
            impression_id=impression_row.id if impression_row else None,
            user_id=user.id,
            interaction_type=payload.interaction_type,
            dwell_ms=payload.dwell_ms,
            event_metadata=payload.metadata or {},
        )
        db.add(event)
        db.commit()
        return {"status": "ok", "event_id": event.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to log event: {e}")


@app.post("/feedback/rating")
async def log_feedback_rating(
    payload: FeedbackRatingRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Log explicit user relevance label for a shown result."""
    if payload.feedback_value not in (-1, 1):
        raise HTTPException(
            status_code=400, detail="feedback_value must be -1 or 1"
        )

    request_row = _resolve_search_request_for_user(db, user, payload.request_id)
    if request_row is None:
        raise HTTPException(status_code=404, detail="Search request not found")

    impression_row = None
    if payload.impression_id is not None:
        impression_row = (
            db.query(SearchImpression)
            .filter(
                SearchImpression.id == payload.impression_id,
                SearchImpression.request_id == request_row.id,
            )
            .first()
        )
        if impression_row is None:
            raise HTTPException(status_code=404, detail="Impression not found")

    try:
        feedback = SearchFeedback(
            request_id=request_row.id,
            impression_id=impression_row.id if impression_row else None,
            user_id=user.id,
            feedback_value=payload.feedback_value,
            comment=(payload.comment or "").strip() or None,
            feedback_metadata=payload.metadata or {},
        )
        db.add(feedback)
        db.commit()
        return {"status": "ok", "feedback_id": feedback.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to log feedback: {e}")


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

    # ── Translation follow-up path (translate previous assistant message) ─
    if _is_translate_request(user_question):
        previous_assistant = None
        for msg in reversed(request.messages):
            if msg.role == "assistant" and msg.content and msg.content.strip():
                previous_assistant = msg.content.strip()
                break

        target_code = _extract_target_language_code(
            question=user_question,
            language_hint=request.language,
        )

        if not previous_assistant:
            translation_text = (
                "I couldn't find a previous assistant answer to translate."
            )
        elif not target_code:
            translation_text = (
                "Please select a target language first, then ask me to translate."
            )
        else:
            translation_text = _translate_text_via_mymemory(
                text=previous_assistant,
                target=target_code,
                source="auto",
            )

        if request.stream:

            async def translation_sse_generator():
                rid = f"chatcmpl-{int(time.time())}"
                payload = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "ATLAS",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": translation_text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(payload)}\n\n"
                done_payload = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "ATLAS",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(done_payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                translation_sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "ATLAS",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": translation_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

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

    allowed_filenames = _get_allowed_filenames(user, db)

    # Load a request-bound StreamingVideoQA wrapper around shared model resources.
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
                    allowed_filenames=allowed_filenames,
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
            allowed_filenames=allowed_filenames,
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

    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if not user_can_access_video(user, str(video.filename)):
        raise HTTPException(status_code=403, detail="Not authorized for this video")

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

    kf = _resolve_keyframe_file_path(scene.keyframe_path)
    if not kf:
        raise HTTPException(status_code=404, detail="Keyframe file not found on disk")

    if kf.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(status_code=400, detail="Invalid keyframe file type")

    return FileResponse(str(kf), media_type=_media_type_for_image(kf))


@app.get("/keyframe")
async def serve_keyframe(
    path: str = Query(..., description="Path to keyframe image"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Serve keyframe images for thumbnails in search results."""
    from database.models import Scene as SceneModel

    keyframe_path = _resolve_keyframe_file_path(path)
    if not keyframe_path:
        raise HTTPException(status_code=404, detail="Keyframe not found")

    if keyframe_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(status_code=400, detail="Invalid file type")

    scene = (
        db.query(SceneModel)
        .join(Video, SceneModel.video_id == Video.id)
        .filter(SceneModel.keyframe_path == path)
        .first()
    )
    if scene is None:
        basename = os.path.basename(path)
        candidates = (
            db.query(SceneModel)
            .join(Video, SceneModel.video_id == Video.id)
            .filter(SceneModel.keyframe_path.ilike(f"%{basename}"))
            .limit(20)
            .all()
        )
        for candidate in candidates:
            candidate_path = _resolve_keyframe_file_path(candidate.keyframe_path)
            if candidate_path == keyframe_path:
                scene = candidate
                break

    if scene is None or not user_can_access_video(user, str(scene.video.filename)):
        raise HTTPException(status_code=403, detail="Not authorized for this keyframe")

    return FileResponse(str(keyframe_path), media_type=_media_type_for_image(keyframe_path))


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
    Re-enrich existing scenes with Qwen2.5-VL captions, object labels, and OCR text.

    Finds scenes missing captions, object labels, or OCR, runs the visual model
    on each keyframe, saves the results back to the scenes table, and refreshes a
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
        from database.ingest import DataIngester

        ingester = DataIngester(db)

        # Query scenes that need enrichment: missing caption, labels, or OCR attempt.
        query = db.query(Scene).filter(Scene.keyframe_path.isnot(None))
        if video_id:
            query = query.filter(Scene.video_id == video_id)

        # Split into: still need enrichment vs already done
        all_scenes = query.all()
        unenriched = [
            s
            for s in all_scenes
            if ingester._clean_optional_text(s.caption) is None
            or not ingester._normalize_object_labels(s.object_labels)
            or (
                ingester._clean_optional_text(s.ocr_text) is None
                and s.ocr_processed_at is None
            )
        ]
        total_remaining = len(unenriched)

        if total_remaining == 0:
            return {
                "status": "already_complete",
                "message": "All scenes with keyframes already have captions, labels, and OCR.",
                "scenes_enriched": 0,
                "scenes_remaining": 0,
            }

        batch = unenriched[:batch_size]

        # Load Qwen2.5-VL via SceneDetector (lazy-loads the model)
        cfg = SceneConfig(enable_visual_enrichment=True)
        detector = SceneDetector(config=cfg)
        qwen = detector._ensure_qwen_vl()
        if qwen is None:
            raise HTTPException(
                status_code=503,
                detail="Qwen2.5-VL model could not be loaded. Check server logs.",
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
                caption = ingester._clean_optional_text(result.get("caption"))
                object_labels = ingester._normalize_object_labels(
                    result.get("object_labels", [])
                )
                ocr_text = ingester._clean_optional_text(result.get("ocr_text"))

                # Update scene columns
                scene.caption = caption
                scene.object_labels = object_labels
                scene.ocr_text = ocr_text
                scene.ocr_text_norm = ingester._normalize_ocr_text(ocr_text)
                if ocr_text or scene.ocr_processed_at is None:
                    scene.ocr_processed_at = datetime.utcnow()
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

                    # Upsert/refresh the scene text embedding after enrichment changes.
                    existing_emb = (
                        db.query(Embedding)
                        .filter(
                            Embedding.scene_id == scene.id,
                            Embedding.segment_id == None,  # noqa: E711
                            Embedding.embedding_model == emb_gen.model_name,
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
    """Returns scene visual-enrichment coverage."""
    from database.models import Scene

    total = db.query(Scene).count()
    with_caption = db.query(Scene).filter(Scene.caption.isnot(None)).count()
    with_ocr = db.query(Scene).filter(Scene.ocr_text.isnot(None)).count()
    def _labels_present(value):
        if isinstance(value, list):
            return len(value) > 0
        if isinstance(value, str):
            raw = value.strip()
            return bool(raw and raw not in {"[]", "null", "None"})
        return bool(value)

    with_labels = sum(
        1
        for scene in db.query(Scene.object_labels).all()
        if _labels_present(scene[0])
    )
    keyframed_scenes = db.query(Scene).filter(Scene.keyframe_path.isnot(None)).all()
    with_keyframe = len(keyframed_scenes)
    scenes_needing = sum(
        1
        for scene in keyframed_scenes
        if not scene.caption
        or not _labels_present(scene.object_labels)
        or (not scene.ocr_text and scene.ocr_processed_at is None)
    )
    return {
        "total_scenes": total,
        "scenes_with_caption": with_caption,
        "scenes_with_ocr": with_ocr,
        "scenes_with_object_labels": with_labels,
        "scenes_needing_enrichment": scenes_needing,
        "scenes_without_keyframe": total - with_keyframe,
        "caption_coverage_pct": round(with_caption / total * 100, 1) if total else 0,
        "ocr_coverage_pct": round(with_ocr / total * 100, 1) if total else 0,
        "object_label_coverage_pct": round(with_labels / total * 100, 1) if total else 0,
    }



# ══════════════════════════════════════════════════════════════════════════
#  DOCUMENT PIPELINE ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════


class DocumentInfo(BaseModel):
    """Document information model."""
    id: int
    filename: str
    file_type: str
    file_size_mb: Optional[float] = None
    total_pages: Optional[int] = None
    extraction_method: Optional[str] = None
    label: Optional[str] = None
    category_id: Optional[int] = None
    category: Optional[str] = None
    processed_at: Optional[str] = None


class DocumentPageResponse(BaseModel):
    """Paginated document listing response."""
    items: List[DocumentInfo]
    next_cursor: Optional[int] = None
    total_count: Optional[int] = None


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Upload and process a document (PDF, DOCX, PPTX, image).
    Extracts text, generates embeddings, and stores in DB.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_DOCUMENT_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(_ALLOWED_DOCUMENT_EXT))}",
        )

    upload_dir = DOCUMENTS_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_filename = _safe_upload_filename(file.filename or "", "document", suffix)
    dest_path = (upload_dir / safe_filename).resolve()
    if dest_path.exists():
        stem = Path(safe_filename).stem
        safe_filename = f"{stem[:100]}_{uuid.uuid4().hex[:8]}{suffix}"
        dest_path = (upload_dir / safe_filename).resolve()
    if not _is_within_directory(dest_path, upload_dir):
        raise HTTPException(status_code=400, detail="Invalid upload path")

    max_bytes = int(os.getenv("MAX_DOCUMENT_UPLOAD_MB", "100")) * 1024 * 1024
    written = 0
    try:
        with open(dest_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Document exceeds {max_bytes // (1024 * 1024)} MB limit",
                    )
                f.write(chunk)
    except HTTPException:
        dest_path.unlink(missing_ok=True)
        raise

    try:
        from document_pipeline import DocumentPipeline

        pipeline = DocumentPipeline(skip_ingest=False)
        results = await asyncio.to_thread(pipeline.process_file, str(dest_path))

        return {
            "status": "ok",
            "filename": safe_filename,
            "chunks": len(results.get("chunks", [])),
            "pages": results.get("metadata", {}).get("total_pages", 0),
            "extraction_method": results.get("metadata", {}).get("extraction_method"),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all ingested documents."""
    try:
        from database.document_models import Document as DocumentModel

        acl_clause = _document_acl_clause(user)
        query = (
            db.query(
                DocumentModel.id,
                DocumentModel.filename,
                DocumentModel.file_type,
                DocumentModel.file_size_mb,
                DocumentModel.total_pages,
                DocumentModel.extraction_method,
                DocumentModel.label,
                DocumentModel.category_id,
                DocumentModel.processed_at,
                VideoCategory.name.label("category_name"),
            )
            .outerjoin(VideoCategory, DocumentModel.category_id == VideoCategory.id)
        )
        if acl_clause is not None:
            query = query.filter(acl_clause)

        docs = query.order_by(DocumentModel.id.desc()).all()
        return [
            DocumentInfo(
                id=d.id,
                filename=d.filename,
                file_type=d.file_type,
                file_size_mb=d.file_size_mb,
                total_pages=d.total_pages,
                extraction_method=d.extraction_method,
                label=d.label,
                category_id=d.category_id,
                category=d.category_name,
                processed_at=d.processed_at.isoformat() if d.processed_at else None,
            )
            for d in docs
        ]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/page", response_model=DocumentPageResponse)
async def list_documents_page(
    limit: int = Query(50, ge=1, le=200, description="Page size"),
    cursor: Optional[int] = Query(
        None, ge=1, description="Return rows with id < cursor (descending keyset)"
    ),
    include_total: bool = Query(
        False,
        description="Compute total accessible document count (set true for first page only)",
    ),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Efficient paginated document listing (keyset pagination)."""
    try:
        from database.document_models import Document as DocumentModel

        acl_clause = _document_acl_clause(user)

        page_query = (
            db.query(
                DocumentModel.id,
                DocumentModel.filename,
                DocumentModel.file_type,
                DocumentModel.file_size_mb,
                DocumentModel.total_pages,
                DocumentModel.extraction_method,
                DocumentModel.label,
                DocumentModel.category_id,
                DocumentModel.processed_at,
                VideoCategory.name.label("category_name"),
            )
            .outerjoin(VideoCategory, DocumentModel.category_id == VideoCategory.id)
        )
        if acl_clause is not None:
            page_query = page_query.filter(acl_clause)
        if cursor is not None:
            page_query = page_query.filter(DocumentModel.id < cursor)

        rows = (
            page_query.order_by(DocumentModel.id.desc())
            .limit(limit + 1)
            .all()
        )

        has_more = len(rows) > limit
        rows = rows[:limit]
        next_cursor = rows[-1].id if has_more and rows else None

        total_count = None
        if include_total:
            count_query = (
                db.query(DocumentModel.id)
                .outerjoin(VideoCategory, DocumentModel.category_id == VideoCategory.id)
            )
            if acl_clause is not None:
                count_query = count_query.filter(acl_clause)
            total_count = count_query.count()

        return DocumentPageResponse(
            items=[
                DocumentInfo(
                    id=row.id,
                    filename=row.filename,
                    file_type=row.file_type,
                    file_size_mb=row.file_size_mb,
                    total_pages=row.total_pages,
                    extraction_method=row.extraction_method,
                    label=row.label,
                    category_id=row.category_id,
                    category=row.category_name,
                    processed_at=row.processed_at.isoformat()
                    if row.processed_at
                    else None,
                )
                for row in rows
            ],
            next_cursor=next_cursor,
            total_count=total_count,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/stream/{doc_id}")
async def stream_document(
    doc_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Serve the original document file for in-browser viewing."""
    try:
        from database.document_models import Document as DocumentModel

        doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        allowed_document_ids = _get_allowed_document_ids(user, db)
        if (
            allowed_document_ids is not None
            and doc.id not in allowed_document_ids
        ):
            raise HTTPException(
                status_code=403, detail="Access denied to this document"
            )

        resolved_path = _resolve_document_file_path(doc.file_path)
        if resolved_path is None:
            raise HTTPException(
                status_code=404, detail=f"Document file not found: {doc.file_path}"
            )

        media_type = mimetypes.guess_type(str(resolved_path))[0]
        if not media_type:
            fallback_types = {
                "pdf": "application/pdf",
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "tiff": "image/tiff",
                "doc": "application/msword",
                "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "ppt": "application/vnd.ms-powerpoint",
                "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            }
            media_type = fallback_types.get(
                str(getattr(doc, "file_type", "")).lower(),
                "application/octet-stream",
            )
        safe_filename = doc.filename.replace('"', "")
        return FileResponse(
            path=str(resolved_path),
            media_type=media_type,
            filename=safe_filename,
            content_disposition_type="inline",
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: int,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Delete a document and its chunks/embeddings (admin only)."""
    try:
        from database.document_models import Document as DocumentModel

        doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        filename = doc.filename
        db.delete(doc)  # Cascade deletes chunks & embeddings
        db.commit()
        return {"status": "ok", "message": f"Document '{filename}' deleted"}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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
