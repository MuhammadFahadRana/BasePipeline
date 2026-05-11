"""Authentication & authorisation helpers for the ATLAS API.

Provides:
- Password hashing (bcrypt via hashlib fallback)
- JWT token creation / verification
- FastAPI dependency ``get_current_user`` that extracts the caller from
  a bearer header or HttpOnly auth cookie
- ``require_admin`` dependency that additionally enforces admin role
- ``get_video_category`` helper used to check per-video access
"""

import os
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt  # PyJWT
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from database.config import get_db
from database.models import User, UserCategoryAccess

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _get_stable_jwt_secret() -> str:
    """Return a JWT secret that persists only for the lifetime of the server process.

    This ensures that token invalidation happens automatically on server restart."""
    env = os.getenv("JWT_SECRET")
    if env:
        return env
    return secrets.token_hex(32)

JWT_SECRET = _get_stable_jwt_secret()
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))
AUTH_COOKIE_NAME = "atlas_token"

# ---------------------------------------------------------------------------
# Password hashing  (bcrypt when available, PBKDF2-SHA256 fallback)
# ---------------------------------------------------------------------------
_PBKDF2_ITERATIONS = 600_000

try:
    import bcrypt as _bcrypt  # type: ignore
    _HAS_BCRYPT = True
except ImportError:
    _HAS_BCRYPT = False


def _verify_pbkdf2(plain: str, hashed: str) -> bool:
    """Verify a PBKDF2-SHA256 hash (format: pbkdf2:sha256:<iter>$<salt>$<dk>)."""
    try:
        header, salt, stored_dk = hashed.split("$")
        iterations = int(header.split(":")[-1])
        dk = hashlib.pbkdf2_hmac("sha256", plain.encode(), salt.encode(), iterations)
        return hmac.compare_digest(dk.hex(), stored_dk)
    except Exception:
        return False


def hash_password(plain: str) -> str:
    if _HAS_BCRYPT:
        return _bcrypt.hashpw(plain.encode(), _bcrypt.gensalt()).decode()
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", plain.encode(), salt.encode(), _PBKDF2_ITERATIONS)
    return f"pbkdf2:sha256:{_PBKDF2_ITERATIONS}${salt}${dk.hex()}"


def verify_password(plain: str, hashed: str) -> bool:
    # Detect hash type and route accordingly
    if hashed.startswith("pbkdf2:"):
        return _verify_pbkdf2(plain, hashed)
    if _HAS_BCRYPT:
        try:
            return _bcrypt.checkpw(plain.encode(), hashed.encode())
        except Exception:
            return False
    return False


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------
def create_access_token(user_id: int, username: str, role: str) -> str:
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------
_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    request: Request,
    creds: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Extract and validate the JWT from Authorization or an auth cookie.

    The cookie path is used for browser-native media requests that cannot set
    custom headers.
    """
    raw_token = None
    if creds and creds.credentials:
        raw_token = creds.credentials
    elif request.cookies.get(AUTH_COOKIE_NAME):
        raw_token = request.cookies.get(AUTH_COOKIE_NAME)
    if not raw_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_access_token(raw_token)
    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Reject non-admin callers with 403."""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ---------------------------------------------------------------------------
# Category classification  (mirrors the frontend function)
# ---------------------------------------------------------------------------
def get_video_category(filename: str) -> str:
    """Derive category / installation from a video filename."""
    if filename.startswith("Johan Sverdrup"):
        return "Johan Sverdrup"
    if filename.startswith("AkerBP"):
        return "AkerBP"
    if filename.endswith("- TED Talk.mp4"):
        return "TED Talks"
    return "Other"


def user_can_access_video(user: User, filename: str) -> bool:
    """Return True if *user* is allowed to see the video with *filename*."""
    if user.role == "admin":
        return True
    allowed = {a.category for a in user.category_access}
    return get_video_category(filename) in allowed


def get_user_allowed_categories(user: User) -> Optional[set]:
    """Return the set of categories the user may see, or None for admins (= all)."""
    if user.role == "admin":
        return None
    return {a.category for a in user.category_access}
