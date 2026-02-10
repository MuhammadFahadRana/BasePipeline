"""Database configuration and connection setup - OPTIMIZED."""

import os
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "video_semantic_search")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# OPTIMIZATION #3: Performance tuning parameters
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "20"))         # Increased from 10
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "30"))   # Increased from 20
POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600")) # 1 hour (NEW)
POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))   # 30 seconds (NEW)
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "100")) # OPTIMIZATION #6

# Construct database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# OPTIMIZATION #3: Create optimized engine with better connection pooling
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL logging
    
    # Connection pool optimization
    poolclass=QueuePool,
    pool_size=POOL_SIZE,           # 20 connections (was 10)
    max_overflow=MAX_OVERFLOW,     # +30 overflow (was 20)
    pool_recycle=POOL_RECYCLE,     # Recycle connections after 1 hour
    pool_timeout=POOL_TIMEOUT,     # Wait up to 30s for connection
    pool_pre_ping=True,            # Verify connections before using
    
    # OPTIMIZATION #6 & #7: Performance settings
    connect_args={
        "options": f"-c hnsw.ef_search={HNSW_EF_SEARCH}",  # Better HNSW recall
        "connect_timeout": 10,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    },
)


# OPTIMIZATION #7: Set session parameters for better performance
@event.listens_for(engine, "connect")
def set_session_parameters(dbapi_conn, connection_record):
    """Optimize PostgreSQL parameters for each connection."""
    cursor = dbapi_conn.cursor()
    
    # Set work memory for complex queries
    cursor.execute("SET work_mem = '256MB'")
    
    # Optimize for SSD storage
    cursor.execute("SET random_page_cost = 1.1")
    
    # Enable parallel queries
    cursor.execute("SET max_parallel_workers_per_gather = 4")
    
    # Set effective cache size (adjust based on your RAM)
    cursor.execute("SET effective_cache_size = '4GB'")
    
    cursor.close()

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Get database session (for dependency injection)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database with schema."""
    from database.models import Base

    print(f"Creating database tables on {DB_HOST}:{DB_PORT}/{DB_NAME}...")
    Base.metadata.create_all(bind=engine)
    print("✓ Database initialized successfully")


def test_connection():
    """Test database connection."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"✓ Connected to PostgreSQL: {version}")
            return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False
