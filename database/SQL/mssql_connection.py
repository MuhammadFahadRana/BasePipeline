"""SQL Server connection helper for local Express setup and Slurm jobs."""

import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is not None and value.strip():
        return value.strip()
    return default


MSSQL_CONNECTOR = (_env("MSSQL_CONNECTOR") or ("pytds" if os.name != "nt" else "pyodbc")).lower()
MSSQL_SERVER = _env("MSSQL_SERVER", _env("DB_HOST", r"LAPTOP-GMO7MPTH\SQLEXPRESS"))
MSSQL_PORT = int(_env("MSSQL_PORT", _env("DB_PORT", "1433")) or "1433")
MSSQL_DATABASE = _env("MSSQL_DATABASE", _env("DB_NAME", "VideoSemanticDB"))
MSSQL_USER = _env("MSSQL_USER", _env("DB_USER"))
MSSQL_PASSWORD = _env("MSSQL_PASSWORD", _env("DB_PASSWORD"))
MSSQL_DRIVER = _env("MSSQL_DRIVER", "ODBC Driver 18 for SQL Server")
MSSQL_TRUSTED_CONNECTION = _env("MSSQL_TRUSTED_CONNECTION", "yes")
MSSQL_TRUST_SERVER_CERTIFICATE = _env("MSSQL_TRUST_SERVER_CERTIFICATE", "yes")
MSSQL_ENCRYPT = _env("MSSQL_ENCRYPT", "no")


def _build_pyodbc_engine_url() -> str:
    odbc_connection = (
        f"Driver={{{MSSQL_DRIVER}}};"
        f"Server={MSSQL_SERVER};"
        f"Database={MSSQL_DATABASE};"
        f"Trusted_Connection={MSSQL_TRUSTED_CONNECTION};"
        f"TrustServerCertificate={MSSQL_TRUST_SERVER_CERTIFICATE};"
        f"Encrypt={MSSQL_ENCRYPT};"
    )
    return f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_connection)}"


def _connect_pytds():
    try:
        import pytds
    except ImportError as exc:
        raise RuntimeError(
            "pytds is required for the MSSQL_CONNECTOR=pytds backend. "
            "Install python-tds and sqlalchemy-pytds in the Slurm environment."
        ) from exc

    if not MSSQL_USER or not MSSQL_PASSWORD:
        raise RuntimeError(
            "MSSQL_USER and MSSQL_PASSWORD must be set for the pytds backend."
        )

    return pytds.connect(
        server=MSSQL_SERVER,
        port=MSSQL_PORT,
        database=MSSQL_DATABASE,
        user=MSSQL_USER,
        password=MSSQL_PASSWORD,
        login_timeout=30,
        timeout=0,
    )


if MSSQL_CONNECTOR == "pytds":
    DATABASE_URL = "mssql+pytds://"
    engine = create_engine(
        DATABASE_URL,
        creator=_connect_pytds,
        pool_pre_ping=True,
        future=True,
    )
else:
    DATABASE_URL = _build_pyodbc_engine_url()
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        future=True,
    )

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def test_connection() -> bool:
    """Quick connectivity smoke test."""
    try:
        with engine.connect() as conn:
            db_name = conn.execute(text("SELECT DB_NAME()")).scalar_one()
            version = conn.execute(text("SELECT @@VERSION")).scalar_one()
            print(f"[OK] Connected to SQL Server database: {db_name}")
            print(f"[OK] Server version: {version}")
        return True
    except Exception as exc:
        print(f"[ERROR] SQL Server connection failed: {exc}")
        return False


if __name__ == "__main__":
    test_connection()
