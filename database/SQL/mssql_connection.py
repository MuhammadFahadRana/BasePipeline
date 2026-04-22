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


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on", "y"}


def _default_local_server() -> str:
    host = _env("MSSQL_HOST", "localhost")
    instance = _env("MSSQL_INSTANCE", "SQLEXPRESS")
    if instance:
        return f"{host}\\{instance}"
    return host


MSSQL_ALLOW_DB_ENV_FALLBACK = _env_bool("MSSQL_ALLOW_DB_ENV_FALLBACK", False)
_DB_HOST_FALLBACK = _env("DB_HOST") if MSSQL_ALLOW_DB_ENV_FALLBACK else None
_DB_PORT_FALLBACK = _env("DB_PORT") if MSSQL_ALLOW_DB_ENV_FALLBACK else None
_DB_NAME_FALLBACK = _env("DB_NAME") if MSSQL_ALLOW_DB_ENV_FALLBACK else None
_DB_USER_FALLBACK = _env("DB_USER") if MSSQL_ALLOW_DB_ENV_FALLBACK else None
_DB_PASSWORD_FALLBACK = _env("DB_PASSWORD") if MSSQL_ALLOW_DB_ENV_FALLBACK else None

MSSQL_CONNECTOR = (_env("MSSQL_CONNECTOR") or ("pytds" if os.name != "nt" else "pyodbc")).lower()
MSSQL_SERVER = _env("MSSQL_SERVER", _DB_HOST_FALLBACK or _default_local_server())
MSSQL_PORT = int(_env("MSSQL_PORT", _DB_PORT_FALLBACK or "1433") or "1433")
MSSQL_DATABASE = _env("MSSQL_DATABASE", _DB_NAME_FALLBACK or "VideoSemanticDB")
MSSQL_USER = _env("MSSQL_USER", _DB_USER_FALLBACK)
MSSQL_PASSWORD = _env("MSSQL_PASSWORD", _DB_PASSWORD_FALLBACK)
MSSQL_DRIVER = _env("MSSQL_DRIVER", "ODBC Driver 18 for SQL Server")
MSSQL_TRUSTED_CONNECTION = _env("MSSQL_TRUSTED_CONNECTION", "yes")
MSSQL_TRUST_SERVER_CERTIFICATE = _env("MSSQL_TRUST_SERVER_CERTIFICATE", "yes")
MSSQL_ENCRYPT = _env("MSSQL_ENCRYPT", "no")


def _server_looks_like_version(value: str | None) -> bool:
    if not value:
        return False
    parts = value.split(".")
    if len(parts) < 3 or any(not part.isdigit() for part in parts):
        return False
    # Treat valid IPv4 literals as host values, not version strings.
    if len(parts) == 4 and all(0 <= int(part) <= 255 for part in parts):
        return False
    return True


def _connection_target_string() -> str:
    base = (
        f"connector={MSSQL_CONNECTOR}, server={MSSQL_SERVER}, "
        f"database={MSSQL_DATABASE}"
    )
    if MSSQL_CONNECTOR == "pytds" and MSSQL_SERVER and "\\" not in MSSQL_SERVER:
        base = f"{base}, port={MSSQL_PORT}"
    return base


def _print_connection_hints(exc: Exception) -> None:
    message = repr(exc)
    server_lower = (MSSQL_SERVER or "").lower()

    if (
        MSSQL_CONNECTOR == "pyodbc"
        and server_lower in {"localhost", "127.0.0.1", ".", "(local)"}
    ):
        print(
            "[HINT] SQL Server Express often uses a named instance. "
            "Try MSSQL_SERVER=localhost\\SQLEXPRESS (or your_machine\\SQLEXPRESS)."
        )

    if (
        not MSSQL_ALLOW_DB_ENV_FALLBACK
        and _env("MSSQL_SERVER") is None
        and _env("DB_HOST") is not None
    ):
        print(
            "[HINT] This helper now ignores generic DB_* env vars by default. "
            "Set MSSQL_SERVER / MSSQL_DATABASE explicitly for SQL Server."
        )

    if "08001" in message and "\\" not in (MSSQL_SERVER or ""):
        print(
            "[HINT] If this is a named instance, include host\\instance "
            "(for example: localhost\\SQLEXPRESS)."
        )


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

    if _server_looks_like_version(MSSQL_SERVER):
        print(
            f"[WARN] MSSQL_SERVER={MSSQL_SERVER!r} looks like a SQL Server version "
            "string, not a host. Set MSSQL_SERVER to a hostname/IP (or host\\instance)."
        )

    connect_kwargs = {
        "database": MSSQL_DATABASE,
        "user": MSSQL_USER,
        "password": MSSQL_PASSWORD,
        "login_timeout": 30,
        "timeout": 0,
    }

    # pytds forbids sending both an instance name and an explicit port.
    if "\\" in MSSQL_SERVER:
        server_name, instance_name = MSSQL_SERVER.split("\\", 1)
        connect_kwargs["server"] = server_name
        connect_kwargs["instance"] = instance_name
    else:
        connect_kwargs["server"] = MSSQL_SERVER
        connect_kwargs["port"] = MSSQL_PORT

    return pytds.connect(**connect_kwargs)


if MSSQL_CONNECTOR == "pytds":
    DATABASE_URL = "mssql+pytds://"
    try:
        engine = create_engine(
            DATABASE_URL,
            creator=_connect_pytds,
            pool_pre_ping=True,
            future=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to initialize SQLAlchemy pytds dialect. "
            "Install `python-tds` and `sqlalchemy-pytds` in this environment."
        ) from exc
else:
    DATABASE_URL = _build_pyodbc_engine_url()
    try:
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            future=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to initialize SQLAlchemy pyodbc dialect. "
            "Confirm `pyodbc` is installed and ODBC Driver 18 is available."
        ) from exc

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
        print(f"[ERROR] SQL Server connection failed ({_connection_target_string()})")
        print(f"[ERROR] Exception type: {type(exc).__name__}")
        print(f"[ERROR] Exception details: {exc!r}")
        _print_connection_hints(exc)
        return False


if __name__ == "__main__":
    test_connection()
