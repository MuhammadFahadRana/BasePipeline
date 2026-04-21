"""SQL Server connection helper for local Express setup."""

import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

MSSQL_SERVER = os.getenv("MSSQL_SERVER", r"LAPTOP-GMO7MPTH\SQLEXPRESS")
MSSQL_DATABASE = os.getenv("MSSQL_DATABASE", "VideoSemanticDB")
MSSQL_DRIVER = os.getenv("MSSQL_DRIVER", "ODBC Driver 18 for SQL Server")
MSSQL_TRUSTED_CONNECTION = os.getenv("MSSQL_TRUSTED_CONNECTION", "yes")
MSSQL_TRUST_SERVER_CERTIFICATE = os.getenv("MSSQL_TRUST_SERVER_CERTIFICATE", "yes")
MSSQL_ENCRYPT = os.getenv("MSSQL_ENCRYPT", "no")

ODBC_CONNECTION = (
    f"Driver={{{MSSQL_DRIVER}}};"
    f"Server={MSSQL_SERVER};"
    f"Database={MSSQL_DATABASE};"
    f"Trusted_Connection={MSSQL_TRUSTED_CONNECTION};"
    f"TrustServerCertificate={MSSQL_TRUST_SERVER_CERTIFICATE};"
    f"Encrypt={MSSQL_ENCRYPT};"
)

DATABASE_URL = f"mssql+pyodbc:///?odbc_connect={quote_plus(ODBC_CONNECTION)}"

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
