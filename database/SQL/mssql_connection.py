import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

SERVER = r"LAPTOP-GMO7MPTH\SQLEXPRESS"
DATABASE = "VideoSemanticDB"

# Windows Integrated Auth:
CONN_STR = (
    "mssql+pyodbc://@"
    + SERVER
    + "/"
    + DATABASE
    + "?driver=ODBC+Driver+18+for+SQL+Server"
    + "&trusted_connection=yes"
    + "&TrustServerCertificate=yes"
)

engine = create_engine(CONN_STR, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
