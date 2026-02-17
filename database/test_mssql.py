from sqlalchemy import text
from db.mssql_connection import engine

with engine.connect() as conn:
    dbname = conn.execute(text("SELECT DB_NAME()")).scalar_one()
    tables = conn.execute(text("SELECT name FROM sys.tables ORDER BY name")).fetchall()
    print("Connected to:", dbname)
    print("Tables:", [t[0] for t in tables])
