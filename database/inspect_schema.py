from database.mssql_connection import engine
from sqlalchemy import text
with engine.connect() as conn:
    cols = conn.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='scenes' ORDER BY ORDINAL_POSITION")).fetchall()
    print([c[0] for c in cols])
