# SQL Server Express Setup (VideoSemanticDB)

This folder contains a SQL Server Express equivalent of PostgreSQL schema, including:

- Core video/search/auth schema
- Document ingestion schema
- Training/evaluation schema
- Optimization + maintenance procedures

The schema is idempotent (`IF NOT EXISTS` patterns), so it can be re-run safely.

## Files

- `00_create_or_reuse_database.sql`: Create `VideoSemanticDB` if missing, otherwise reuse it.
- `00_recreate_database.sql`: Drop and recreate `VideoSemanticDB` (destructive).
- `01_schema_sqlserver.sql`: Main schema (videos, scenes, embeddings, search telemetry, cache, procedures).
- `02_document_schema_sqlserver.sql`: Document tables + embeddings.
- `03_schema_training_sqlserver.sql`: `relevance_judgments` and `model_runs`.
- `04_apply_optimizations_sqlserver.sql`: Composite indexes + stats refresh.
- `05_run_all.sql`: SQLCMD include script to run all setup files in order.
- `06_add_projection_tables_sqlserver.sql`: Dual-storage projection tables for low-dim vector search.
- `mssql_connection.py`: SQLAlchemy connection helper.
- `test_mssql.py`: Quick connectivity + table listing check.
- `ingest_sqlserver.py`: Ingest `processed/` results into SQL Server tables.

## Important Notes on Embeddings

- SQL Server `VECTOR` has a 1998-dimension cap. If you use a >1998-dim Qwen text model, text/document/query embeddings fall back to `NVARCHAR(MAX)` JSON storage, while visual/image columns still use `VECTOR(768)`.
- Dual storage is supported: canonical full embeddings remain in `dbo.embeddings` / `dbo.document_embeddings`, and optional low-dim projected vectors are stored in `dbo.embedding_projections` / `dbo.document_embedding_projections`.
- If `VECTOR` is not available (common on many Express installs), embedding columns automatically fall back to `NVARCHAR(MAX)` JSON with `ISJSON` validation.
- This keeps embedding dimensions from blocking database creation.
- `ingest_sqlserver.py` uses the same `TEXT_EMBEDDING_MODEL` / `EMBEDDING_MODEL` resolution as runtime search, so ingestion and query embeddings stay compatible.
- For mixed English/Norwegian retrieval, use one multilingual text embedding model for all transcript segments and query variants.
- If switching to a model with a different vector dimension, update SQL schema `VECTOR(...)` dimensions in:
  - `01_schema_sqlserver.sql` (`embeddings`, `search_queries`)
  - `02_document_schema_sqlserver.sql` (`document_embeddings`)
    then recreate DB before ingest.

## Run From `sqlcmd` (Recommended)

Run from repository root:

```powershell
sqlcmd -S "LAPTOP-GMO7MPTH\SQLEXPRESS" -E -N o -i "database\SQL\05_run_all.sql" -b
```

Clean rebuild:

```powershell
sqlcmd -S "LAPTOP-GMO7MPTH\SQLEXPRESS" -E -N o -i "database\SQL\00_recreate_database.sql" -b
sqlcmd -S "LAPTOP-GMO7MPTH\SQLEXPRESS" -E -N o -i "database\SQL\01_schema_sqlserver.sql" -b
sqlcmd -S "LAPTOP-GMO7MPTH\SQLEXPRESS" -E -N o -i "database\SQL\02_document_schema_sqlserver.sql" -b
sqlcmd -S "LAPTOP-GMO7MPTH\SQLEXPRESS" -E -N o -i "database\SQL\03_schema_training_sqlserver.sql" -b
sqlcmd -S "LAPTOP-GMO7MPTH\SQLEXPRESS" -E -N o -i "database\SQL\04_apply_optimizations_sqlserver.sql" -b
```

If local SQL setup requires trust cert mode, add `-C`.

## Run From SSMS

1. Open each script in order.
2. Ensure target instance is `LAPTOP-GMO7MPTH\SQLEXPRESS`.
3. Execute:
   - `00_create_or_reuse_database.sql` (or `00_recreate_database.sql`)
   - `01_schema_sqlserver.sql`
   - `02_document_schema_sqlserver.sql`
   - `03_schema_training_sqlserver.sql`
   - `04_apply_optimizations_sqlserver.sql`

## Smoke Test

```powershell
python database/SQL/test_mssql.py
```

## Ingest Processed Data Into SQL Server

Full ingest (videos + documents + embeddings):

```powershell
python database/SQL/ingest_sqlserver.py
```

Example with explicit model override:

```powershell
python database/SQL/ingest_sqlserver.py --text-model "Qwen/Qwen3-Embedding-8B"
```

Documents only:

```powershell
python database/SQL/ingest_sqlserver.py --skip-videos
```

Videos only:

```powershell
python database/SQL/ingest_sqlserver.py --skip-documents
```

Fast metadata-only test (no embeddings):

```powershell
python database/SQL/ingest_sqlserver.py --no-text-embeddings --no-visual-embeddings
```

Python helper scripts require `pyodbc`:

```powershell
pip install pyodbc
```

If needed, set env vars before running Python helpers:

```powershell
$env:MSSQL_SERVER = "LAPTOP-GMO7MPTH\SQLEXPRESS"
$env:MSSQL_DATABASE = "VideoSemanticDB"
$env:TEXT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
$env:SEARCH_QUERY_TRANSLATION_ENABLED = "1"
$env:SEARCH_QUERY_TRANSLATION_PROVIDER = "mymemory"  # or "marian" / "nllb" for local models
$env:SEARCH_QUERY_TRANSLATION_TIMEOUT = "0.75"
$env:RERANKER_MODEL = "Qwen/Qwen3-4B-Instruct"
$env:RERANKER_MODE = "hybrid"
$env:RERANKER_BLEND = "0.70"
$env:MSSQL_TEXT_PROJECTION_DIM = "1024"
$env:MSSQL_ENABLE_TEXT_PROJECTION = "yes"
```

Local translation defaults:

- `marian`: `Helsinki-NLP/opus-mt-en-gmq` and `Helsinki-NLP/opus-mt-gmq-en`
- `nllb`: `facebook/nllb-200-distilled-600M`

`mssql_connection.py` now reads `MSSQL_*` settings only by default, so PostgreSQL
`DB_*` values in `.env` will not override SQL Server settings.

If you intentionally want MSSQL scripts to reuse `DB_*` vars, opt in:

```powershell
$env:MSSQL_ALLOW_DB_ENV_FALLBACK = "yes"
```

For existing databases, apply projection tables without rebuilding:

```powershell
sqlcmd -S "LAPTOP-GMO7MPTH\SQLEXPRESS" -E -N o -i "database\SQL\06_add_projection_tables_sqlserver.sql" -b
```

## Slurm / No-ODBC Setup

On Linux hosts without Microsoft ODBC drivers, the helper now defaults to `pytds`.

Install the pure-Python SQL Server packages:

```bash
pip install python-tds sqlalchemy-pytds
```

Run the smoke test or ingest with these env vars set:

```bash
export MSSQL_CONNECTOR=pytds
export MSSQL_PORT=1433
export MSSQL_DATABASE="VideoSemanticDB"
export MSSQL_SERVER="<reachable-sql-host-or-ip>"
export MSSQL_USER="<sql-login-username>"
export MSSQL_PASSWORD="<sql-login-password>"
python database/SQL/test_mssql.py
python database/SQL/ingest_sqlserver.py
```

`MSSQL_SERVER` must be a reachable hostname/IP (or `host\\instance`), not a SQL
Server version value such as `17.0.1000`.

If you must use a named instance, set `MSSQL_SERVER="host\\instance"` and leave `MSSQL_PORT` unset.
