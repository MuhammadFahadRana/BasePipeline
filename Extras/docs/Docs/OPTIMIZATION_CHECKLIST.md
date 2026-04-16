# Optimization Implementation Checklist

Use this checklist to track your implementation progress.

---

## ⏱️ Estimated Total Time: 30 minutes

---

## 📋 PRE-IMPLEMENTATION

- [ ] **Backup database** (recommended)
  ```bash
  pg_dump -U postgres video_semantic_search > backup_$(date +%Y%m%d).sql
  ```

- [ ] **Verify database connection**
  ```bash
  psql -U postgres -d video_semantic_search -c "SELECT version();"
  ```

- [ ] **Check if API is running** (you'll need to restart it)
  ```bash
  # Note which terminal has the API running
  ```

---

## 🔧 STEP 1: APPLY SQL OPTIMIZATIONS (5 minutes)

- [ ] **Run the SQL script**
  ```bash
  psql -U postgres -d video_semantic_search -f database/apply_optimizations.sql
  ```

- [ ] **Verify indexes created**
  ```bash
  psql -U postgres -d video_semantic_search -c "\di"
  ```
  
  Check for these indexes:
  - [ ] `idx_transcript_video_time`
  - [ ] `idx_embeddings_segment_model`
  - [ ] `idx_visual_embeddings_scene_model`
  - [ ] `idx_embeddings_vector` (should be recreated)
  - [ ] `idx_query_cache_hash`

- [ ] **Verify cache table created**
  ```bash
  psql -U postgres -d video_semantic_search -c "\dt query_cache"
  ```

- [ ] **Check HNSW parameter**
  ```bash
  psql -U postgres -d video_semantic_search -c "SHOW hnsw.ef_search;"
  ```
  Should show: `100`

---

## 💻 STEP 2: UPDATE CODE (10 minutes)

### **A. Update API (if using)**

- [ ] **Open `api/app.py`**

- [ ] **Find search engine import** (around line 10-15):
  ```python
  from search.semantic_search import SemanticSearchEngine
  ```

- [ ] **Replace with:**
  ```python
  from search.optimized_search import OptimizedSearchEngine
  ```

- [ ] **Find engine initialization** (around line 20-30):
  ```python
  search_engine = SemanticSearchEngine(db)
  ```

- [ ] **Replace with:**
  ```python
  search_engine = OptimizedSearchEngine(db, cache_enabled=True, parallel_enabled=True)
  ```

### **B. Update any other files using search**

- [ ] **Check if you have custom search scripts**
  ```bash
  grep -r "SemanticSearchEngine" --include="*.py"
  ```

- [ ] **Update each file found** with the same changes as above

---

## 🔄 STEP 3: RESTART SERVICES (2 minutes)

- [ ] **Stop API** (Ctrl+C in terminal where it's running)

- [ ] **Verify config changes loaded**
  ```python
  from database.config import POOL_SIZE, HNSW_EF_SEARCH
  print(f"Pool: {POOL_SIZE}, HNSW: {HNSW_EF_SEARCH}")
  # Should show: Pool: 20, HNSW: 100
  ```

- [ ] **Restart API**
  ```bash
  python api/app.py
  ```

- [ ] **Wait for "Application startup complete" message**

---

## ✅ STEP 4: VERIFICATION (10 minutes)

### **A. Quick Functionality Test**

- [ ] **Test basic search** (API or direct):
  ```bash
  curl http://localhost:8000/api/search?query=test&top_k=5
  ```
  Should return results (not error)

- [ ] **Test repeated query** (should be faster):
  ```bash
  curl http://localhost:8000/api/search?query=test&top_k=5
  ```

### **B. Performance Test**

- [ ] **Create test script** `test_performance.py`:
  ```python
  from database.config import SessionLocal
  from search.optimized_search import OptimizedSearchEngine
  import time

  db = SessionLocal()
  search = OptimizedSearchEngine(db)

  # Query 1 (no cache)
  print("Query 1 (cold):")
  start = time.time()
  results = search.search("test drilling operations", top_k=10)
  time1 = (time.time() - start) * 1000
  print(f"  Time: {time1:.0f}ms")

  # Query 2 (cached)
  print("\nQuery 2 (cached):")
  start = time.time()
  results = search.search("test drilling operations", top_k=10)
  time2 = (time.time() - start) * 1000
  print(f"  Time: {time2:.0f}ms")
  print(f"  Speedup: {time1/time2:.0f}x")

  # Stats
  print("\nStats:")
  for k, v in search.get_stats().items():
      print(f"  {k}: {v}")
  ```

- [ ] **Run test:**
  ```bash
  python test_performance.py
  ```

- [ ] **Verify results:**
  - [ ] First query: 40-100ms
  - [ ] Second query: 1-5ms
  - [ ] Speedup: 20-100x
  - [ ] Cache hit rate: 50%

### **C. Database Verification**

- [ ] **Check cache is working:**
  ```sql
  SELECT COUNT(*) FROM query_cache;
  ```
  Should show > 0

- [ ] **Check index usage:**
  ```sql
  SELECT indexname, idx_scan 
  FROM pg_stat_user_indexes 
  WHERE schemaname = 'public' 
  ORDER BY idx_scan DESC 
  LIMIT 10;
  ```
  Should show your new indexes being scanned

---

## 📊 STEP 5: MONITORING (Ongoing)

### **Daily:**

- [ ] **Check cache hit rate**
  ```python
  from database.config import SessionLocal
  from search.optimized_search import OptimizedSearchEngine
  
  db = SessionLocal()
  search = OptimizedSearchEngine(db)
  stats = search.get_stats()
  print(f"Hit rate: {stats['hit_rate']}")
  ```
  Target: >70% after a few days

### **Weekly:**

- [ ] **Clean expired cache**
  ```python
  deleted = search.cleanup_expired_cache()
  print(f"Cleaned {deleted} entries")
  ```

- [ ] **Check query performance**
  ```sql
  SELECT 
      query_text, 
      hit_count, 
      last_used 
  FROM query_cache 
  ORDER BY hit_count DESC 
  LIMIT 10;
  ```

### **Monthly:**

- [ ] **Update statistics**
  ```sql
  VACUUM ANALYZE embeddings;
  VACUUM ANALYZE transcript_segments;
  VACUUM ANALYZE query_cache;
  ```

- [ ] **Refresh analytics**
  ```sql
  REFRESH MATERIALIZED VIEW top_searched_segments;
  ```

---

## 🐛 TROUBLESHOOTING

### **Issue: SQL script fails**

- [ ] Check PostgreSQL version: `SELECT version();` (need 15+)
- [ ] Check pgvector installed: `SELECT * FROM pg_extension WHERE extname='vector';`
- [ ] Check permissions: `GRANT ALL ON DATABASE video_semantic_search TO postgres;`

### **Issue: ImportError**

- [ ] Verify file exists: `ls -la search/optimized_search.py`
- [ ] Check Python path: `echo $PYTHONPATH`
- [ ] Restart Python interpreter/IDE

### **Issue: Slow queries**

- [ ] Check if indexes exist: `\di` in psql
- [ ] Check if cache enabled: `search.cache_enabled` should be `True`
- [ ] Check if parallel enabled: `search.parallel_enabled` should be `True`
- [ ] Run `EXPLAIN ANALYZE` on your queries

### **Issue: Cache not working**

- [ ] Check table exists: `\dt query_cache`
- [ ] Check for errors in logs
- [ ] Try clearing cache: `search.clear_cache()`
- [ ] Verify cache_enabled=True when creating engine

---

## 📈 EXPECTED RESULTS

After implementation, you should see:

**Query Performance:**
- ✅ 3-5x faster first-time queries
- ✅ 100x faster repeated queries
- ✅ Consistent latency (no spikes)

**Database:**
- ✅ More indexes (10-15 total)
- ✅ query_cache table with entries
- ✅ HNSW ef_search = 100

**Code:**
- ✅ OptimizedSearchEngine in use
- ✅ Cache statistics available
- ✅ Parallel execution enabled

**Ingestion:**
- ✅ 2-3x faster embedding generation
- ✅ Batch processing logs visible

---

## ✅ COMPLETION

- [ ] **All SQL optimizations applied**
- [ ] **All code updated**
- [ ] **API restarted successfully**
- [ ] **Performance tests passing**
- [ ] **Cache working (hit rate >0%)**
- [ ] **No errors in logs**

**Congratulations! Your database is now optimized! 🎉**

---

## 📝 NOTES

Add any observations or issues here:

```
Date: _____________
Notes:











```

---

## 📚 REFERENCE

- **Full guide:** `OPTIMIZATION_IMPLEMENTATION.md`
- **Quick summary:** `OPTIMIZATIONS_SUMMARY.md`
- **Architecture analysis:** `Docs/DATABASE_ARCHITECTURE_ANALYSIS.md`
- **Research Q&A:** `Docs/RESEARCH_QUESTIONS_ANSWERS.md`
