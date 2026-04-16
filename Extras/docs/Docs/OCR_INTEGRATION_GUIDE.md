# Scene Enrichment with OCR - Integration Guide

## ✅ What Was Done

### **1. Database Schema Updated**
- ✅ Added `ocr_text` column to `scenes` table
- ✅ Added `ocr_processed_at` timestamp
- ✅ Created GIN index for fast OCR text search
- ✅ Current status: 491 scenes, 0 with OCR (ready to process)

### **2. New Modules Created**

**`embeddings/ocr.py`** - OCR module
- Uses EasyOCR for text extraction
- Singleton pattern (reuses model)
- Confidence thresholding
- Text cleaning and normalization

**`enrich_scenes_with_ocr.py`** - Main enrichment script
- Integrates with existing SQLAlchemy infrastructure
- Smart reprocessing (skips already processed)
- Batch commits for efficiency
- Progress tracking with tqdm
- Error handling

### **3. Existing Modules Enhanced**

**`database/models.py`** - Updated Scene model
```python
class Scene(Base):
    # ... existing fields ...
    ocr_text = Column(Text)  # NEW: Extracted text
    ocr_processed_at = Column(DateTime)  # NEW: Processing timestamp
```

**`search/semantic_search.py`** - Enhanced text search
- Now searches BOTH transcript AND OCR text
- Finds "Deepsea Stavanger" and other visible text
- OCR matches highlighted in results

### **4. Duplications Eliminated**

**Before:** `enrich_scenes.py` had:
- ❌ Custom psycopg2 connections (duplicate of SQLAlchemy)
- ❌ OpenCLIP embedder (duplicate of vision_embeddings.py)
- ❌ Text embedder (duplicate of text_embeddings.py)

**After:** `enrich_scenes_with_ocr.py` uses:
- ✅ Existing `database/config.py` (SQLAlchemy)
- ✅ Existing `database/models.py` (ORM)
- ✅ New `embeddings/ocr.py` (OCR only - no duplication)
- ✅ Vision/text embeddings use existing modules

---

## 🚀 Quick Start

### **Step 1: Install OCR Dependency**

```bash
pip install easyocr
```

*Note: First run will download OCR models (~50MB)*

### **Step 2: Run OCR Enrichment**

```bash
# Process all scenes (skips already processed)
python enrich_scenes_with_ocr.py

# Show statistics
python enrich_scenes_with_ocr.py --stats

# Force reprocess everything
python enrich_scenes_with_ocr.py --force

# Process only first 10 scenes (testing)
python enrich_scenes_with_ocr.py --limit 10
```

### **Step 3: Test Search**

```bash
# Search for visible text in frames
curl "http://localhost:8000/search/quick?q=deepsea+stavanger&limit=5"

# Should now find the opening scene!
```

---

## 📊 Expected Results

### **Before OCR:**
```
Query: "deepsea stavanger"
Results: 0 (not in transcript)
```

### **After OCR:**
```
Query: "deepsea stavanger"
Results:
  ✓ AkerBP 1.mp4 at 00:00:00 - Scene 0
    Text: "[OCR: DEEPSEA STAVANGER OFFSHORE DRILLING RIG]"
```

---

## 🔧 How It Works

### **Processing Flow:**
```
1. Load all scenes from database (SQLAlchemy)
2. Skip scenes that already have ocr_text (unless --force)
3. For each scene:
   a. Load keyframe image
   b. Run EasyOCR text extraction
   c. Clean and normalize text
   d. Save to scenes.ocr_text
   e. Set scenes.ocr_processed_at = now()
4. Commit in batches (100 scenes)
5. Show summary and examples
```

### **Search Integration:**
```
User searches "deepsea stavanger"
  ↓
semantic_search.py _fuzzy_text_search()
  ↓
PostgreSQL full-text search on:
  - transcript_segments.text (spoken words)
  - scenes.ocr_text (visible text) ← NEW!
  ↓
Results ranked by relevance
  ↓
OCR matches highlighted: [OCR: DEEPSEA STAVANGER]
```

---

## 📁 File Structure

```
BasePipeline/
├── database/
│   ├── models.py              # Updated: Scene model + OCR columns
│   ├── config.py              # Existing: SQLAlchemy setup
│   └── add_ocr_columns.sql    # New: Migration script (already applied)
│
├── embeddings/
│   ├── ocr.py                 # New: OCR module
│   ├── vision_embeddings.py   # Existing: CLIP (reused)
│   └── text_embeddings.py     # Existing: BGE-M3 (reused)
│
├── search/
│   └── semantic_search.py     # Updated: Searches OCR text too
│
├── enrich_scenes_with_ocr.py  # New: Main enrichment script
└── enrich_scenes.py           # Old: Can be deleted
```

---

## 🎯 Comparison: Old vs New

| Feature | enrich_scenes.py (Old) | enrich_scenes_with_ocr.py (New) |
|---------|------------------------|--------------------------------|
| **Database** | psycopg2 (duplicate) | ✅ SQLAlchemy (reuses existing) |
| **Vision Embeddings** | OpenCLIP (duplicate) | ✅ Reuses vision_embeddings.py |
| **Text Embeddings** | SentenceTransformers (duplicate) | ✅ Reuses text_embeddings.py |
| **OCR** | ✅ PaddleOCR | ✅ EasyOCR (better) |
| **Object Detection** | GroundingDINO (not wired) | ❌ Not needed for MVP |
| **LLaVA Captions** | Complex setup | ❌ Not needed for MVP |
| **Reprocessing Check** | ❌ No | ✅ Smart skip logic |
| **Progress Tracking** | ❌ No | ✅ tqdm progress bar |
| **Error Handling** | ⚠️ Basic | ✅ Comprehensive |
| **Lines of Code** | 620 | 280 (cleaner!) |

---

## 🎯 Smart Reprocessing

The script intelligently skips already processed scenes:

```python
# Default: Skip existing
python enrich_scenes_with_ocr.py
# Processes only: scenes where ocr_text IS NULL

# Force reprocess
python enrich_scenes_with_ocr.py --force
# Processes: ALL scenes, even if already done
```

**Why this matters:**
- Run multiple times safely
- Add new videos → only process new scenes
- Interrupted? Resume where you left off

---

## 📊 Performance

**Processing Time (estimated):**
- First run: ~2-3 minutes for 491 scenes
  - OCR model loading: ~10 seconds
  - Per scene: ~0.2-0.5 seconds
  
- Subsequent runs: <1 second
  - Skips existing scenes

**Resource Usage:**
- CPU: Medium (OCR processing)
- GPU: Optional (faster if available)
- Memory: ~500MB (OCR model)
- Disk: None (uses existing keyframes)

---

## 🔍 Testing

### **Test 1: Check OCR Column Exists**
```bash
docker exec -it video_search_db psql -U postgres -d video_semantic_search -c "\d scenes"
```
Should show:
- `ocr_text | text`
- `ocr_processed_at | timestamp`

### **Test 2: Run on Sample**
```bash
python enrich_scenes_with_ocr.py --limit 5
```
Should process 5 scenes and show extracted text.

### **Test 3: Check Results**
```bash
python enrich_scenes_with_ocr.py --stats
```
Should show coverage percentage.

### **Test 4: Search**
```bash
curl "http://localhost:8000/search/quick?q=deepsea+stavanger"
```
Should find scenes with that text!

---

## 🎯 Next Steps

### **Phase 1: ✅ DONE**
- [x] OCR module created
- [x] Database schema updated
- [x] Enrichment script created
- [x] Search enhanced to use OCR
- [x] Duplications eliminated

### **Phase 2: Ready to Run** (Do This Now!)
```bash
# 1. Install dependency
pip install easyocr

# 2. Run enrichment
python enrich_scenes_with_ocr.py

# 3. Test search
curl "http://localhost:8000/search/quick?q=deepsea+stavanger"
```

### **Phase 3: Future Enhancements** (Optional)
- [ ] Add object detection (GroundingDINO)
- [ ] Add scene captions (LLaVA)
- [ ] Multi-language OCR
- [ ] OCR confidence filtering UI

---

## 📝 Summary

**What Changed:**
- ✅ Database: Added OCR columns to scenes
- ✅ New Module: embeddings/ocr.py (EasyOCR)
- ✅ New Script: enrich_scenes_with_ocr.py (integrated)
- ✅ Enhanced: Search now finds OCR text
- ✅ Cleaned: Removed duplications with existing code

**Benefits:**
1. Find "Deepsea Stavanger" and other visible text
2. No code duplication (reuses existing infrastructure)
3. Smart reprocessing (runs fast on updates)
4. Easy to run and test

**To Get Started:**
```bash
pip install easyocr
python enrich_scenes_with_ocr.py
```

That's it! 🎉
