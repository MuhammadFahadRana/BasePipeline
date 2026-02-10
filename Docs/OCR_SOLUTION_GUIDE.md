# Solution: Finding "Deepsea Stavanger" in Video Frames

## 🔍 **The Problem**

**Query:** "deepsea stavanger"  
**Where it appears:** Opening scene title card (visible text in frame)  
**Current result:** ❌ No results

**Why it's not found:**
1. ❌ Not in transcript (Whisper didn't hear it - it's written text, not spoken)
2. ❌ CLIP embeddings can't read text (CLIP recognizes objects/scenes, not text)
3. ✅ Keyframes exist (`processed/scenes/AkerBP 1/AkerBP 1_scene_0.jpg`)
4. ✅ Visual embeddings exist (491 in database)

**The gap:** You need **OCR (Optical Character Recognition)** to extract text from frames!

---

## 🎯 **Solution: Add OCR to Your Pipeline**

### **Architecture:**

```
Video Frame → CLIP (objects/scenes) + OCR (text in frame) → Searchable
```

**Current:**
```
"Deepsea Stavanger" title card
  ↓ CLIP embedding
  "offshore platform, ocean, industrial scene" ✓
  ↓ OCR
  (not extracted) ✗
```

**After fix:**
```
"Deepsea Stavanger" title card
  ↓ CLIP embedding
  "offshore platform, ocean, industrial scene" ✓
  ↓ OCR
  "Deepsea Stavanger" extracted! ✓
  → Now searchable!
```

---

## 🔧 **Implementation: Add OCR**

### **Step 1: Install OCR Library**

```bash
pip install easyocr
# or
pip install pytesseract pillow
```

### **Step 2: Create OCR Module**

Create `embeddings/frame_ocr.py`:

```python
"""Extract text from video frames using OCR."""
import easyocr
from pathlib import Path
from typing import Optional, List
import re

class FrameOCR:
    """Extract text from video keyframes."""
    
    def __init__(self, languages=['en']):
        """
        Initialize OCR reader.
        
        Args:
            languages: List of languages to detect (default: English)
        """
        print(f"Loading OCR model (languages: {languages})...")
        self.reader = easyocr.Reader(languages, gpu=True)  # Use GPU if available
        print("✓ OCR model loaded")
    
    def extract_text(self, image_path: str) -> str:
        """
        Extract all text from an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text as a single string
        """
        if not Path(image_path).exists():
            return ""
        
        try:
            # Perform OCR
            results = self.reader.readtext(image_path)
            
            # Extract text (results format: [[bbox, text, confidence], ...])
            texts = [text for (bbox, text, conf) in results if conf > 0.5]
            
            # Join and clean
            full_text = " ".join(texts)
            full_text = self._clean_text(full_text)
            
            return full_text
            
        except Exception as e:
            print(f"OCR failed for {image_path}: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that are noise
        text = re.sub(r'[^\w\s\-.,@#]', '', text)
        return text.strip()
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from OCR text."""
        # Simple keyword extraction (you can use more advanced NLP)
        words = text.split()
        # Keep words longer than 3 characters
        keywords = [w for w in words if len(w) > 3]
        return keywords


# Singleton instance
_ocr_instance = None

def get_ocr_reader():
    """Get or create OCR reader instance."""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = FrameOCR()
    return _ocr_instance
```

### **Step 3: Update Database Schema**

Add OCR text to scenes table:

```sql
-- Add column for OCR text
ALTER TABLE scenes ADD COLUMN IF NOT EXISTS ocr_text TEXT;

-- Create index for text search
CREATE INDEX IF NOT EXISTS idx_scenes_ocr_text 
ON scenes USING GIN(to_tsvector('english', COALESCE(ocr_text, '')));
```

Run it:
```bash
docker exec -i video_search_db psql -U postgres -d video_semantic_search << 'EOF'
ALTER TABLE scenes ADD COLUMN IF NOT EXISTS ocr_text TEXT;
CREATE INDEX IF NOT EXISTS idx_scenes_ocr_text 
ON scenes USING GIN(to_tsvector('english', COALESCE(ocr_text, '')));
EOF
```

### **Step 4: Update Scene Processing**

Modify your scene detector to extract OCR text. Create `process_ocr.py`:

```python
"""Process existing keyframes and extract OCR text."""
from pathlib import Path
from database.config import SessionLocal
from database.models import Scene
from embeddings.frame_ocr import get_ocr_reader

def process_scenes_ocr():
    """Extract OCR text from all existing keyframes."""
    db = SessionLocal()
    ocr = get_ocr_reader()
    
    # Get scenes without OCR text
    scenes = db.query(Scene).filter(
        (Scene.ocr_text == None) | (Scene.ocr_text == '')
    ).all()
    
    print(f"Processing {len(scenes)} scenes for OCR...")
    
    for i, scene in enumerate(scenes, 1):
        if not scene.keyframe_path or not Path(scene.keyframe_path).exists():
            continue
        
        print(f"[{i}/{len(scenes)}] Processing {scene.keyframe_path}")
        
        # Extract text from keyframe
        ocr_text = ocr.extract_text(scene.keyframe_path)
        
        if ocr_text:
            print(f"  Extracted: {ocr_text[:100]}...")
            scene.ocr_text = ocr_text
        
        # Commit every 10 scenes
        if i % 10 == 0:
            db.commit()
            print(f"  ✓ Committed {i} scenes")
    
    db.commit()
    print(f"\n✓ Processed {len(scenes)} scenes")
    db.close()

if __name__ == "__main__":
    process_scenes_ocr()
```

Run it:
```bash
python process_ocr.py
```

### **Step 5: Update Search to Include OCR Text**

Modify `search/semantic_search.py` to also search OCR text:

```python
# In _fuzzy_text_search method, add OCR search

def _fuzzy_text_search(
    self, query: str, top_k: int = 10, video_filter: Optional[str] = None
) -> List[SearchResult]:
    """
    Fuzzy text search using PostgreSQL full-text search.
    NOW INCLUDES OCR TEXT FROM KEYFRAMES!
    """
    
    # ... existing code ...
    
    # ADDITION: Also search OCR text in scenes
    ocr_query = text(f"""
        SELECT DISTINCT
            ts.id,
            ts.video_id,
            v.filename,
            v.file_path,
            ts.start_time,
            ts.end_time,
            ts.text,
            COALESCE(
                ts_rank(to_tsvector('english', ts.text), query),
                0
            ) + 
            COALESCE(
                ts_rank(to_tsvector('english', s.ocr_text), query) * 0.8,
                0
            ) AS rank
        FROM transcript_segments ts
        JOIN videos v ON ts.video_id = v.id
        LEFT JOIN scenes s ON ts.video_id = s.video_id 
            AND ts.start_time >= s.start_time 
            AND ts.start_time <= s.end_time
        WHERE 
            to_tsvector('english', ts.text) @@ query
            OR to_tsvector('english', COALESCE(s.ocr_text, '')) @@ query
            {query_filter}
        ORDER BY rank DESC
        LIMIT :top_k
    """)
    
    # Rest of the code...
```

---

## 🚀 **Quick Test**

After implementing OCR:

```python
from database.config import SessionLocal
from search.semantic_search import SemanticSearchEngine

db = SessionLocal()
search = SemanticSearchEngine(db)

# Should now find "Deepsea Stavanger"!
results = search.search("deepsea stavanger", top_k=5)

for r in results:
    print(f"{r.video_filename} at {r.timestamp}")
    print(f"  {r.text[:100]}...")
```

---

## 📊 **What Each Component Does**

| Component | What It Finds | Example |
|-----------|---------------|---------|
| **Whisper (Transcript)** | Spoken words | "We are drilling at 3000 meters" |
| **CLIP (Visual)** | Objects, scenes, activities | Drilling rig, ocean, machinery |
| **OCR (Frame Text)** | Text visible in frame | "Deepsea Stavanger", "Safety First" |
| **Combined** | Everything! | All of the above |

---

## 🎯 **For "Deepsea Stavanger" Specifically:**

**Opening scene likely has:**
- **Visual:** Ocean, platform, industrial scene → CLIP finds this ✓
- **Text:** "Deepsea Stavanger" title → OCR finds this ✓
- **Audio:** Intro music, maybe narration → Whisper transcribes ✓

**After OCR implementation:**
```
Query: "deepsea stavanger"
  → OCR index: "Deepsea Stavanger" in scene 0 
  → Result: AkerBP 1.mp4 at 00:00:00
  ✓ FOUND!
```

---

## ⚡ **Alternative: Quick Fix Without OCR**

If you know the scene number where "Deepsea Stavanger" appears:

**Option 1: Add manual keywords to database**
```sql
-- Add OCR text manually for specific scene
UPDATE scenes 
SET ocr_text = 'Deepsea Stavanger offshore drilling rig platform'
WHERE video_id = (SELECT id FROM videos WHERE filename = 'AkerBP 1.mp4')
AND scene_id = 0;
```

**Option 2: Add to transcript**
Add the title card text as a transcript segment at time 0:00.

---

## 🔮 **Future: Advanced Spatio-Temporal Understanding**

Beyond OCR and CLIP, you can add:

### **1. Object Detection**
Detect specific objects: "drilling bit", "hard hat", "fire extinguisher"

### **2. Action Recognition**  
Understand activities: "person drilling", "equipment being lifted", "alarm triggered"

### **3. Scene Understanding**
Context: "emergency situation", "routine operation", "training session"

### **4. Temporal Sequences**
Understand sequences: "person puts on helmet, then climbs ladder, then operates drill"

---

## 📝 **Summary**

**Your Current System:**
- ✅ Has keyframes (491 scenes)
- ✅ Has CLIP embeddings (for objects/scenes)
- ❌ No OCR (can't read text in frames)

**Why "Deepsea Stavanger" Not Found:**
- It's written text in the frame (title card)
- CLIP doesn't read text
- Whisper didn't hear it (not spoken)

**Solution:**
1. **Immediate:** Add OCR (`easyocr` or `pytesseract`)
2. **Run:** `process_ocr.py` to extract text from existing keyframes
3. **Update:** Search to include OCR text
4. **Result:** Can now search for text visible in videos!

**Benefits:**
- Find company names, equipment labels, location names
- Search for visible signage, warnings, titles
- Discover branded equipment, product names
- Identify text-based information in presentations

---

## 🎯 **Next Step**

Want me to:
1. ✅ Help you implement the OCR solution?
2. ✅ Create the complete `process_ocr.py` script?
3. ✅ Update your search to use OCR text?

Just let me know! 🚀
