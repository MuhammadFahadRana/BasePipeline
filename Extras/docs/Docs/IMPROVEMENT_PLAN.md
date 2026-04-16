# Video Understanding Plan

## Current System Analysis

### What Your BasePipeline Does NOW:

1. **Audio-Only Processing:**
   - Transcribes video audio using ASR models (Whisper, WhisperX, etc.)
   - Detects scene boundaries based on visual content changes
   - Generates text embeddings from transcripts (BAAI/bge-m3, 1024-dim)
   - Stores transcript segments with timestamps

2. **Search Mechanism:**
   - **Semantic Search**: Text embedding similarity (70% weight)
   - **Fuzzy Text Search**: PostgreSQL full-text search (30% weight)
   - Typo correction for queries
   - Returns top-k results based on **text-only** matching

### The Core Problem:

**Your system is "blind" to visual content!** 

It only understands:
- ✅ What was **said** in the video
- ✅ **When** scene changes occur (but not what's in the scenes)

It does NOT understand:
- ❌ What **objects/people** are visible
- ❌ What **actions** are happening
- ❌ **Spatial relationships** (e.g., "person on the left drilling")
- ❌ **Temporal dynamics** (e.g., "drilling speed increases")
- ❌ **Visual-textual alignment** (matching spoken words to visual content)

### Why You Get Irrelevant Results:

Example Query: "drilling techniques"
- Your system matches: Any video **mentioning** "drilling" in the transcript
- But ignores: Videos **showing** drilling without mentioning it
- Result: False positives (mentions drilling, doesn't show it) + False negatives (shows drilling, doesn't mention it)

---

## 🚀 Improvement Strategy: Multi-Modal Video Understanding

### Phase 1: Visual Understanding Foundation (RECOMMENDED START)

#### 1.1 Vision Embeddings with CLIP
**What**: Add visual understanding using OpenAI's CLIP or similar vision-language models

**Implementation**:
```python
# New file: embeddings/vision_embeddings.py

Model Options:
A. CLIP (OpenAI/HuggingFace)
   - Model: "openai/clip-vit-large-patch14" 
   - Embeddings: 768-dim
   - Strengths: Text-image alignment, zero-shot
   
B. SigLIP (Google)
   - Model: "google/siglip-so400m-patch14-384"
   - Embeddings: 1152-dim
   - Strengths: Better retrieval performance than CLIP

C. BLIP-2 (Salesforce)
   - Model: "Salesforce/blip2-opt-2.7b"
   - Strengths: Image captioning + retrieval
```

**Process**:
1. Extract keyframes from each scene (you already have scene detection!)
2. Generate vision embeddings for each keyframe
3. Store in new database table: `visual_embeddings`

**Database Schema Change**:
```sql
CREATE TABLE visual_embeddings (
    id SERIAL PRIMARY KEY,
    scene_id INTEGER REFERENCES scenes(id),
    keyframe_path TEXT,
    embedding VECTOR(768),  -- CLIP dimension
    embedding_model VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON visual_embeddings 
USING hnsw (embedding vector_cosine_ops);
```

---

#### 1.2 Image Captioning for Dense Retrieval
**What**: Generate textual descriptions of visual content to index alongside transcripts

**Models**:
- **BLIP-2**: "Salesforce/blip2-opt-2.7b"
- **LLaVA**: "llava-hf/llava-1.5-7b-hf" (better reasoning)
- **CogVLM**: "THUDM/cogvlm-chat-hf" (state-of-the-art)

**Process**:
1. For each keyframe, generate detailed caption
2. Store caption as additional searchable text
3. Generate text embeddings from captions
4. Merge with transcript-based search

**Benefits**:
- Bridges gap between visual content and text search
- Works with existing text search infrastructure
- Enables finding videos by visual content described in words

---

### Phase 2: Spatio-Temporal Video Understanding (ADVANCED)

#### 2.1 Video-Specific Embeddings
**What**: Use models designed specifically for video understanding (not just image models)

**Model Options**:

**A. TimeS former / VideoMAE**
- Architecture: Vision Transformer for videos
- Input: Sequences of frames (spatio-temporal)
- Embeddings: Capture motion and temporal dynamics
- Use case: Action recognition, activity understanding

```python
# Example: VideoMAE
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics"
)
# Extracts embeddings from 16-frame clips
```

**B. VideoCLIP / X-CLIP**
- Architecture: CLIP extended to video domain
- Input: Text + video clips
- Strengths: Text-video retrieval, temporal understanding

```python
# X-CLIP for video-text retrieval
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("MCG-NJU/videomae-base")
# Process video clips with temporal context
```

**C. LanguageBind (Recommended for Research)**
- Model: "LanguageBind/Video_V1.5_FT_Audio"
- Multi-modal: Video + Audio + Text
- State-of-the-art video-text retrieval

---

#### 2.2 Object Detection & Tracking
**What**: Identify and track objects/people across frames

**Models**:
- **YOLO v8/v9**: Real-time object detection (you already have yolov8n.pt!)
- **Grounding DINO**: Open-vocabulary object detection
- **SAM**: Segment Anything Model for precise segmentation

**Process**:
1. Run object detector on keyframes
2. Extract object bounding boxes + labels
3. Store detected objects with timestamps
4. Enable queries like: "show me videos with cranes"

**Database Schema**:
```sql
CREATE TABLE detected_objects (
    id SERIAL PRIMARY KEY,
    scene_id INTEGER REFERENCES scenes(id),
    object_class VARCHAR(100),  -- e.g., "crane", "drill", "person"
    confidence FLOAT,
    bbox JSON,  -- {x, y, width, height}
    timestamp_start FLOAT,
    timestamp_end FLOAT
);

CREATE INDEX ON detected_objects (object_class);
```

---

#### 2.3 Action/Activity Recognition
**What**: Understand what's happening in the video

**Models**:
- **SlowFast**: Two-pathway model for action recognition
- **VideoSwin**: Transformer-based activity recognition
- **I3D**: Inflated 3D ConvNets

**Example Use**:
- Query: "show me drilling operations"
- System recognizes: "drilling" action in frames 100-500
- Returns: Timestamp where drilling actually occurs

---

### Phase 3: Multi-Modal Fusion (MOST POWERFUL)

#### 3.1 Cross-Modal Retrieval Architecture

```python
# Multi-modal search scoring

def multi_modal_search(query, top_k=10):
    # 1. Text-based search (current system)
    text_scores = semantic_search_text(query)
    
    # 2. Vision-based search (new)
    vision_scores = semantic_search_vision(query)  # CLIP text-to-image
    
    # 3. Caption-based search (new)
    caption_scores = semantic_search_captions(query)
    
    # 4. Object-based filter (new)
    object_matches = filter_by_detected_objects(query)
    
    # 5. Weighted fusion
    final_scores = (
        0.3 * text_scores +        # Speech content
        0.4 * vision_scores +       # Visual similarity
        0.2 * caption_scores +      # Scene descriptions
        0.1 * object_matches        # Object presence
    )
    
    return top_k_results(final_scores)
```

---

## 📋 Implementation Roadmap

### **Immediate Actions (Week 1-2)**: Foundation

**Step 1: Add CLIP Vision Embeddings**
- [ ] Install: `pip install transformers pillow`
- [ ] Create `embeddings/vision_embeddings.py`
- [ ] Add `visual_embeddings` table to database
- [ ] Modify `basic_pipeline.py` to extract keyframe embeddings
- [ ] Update `database/ingest.py` to store vision embeddings

**Step 2: Implement Vision-Text Fusion Search**
- [ ] Modify `search/semantic_search.py` to include vision search
- [ ] Add combined scoring function
- [ ] Test with sample queries

**Expected Improvement**: 30-50% better relevance

---

### **Short-term (Week 3-4)**: Enhancement

**Step 3: Add Image Captioning**
- [ ] Install BLIP-2 or LLaVA
- [ ] Generate captions for all keyframes
- [ ] Index captions as additional searchable text
- [ ] Merge caption embeddings with transcript embeddings

**Step 4: Object Detection Integration**
- [ ] Use existing YOLO model for object detection
- [ ] Create `detected_objects` table
- [ ] Add object-based filtering to search
- [ ] Enable queries like "show videos with excavators"

**Expected Improvement**: 60-70% better relevance

---

### **Medium-term (Month 2)**: Advanced

**Step 5: Video-Specific Models**
- [ ] Evaluate VideoMAE / X-CLIP
- [ ] Extract spatio-temporal embeddings from video clips
- [ ] Fine-tune on your domain (drilling, oil & gas operations)
- [ ] Implement temporal action localization

**Step 6: Multi-Modal Evaluation**
- [ ] Create labeled test set with relevance judgments
- [ ] Measure Precision@K, Recall@K, NDCG
- [ ] A/B test different fusion weights
- [ ] Optimize for your specific use case

---

## 🔧 Recommended Architecture (Next Generation)

```
┌─────────────────────────────────────────────────────────┐
│                   VIDEO INPUT                           │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   AUDIO     │ │   VISUAL    │ │   METADATA  │
│ Transcribe  │ │  Keyframes  │ │Scene Detect │
│ (Whisper)   │ │  (Extract)  │ │(PySceneDet.)│
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │
       ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   TEXT      │ │   VISION    │ │   OBJECTS   │
│ Embeddings  │ │ Embeddings  │ │  Detection  │
│   (BGE)     │ │   (CLIP)    │ │   (YOLO)    │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │
       └───────────────┼───────────────┘
                       ▼
              ┌─────────────────┐
              │  MULTI-MODAL    │
              │    DATABASE     │
              │  PostgreSQL +   │
              │    pgvector     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  FUSION SEARCH  │
              │  Text + Vision  │
              │  + Objects      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   API RESULTS   │
              │ (Ranked by      │
              │  multi-modal    │
              │  relevance)     │
              └─────────────────┘
```

---

## 💡 Quick Win: Minimal Changes for Maximum Impact

If you want the **fastest improvement with minimal code changes**:

### Option A: CLIP-Enhanced Search (3-4 hours work)

1. **Add vision search alongside text search**
2. **Use existing keyframes** (already extracted!)
3. **Simple weighted fusion**: 50% text + 50% vision

**Code Outline**:
```python
# embeddings/vision_embeddings.py
from transformers import CLIPProcessor, CLIPModel

class VisionEmbeddingGenerator:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def encode_image(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model.get_image_features(**inputs)
        return image_features.detach().numpy()[0]
    
    def encode_text(self, text):
        inputs = self.processor(text=[text], return_tensors="pt")
        text_features = self.model.get_text_features(**inputs)
        return text_features.detach().numpy()[0]

# search/multi_modal_search.py
def search_multi_modal(query, top_k=10):
    # Get text-to-text matches (current system)
    text_results = search_transcripts(query)
    
    # Get text-to-image matches (new!)
    vision_embedding = vision_encoder.encode_text(query)
    vision_results = search_keyframes(vision_embedding)
    
    # Combine scores
    combined = merge_results(text_results, vision_results, 
                            text_weight=0.5, vision_weight=0.5)
    return combined[:top_k]
```

---

## 📊 Expected Improvements by Phase

| Phase | Capability | Relevance Gain | Implementation Time |
|-------|-----------|----------------|---------------------|
| Current | Text-only (transcripts) | Baseline | - |
| Phase 1.1 | + Vision embeddings (CLIP) | +40% | 1-2 weeks |
| Phase 1.2 | + Image captions | +20% | 1 week |
| Phase 2.1 | + Video models (temporal) | +15% | 2-3 weeks |
| Phase 2.2 | + Object detection | +10% | 1 week |
| Phase 3 | Full multi-modal fusion | +15% | Ongoing |
| **Total** | **Multi-modal system** | **~100% improvement** | **2-3 months** |

---

## 🎯 Recommendations

**Start with Phase 1.1 (CLIP Vision Embeddings)**

**Why**:
1. ✅ Biggest bang for buck (40% improvement)
2. ✅ You already have keyframes extracted
3. ✅ Minimal code changes
4. ✅ Works with existing infrastructure
5. ✅ Can incrementally add more features

**Next Steps**:
1. Implement CLIP vision embeddings
2. Modify database schema to add `visual_embeddings` table
3. Update ingestion pipeline to generate vision embeddings
4. Enhance search to use both text and vision
5. Test and measure improvement

---

Want TODOs:
1. **Implement Phase 1.1 now** (CLIP integration)?
2. **Create detailed code** for the multi-modal architecture?
3. **Set up evaluation metrics** to measure improvements?

Let me know which direction you'd like to go!
