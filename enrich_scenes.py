"""
enrich_scenes.py

Offline enrichment for scene keyframes/clips:
- OCR (background text)
- Open-vocabulary object detection (e.g., "cup")
- OpenCLIP vision embeddings (512-dim, matches your pgvector dimension)
- Optional: LLaVA-NeXT-Video captions (stored as text; can also embed into 1024-dim text embedding table)

Assumptions (adjust in CONFIG if needed):
- scenes table has: id, video_id, start_time, end_time, keyframe_path
- transcript_segments table has scene_id and text (optional; used to build richer "scene_text")
- visual_embeddings table stores 512-dim embedding per scene
- embeddings table stores 1024-dim text embedding per scene (optional)

Install (pick what you need):
  pip install psycopg2-binary pgvector numpy pillow tqdm
  pip install open-clip-torch torch torchvision  # for OpenCLIP
  pip install paddleocr                         # for OCR (optional)
  pip install groundingdino-py                  # not official; often you install from repo (see notes)
  pip install transformers accelerate           # for LLaVA captioning (optional)
"""

from __future__ import annotations

import os
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# --- DB ---
import psycopg2
import psycopg2.extras

# --- progress ---
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x  # fallback

# --- OpenCLIP ---
try:
    import torch
    import open_clip
except Exception:
    torch = None
    open_clip = None

# --- OCR ---
try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

# --- Text embeddings (1024-dim) ---
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# --- LLaVA-NeXT-Video (optional) ---
# NOTE: exact class names depend on the checkpoint + transformers version.
# We keep this behind a wrapper and fail gracefully if not available.
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
except Exception:
    AutoProcessor = None
    AutoModelForVision2Seq = None


@dataclass
class Config:
    # ---- DB ----
    pg_dsn: str = "postgresql://postgres:YOUR_PASSWORD@localhost:5432/YOUR_DB"

    # Tables
    scenes_table: str = "scenes"
    transcript_table: str = "transcript_segments"
    visual_embeddings_table: str = "visual_embeddings"
    text_embeddings_table: str = "embeddings"

    # Column names (adjust if your schema differs)
    scene_id_col: str = "id"
    scene_video_id_col: str = "video_id"
    scene_start_col: str = "start_time"
    scene_end_col: str = "end_time"
    scene_keyframe_col: str = "keyframe_path"

    # Enrichment columns to store back into scenes (created if missing)
    scene_ocr_col: str = "ocr_text"
    scene_objects_col: str = "object_tags"  # JSONB string
    scene_caption_col: str = "scene_caption"  # text

    # ---- Models ----
    # Vision embedding: OpenCLIP should output 512-dim for many ViT-B/32 variants.
    openclip_model: str = "ViT-B-32"
    openclip_pretrained: str = "laion2b_s34b_b79k"  # good general checkpoint
    device: str = "cuda"  # "cuda" or "cpu" (auto handled below)

    # OCR
    enable_ocr: bool = True
    ocr_lang: str = "en"  # change if needed
    ocr_use_gpu: bool = True

    # Objects (open-vocab)
    enable_objects: bool = True
    # We'll use a prompt list approach; detection backend is left pluggable.
    object_prompts: List[str] = None  # set in __post_init__

    # Text embeddings (1024-dim)
    enable_text_embedding: bool = True
    text_embed_model: str = "intfloat/e5-large-v2"  # outputs 1024-dim

    # LLaVA-NeXT-Video captions
    enable_llava_video_caption: bool = True
    # Make this configurable because checkpoint names vary across releases.
    llava_model_id: str = "YOUR_LLAVA_NEXT_VIDEO_CHECKPOINT"
    llava_max_new_tokens: int = 96

    # Frame sampling for video captioning
    sample_frames_for_caption: int = 8  # sample N frames between start/end
    max_frame_side: int = 720  # resize long side to keep VLM fast

    # Upserts
    overwrite_existing: bool = False

    def __post_init__(self):
        if self.object_prompts is None:
            # For your “cup” example, include synonyms.
            self.object_prompts = [
                "cup", "mug", "coffee cup", "water bottle", "microphone", "podium",
                "flag", "desk", "laptop", "whiteboard", "slide", "poster", "chart"
            ]


# --------------------------
# Utility: pgvector-friendly
# --------------------------
def np_to_pgvector_str(vec: np.ndarray) -> str:
    # pgvector accepts: '[0.1,0.2,...]'
    vec = vec.astype(np.float32).tolist()
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"


# --------------------------
# DB helpers
# --------------------------
class DB:
    def __init__(self, dsn: str):
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True

    def close(self):
        self.conn.close()

    def execute(self, sql: str, params: Optional[tuple] = None):
        with self.conn.cursor() as cur:
            cur.execute(sql, params)

    def fetchall(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())

    def ensure_scene_columns(self, cfg: Config):
        # Add columns if missing (safe for iteration).
        # If you don't want schema changes, remove this and store in separate tables instead.
        self.execute(f"""
        ALTER TABLE {cfg.scenes_table}
        ADD COLUMN IF NOT EXISTS {cfg.scene_ocr_col} TEXT,
        ADD COLUMN IF NOT EXISTS {cfg.scene_caption_col} TEXT,
        ADD COLUMN IF NOT EXISTS {cfg.scene_objects_col} JSONB;
        """)


# --------------------------
# OCR
# --------------------------
class OCRRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ocr = None

    def _ensure(self):
        if not self.cfg.enable_ocr:
            return
        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR not installed. Run: pip install paddleocr")
        if self.ocr is None:
            use_gpu = self.cfg.ocr_use_gpu and (torch is not None and torch.cuda.is_available())
            self.ocr = PaddleOCR(use_angle_cls=True, lang=self.cfg.ocr_lang, use_gpu=use_gpu)

    def run(self, image_path: str) -> str:
        if not self.cfg.enable_ocr:
            return ""
        if not image_path or not os.path.exists(image_path):
            return ""
        self._ensure()

        result = self.ocr.ocr(image_path, cls=True)
        # result structure: list of lines -> (box, (text, score))
        texts = []
        try:
            for line in result[0]:
                txt = line[1][0]
                if txt and txt.strip():
                    texts.append(txt.strip())
        except Exception:
            return ""
        return " ".join(texts)


# --------------------------
# OpenCLIP embeddings (512-dim)
# --------------------------
class OpenCLIPEmbedder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.preprocess = None
        self.device = "cpu"

    def _ensure(self):
        if open_clip is None or torch is None:
            raise RuntimeError("open-clip-torch/torch not installed. Run: pip install open-clip-torch torch torchvision")
        if self.cfg.device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        if self.model is None:
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.cfg.openclip_model, pretrained=self.cfg.openclip_pretrained
            )
            self.model = model.to(self.device).eval()
            self.preprocess = preprocess

    def embed_image(self, image_path: str) -> Optional[np.ndarray]:
        if not image_path or not os.path.exists(image_path):
            return None
        self._ensure()

        img = Image.open(image_path).convert("RGB")
        x = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = self.model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.squeeze(0).detach().cpu().numpy().astype(np.float32)


# --------------------------
# Objects (open-vocab) via a pluggable backend
# --------------------------
class ObjectTagger:
    """
    This is a “best effort” wrapper.

    For true open-vocab detection you want GroundingDINO.
    Its installation varies (often via git clone + pip install -e .),
    so we keep the interface stable and you can plug your implementation.

    For now, this tagger supports:
      - 'prompt list' -> produces tags with dummy confidences if backend missing
      - You can replace detect() with your GroundingDINO call.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Try to import a GroundingDINO-like backend if available.
        self.backend = None
        try:
            # Example placeholder; replace with your real GroundingDINO import.
            # from groundingdino.util.inference import load_model, predict
            self.backend = "TODO"
        except Exception:
            self.backend = None

    def detect(self, image_path: str) -> List[Dict[str, Any]]:
        if not self.cfg.enable_objects:
            return []

        if not image_path or not os.path.exists(image_path):
            return []

        # --- If you wire GroundingDINO here, return:
        # [{"label":"cup","score":0.82}, ...]
        #
        # For now, return empty to avoid misleading detections.
        return []


# --------------------------
# Text embeddings (1024-dim)
# --------------------------
class TextEmbedder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None

    def _ensure(self):
        if not self.cfg.enable_text_embedding:
            return
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")
        if self.model is None:
            self.model = SentenceTransformer(self.cfg.text_embed_model)

    def embed(self, text: str) -> Optional[np.ndarray]:
        if not self.cfg.enable_text_embedding:
            return None
        if not text or not text.strip():
            return None
        self._ensure()
        vec = self.model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float32)


# --------------------------
# LLaVA-NeXT-Video captioner (optional)
# --------------------------
class LLaVANextVideoCaptioner:
    """
    Intended usage:
      - sample N frames from [start_time, end_time]
      - pass frames to a video-VLM to produce a dense caption

    This is intentionally defensive because checkpoints/APIs vary.
    If transformers doesn't expose the correct model class for your checkpoint,
    you can still keep the pipeline and swap this class.

    If you tell me the exact HF model id you're using, I can lock this down
    to the correct AutoModel class & processor inputs.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.processor = None
        self.device = None

    def _ensure(self):
        if not self.cfg.enable_llava_video_caption:
            return
        if AutoProcessor is None or AutoModelForVision2Seq is None or torch is None:
            raise RuntimeError("transformers/torch not installed. Run: pip install transformers accelerate torch")

        self.device = "cuda" if (self.cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"

        if self.model is None:
            self.processor = AutoProcessor.from_pretrained(self.cfg.llava_model_id)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.cfg.llava_model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device == "cpu":
                self.model = self.model.to("cpu")

    def caption_from_frames(self, frames: List[Image.Image]) -> str:
        """
        Generic caption call.
        Your checkpoint may require different keys. This will need adjustment to the exact model.
        """
        self._ensure()
        if not frames:
            return ""

        prompt = "Describe the scene in detail, including objects and any visible text."
        # Many VLMs expect a single image; video variants often accept a list.
        inputs = self.processor(text=prompt, images=frames, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.cfg.llava_max_new_tokens)

        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        return text.strip()


# --------------------------
# Frame sampling from a video file (optional for video captioning)
# --------------------------
def sample_frames_from_video(video_path: str, start_s: float, end_s: float, n: int, max_side: int) -> List[Image.Image]:
    """
    Uses OpenCV to sample N frames between start and end.
    You can switch to ffmpeg extraction if you prefer.
    """
    import cv2

    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    start_frame = int(start_s * fps)
    end_frame = max(start_frame + 1, int(end_s * fps))
    total = end_frame - start_frame

    if total <= 1:
        indices = [start_frame]
    else:
        indices = [start_frame + int(i * total / n) for i in range(n)]

    frames: List[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # resize for speed
        w, h = img.size
        scale = max(w, h) / float(max_side)
        if scale > 1.0:
            img = img.resize((int(w / scale), int(h / scale)))

        frames.append(img)

    cap.release()
    return frames


# --------------------------
# Main enrichment
# --------------------------
def build_scene_text(transcript_text: str, ocr_text: str, objects: List[Dict[str, Any]], caption: str) -> str:
    obj_tokens = []
    for o in objects:
        lab = o.get("label")
        if lab:
            obj_tokens.append(lab)
    obj_str = ", ".join(sorted(set(obj_tokens)))

    parts = []
    if transcript_text:
        parts.append(f"TRANSCRIPT: {transcript_text}")
    if ocr_text:
        parts.append(f"OCR: {ocr_text}")
    if obj_str:
        parts.append(f"OBJECTS: {obj_str}")
    if caption:
        parts.append(f"CAPTION: {caption}")
    return "\n".join(parts)


def main():
    cfg = Config()

    db = DB(cfg.pg_dsn)
    db.ensure_scene_columns(cfg)

    ocr = OCRRunner(cfg)
    clip = OpenCLIPEmbedder(cfg)
    obj = ObjectTagger(cfg)
    txt = TextEmbedder(cfg)
    llava = LLaVANextVideoCaptioner(cfg)

    # Fetch scenes + transcript snippet (optional join)
    # Adjust join condition if your transcript_segments uses a different relation
    scenes = db.fetchall(f"""
        SELECT
            s.{cfg.scene_id_col} AS scene_id,
            s.{cfg.scene_video_id_col} AS video_id,
            s.{cfg.scene_start_col} AS start_time,
            s.{cfg.scene_end_col} AS end_time,
            s.{cfg.scene_keyframe_col} AS keyframe_path,
            COALESCE(string_agg(t.text, ' '), '') AS transcript_text,
            s.{cfg.scene_ocr_col} AS existing_ocr,
            s.{cfg.scene_caption_col} AS existing_caption,
            s.{cfg.scene_objects_col} AS existing_objects
        FROM {cfg.scenes_table} s
        LEFT JOIN {cfg.transcript_table} t
          ON t.scene_id = s.{cfg.scene_id_col}
        GROUP BY s.{cfg.scene_id_col}, s.{cfg.scene_video_id_col}, s.{cfg.scene_start_col}, s.{cfg.scene_end_col},
                 s.{cfg.scene_keyframe_col}, s.{cfg.scene_ocr_col}, s.{cfg.scene_caption_col}, s.{cfg.scene_objects_col}
        ORDER BY s.{cfg.scene_id_col} ASC;
    """)

    print(f"Found {len(scenes)} scenes to enrich")

    for row in tqdm(scenes, desc="Enriching scenes"):
        scene_id = row["scene_id"]
        keyframe_path = row.get("keyframe_path") or ""

        # Decide skip/update
        if not cfg.overwrite_existing:
            has_any = bool(row.get("existing_ocr") or row.get("existing_caption") or row.get("existing_objects"))
        else:
            has_any = False

        # Always compute vision embedding (safe to overwrite in embedding table)
        # OCR/objects/caption can respect overwrite flag.
        ocr_text = row.get("existing_ocr") or ""
        caption = row.get("existing_caption") or ""
        objects = row.get("existing_objects") or []

        if cfg.enable_ocr and (cfg.overwrite_existing or not ocr_text):
            try:
                ocr_text = ocr.run(keyframe_path)
            except Exception as e:
                print(f"[scene {scene_id}] OCR failed: {e}")
                ocr_text = ocr_text or ""

        if cfg.enable_objects and (cfg.overwrite_existing or not objects):
            try:
                objects = obj.detect(keyframe_path)
            except Exception as e:
                print(f"[scene {scene_id}] Objects failed: {e}")
                objects = objects or []

        # Optional: LLaVA-NeXT-Video captioning
        # Needs the video file path; you likely store it in videos table.
        if cfg.enable_llava_video_caption and (cfg.overwrite_existing or not caption):
            try:
                # Fetch video path from videos table
                vid = db.fetchall("SELECT path FROM videos WHERE id = %s;", (row["video_id"],))
                video_path = vid[0]["path"] if vid else None

                if video_path and os.path.exists(video_path):
                    frames = sample_frames_from_video(
                        video_path,
                        float(row["start_time"]),
                        float(row["end_time"]),
                        cfg.sample_frames_for_caption,
                        cfg.max_frame_side
                    )
                    if frames:
                        caption = llava.caption_from_frames(frames)
            except Exception as e:
                print(f"[scene {scene_id}] LLaVA caption failed: {e}")
                caption = caption or ""

        # Vision embedding (512-dim)
        vision_vec = None
        try:
            vision_vec = clip.embed_image(keyframe_path)
            if vision_vec is not None and vision_vec.shape[0] != 512:
                print(f"[scene {scene_id}] WARNING: vision dim {vision_vec.shape[0]} != 512 (check OpenCLIP model)")
        except Exception as e:
            print(f"[scene {scene_id}] OpenCLIP failed: {e}")
            vision_vec = None

        # Build a richer scene text and embed (1024-dim)
        scene_text = build_scene_text(
            transcript_text=(row.get("transcript_text") or "").strip(),
            ocr_text=(ocr_text or "").strip(),
            objects=objects if isinstance(objects, list) else [],
            caption=(caption or "").strip(),
        )

        text_vec = None
        if cfg.enable_text_embedding:
            try:
                text_vec = txt.embed(scene_text)
                if text_vec is not None and text_vec.shape[0] != 1024:
                    print(f"[scene {scene_id}] WARNING: text dim {text_vec.shape[0]} != 1024 (pick a 1024-dim model)")
            except Exception as e:
                print(f"[scene {scene_id}] Text embedding failed: {e}")
                text_vec = None

        # --- Write back into scenes table (ocr/objects/caption)
        try:
            db.execute(
                f"""
                UPDATE {cfg.scenes_table}
                SET {cfg.scene_ocr_col} = %s,
                    {cfg.scene_caption_col} = %s,
                    {cfg.scene_objects_col} = %s
                WHERE {cfg.scene_id_col} = %s;
                """,
                (ocr_text, caption, json.dumps(objects), scene_id),
            )
        except Exception as e:
            print(f"[scene {scene_id}] Update scenes failed: {e}")

        # --- Upsert into visual_embeddings (512-dim)
        if vision_vec is not None:
            try:
                db.execute(
                    f"""
                    INSERT INTO {cfg.visual_embeddings_table} (scene_id, embedding)
                    VALUES (%s, %s::vector)
                    ON CONFLICT (scene_id) DO UPDATE SET embedding = EXCLUDED.embedding;
                    """,
                    (scene_id, np_to_pgvector_str(vision_vec)),
                )
            except Exception as e:
                print(f"[scene {scene_id}] Upsert visual_embeddings failed: {e}")

        # --- Upsert into embeddings (1024-dim) (optional)
        if text_vec is not None:
            try:
                # If your embeddings table also stores a type/namespace, add it here (e.g., 'scene_text')
                db.execute(
                    f"""
                    INSERT INTO {cfg.text_embeddings_table} (scene_id, embedding, embedding_model)
                    VALUES (%s, %s::vector, %s)
                    ON CONFLICT (scene_id, embedding_model) WHERE segment_id IS NULL DO UPDATE SET embedding = EXCLUDED.embedding;
                    """,
                    (scene_id, np_to_pgvector_str(text_vec), cfg.text_embed_model),
                )
            except Exception as e:
                print(f"[scene {scene_id}] Upsert embeddings failed: {e}")

    db.close()
    print("Done.")


if __name__ == "__main__":
    main()