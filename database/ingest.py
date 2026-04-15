"""Ingest processed video data into the database."""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import re
import cv2
from typing import Dict, Optional, List, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text

from database.config import SessionLocal
from database.models import Video, Scene, TranscriptSegment, Embedding, VisualEmbedding
from embeddings.text_embeddings import get_embedding_generator
from embeddings.vision_embeddings import get_vision_embedding_generator


class DataIngester:
    """Ingest processed video data into database."""

    def __init__(self, db: Optional[Session] = None):
        """
        Initialize data ingester.

        Args:
            db: Database session (creates new if not provided)
        """
        self.db = db or SessionLocal()
        self.own_session = db is None
        self.embedding_gen = get_embedding_generator()
        self.vision_gen = None  # Lazy load
        self.visual_enricher = None  # Lazy load (Qwen2-VL captions/OCR)
        self.ocr_reader = None  # Lazy load (EasyOCR fallback)
        self.visual_enrichment_model = os.getenv(
            "VISUAL_ENRICHMENT_MODEL", "Qwen/Qwen2-VL-2B-Instruct"
        )
        self.visual_enrichment_load_in_4bit = (
            os.getenv("VISUAL_ENRICHMENT_LOAD_IN_4BIT", "false").strip().lower()
            in ("1", "true", "yes", "on")
        )
        self.visual_enrichment_enabled = (
            os.getenv("VISUAL_ENRICHMENT_ENABLED", "true").strip().lower()
            in ("1", "true", "yes", "on")
        )
        self._ensure_schema_extensions()

    def _get_vision_gen(self):
        if self.vision_gen is None:
            self.vision_gen = get_vision_embedding_generator()
        return self.vision_gen

    def _get_visual_enricher(self):
        """Lazy-load the Qwen2-VL enricher used for caption/OCR backfill."""
        if not self.visual_enrichment_enabled:
            return None
        if self.visual_enricher is not None:
            return self.visual_enricher

        try:
            from extract_visual_features import VisualFeatureExtractor
            from transcriber_utils import get_device

            device = get_device()
            load_in_4bit = self.visual_enrichment_load_in_4bit and device == "cuda"
            self.visual_enricher = VisualFeatureExtractor(
                model_name=self.visual_enrichment_model,
                device=device,
                load_in_4bit=load_in_4bit,
            )
        except Exception as exc:
            print(f"Warning: visual enricher unavailable: {exc}")
            self.visual_enricher = None
        return self.visual_enricher

    def _get_ocr_reader(self):
        """Lazy-load OCR fallback for scenes when model OCR is missing."""
        if self.ocr_reader is not None:
            return self.ocr_reader

        try:
            from embeddings.ocr import get_ocr_reader
            from transcriber_utils import get_device

            use_gpu = get_device() == "cuda"
            self.ocr_reader = get_ocr_reader(languages=["en", "no"], use_gpu=use_gpu)
        except Exception as exc:
            print(f"Warning: OCR fallback unavailable: {exc}")
            self.ocr_reader = None
        return self.ocr_reader

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.own_session:
            self.db.close()

    def _ensure_schema_extensions(self):
        """Best-effort schema evolution for multi-keyframe and OCR metadata."""
        ddl = [
            "ALTER TABLE scenes ADD COLUMN IF NOT EXISTS ocr_text_norm TEXT",
            "ALTER TABLE scenes ADD COLUMN IF NOT EXISTS ocr_confidence FLOAT",
            "ALTER TABLE scenes ADD COLUMN IF NOT EXISTS ocr_processed_at TIMESTAMP",
            "ALTER TABLE visual_embeddings ADD COLUMN IF NOT EXISTS sample_time FLOAT",
            "ALTER TABLE visual_embeddings ADD COLUMN IF NOT EXISTS frame_role VARCHAR(20)",
            "ALTER TABLE visual_embeddings ADD COLUMN IF NOT EXISTS frame_index INTEGER",
            "ALTER TABLE visual_embeddings ALTER COLUMN frame_role SET DEFAULT 'mid'",
            "ALTER TABLE visual_embeddings DROP CONSTRAINT IF EXISTS uq_scene_visual_embedding",
            "ALTER TABLE visual_embeddings ADD CONSTRAINT uq_scene_visual_embedding UNIQUE (scene_id, embedding_model, frame_role, sample_time)",
        ]
        for stmt in ddl:
            try:
                self.db.execute(text(stmt))
            except Exception:
                # Keep ingestion resilient on partially migrated environments.
                self.db.rollback()
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()

    @staticmethod
    def _normalize_ocr_text(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        txt = value.lower().strip()
        txt = re.sub(r"\s+", " ", txt)
        txt = re.sub(r"[^\w\s\-.,!?@#&()\[\]/]", "", txt)
        return txt or None

    @staticmethod
    def _clean_optional_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        txt = str(value).strip()
        if not txt:
            return None
        if txt.lower() in {"none", "null", "n/a", "na", "no text", "no visible text"}:
            return None
        return txt

    @staticmethod
    def _normalize_object_labels(value: Any) -> List[str]:
        invalid = {"none", "null", "n/a", "na", "no objects"}
        if value is None:
            return []
        if isinstance(value, list):
            labels = [str(v).strip() for v in value if str(v).strip()]
        elif isinstance(value, str):
            labels = [v.strip() for v in value.split(",") if v.strip()]
        else:
            labels = [str(value).strip()] if str(value).strip() else []
        labels = [lbl for lbl in labels if lbl.lower() not in invalid]
        # Keep order stable while de-duplicating.
        return list(dict.fromkeys(labels))

    @classmethod
    def _scene_has_enrichment_text(cls, scene: Scene) -> bool:
        caption = cls._clean_optional_text(scene.caption) or ""
        ocr = cls._clean_optional_text(scene.ocr_text) or ""
        labels = cls._normalize_object_labels(scene.object_labels)
        return bool(caption or ocr or labels)

    @classmethod
    def _compose_scene_text(
        cls,
        caption: Optional[str],
        object_labels: Any,
        ocr_text: Optional[str],
    ) -> Optional[str]:
        parts: List[str] = []
        if caption and str(caption).strip():
            clean_caption = cls._clean_optional_text(caption)
            if clean_caption:
                parts.append(clean_caption)
        labels = cls._normalize_object_labels(object_labels)
        if labels:
            parts.append(" ".join(labels))
        clean_ocr = cls._clean_optional_text(ocr_text)
        if clean_ocr:
            parts.append(clean_ocr)
        combined = " ".join(parts).strip()
        return combined or None

    @staticmethod
    def _scene_payload_map(scenes_data: List[Dict]) -> Dict[int, Dict]:
        payload: Dict[int, Dict] = {}
        for raw in scenes_data or []:
            scene_id = raw.get("scene_id")
            if scene_id is None:
                continue
            payload[int(scene_id)] = raw
        return payload

    def _resolve_keyframe_path_for_scene(
        self, scene: Scene, scene_payload: Optional[Dict] = None
    ) -> Optional[Path]:
        raw_candidates = [
            scene.keyframe_path,
            (scene_payload or {}).get("keyframe_path"),
        ]
        for raw in raw_candidates:
            if not raw:
                continue
            p = Path(raw)
            if p.exists():
                return p
            if not p.is_absolute():
                candidate = Path.cwd() / p
                if candidate.exists():
                    return candidate
        return None

    def _fallback_ocr_for_scene(
        self, scene: Scene, keyframe_path: Optional[Path]
    ) -> tuple[Optional[str], Optional[float]]:
        """Extract OCR text directly from keyframe when model OCR is empty."""
        reader = self._get_ocr_reader()
        if reader is None:
            return None, None

        frame_candidates: List[Path] = []
        if keyframe_path is not None and keyframe_path.exists():
            frame_candidates.append(keyframe_path)

        # OCR from start/end frames catches title cards that may not appear at mid-frame.
        video_path = self._resolve_video_path(scene)
        if video_path is not None:
            output_dir = (
                keyframe_path.parent
                if keyframe_path is not None
                and keyframe_path.parent.exists()
                else (Path("processed") / "scenes" / video_path.stem)
            )
            for role, sample_time in (
                ("start", float(scene.start_time or 0.0)),
                ("end", float(scene.end_time or scene.start_time or 0.0)),
            ):
                extracted = self._extract_frame_image(
                    video_path,
                    sample_time,
                    output_dir,
                    scene.scene_id,
                    role,
                )
                if extracted is None:
                    continue
                frame_path, _ = extracted
                if frame_path.exists():
                    frame_candidates.append(frame_path)

        # Stable de-dup preserving order.
        seen = set()
        unique_frames: List[Path] = []
        for frame in frame_candidates:
            key = str(frame.resolve()) if frame.exists() else str(frame)
            if key in seen:
                continue
            seen.add(key)
            unique_frames.append(frame)

        best_text = None
        best_conf = None
        for frame in unique_frames:
            try:
                detections = reader.extract_with_confidence(
                    str(frame), confidence_threshold=0.35
                )
            except Exception as exc:
                print(f"Warning: OCR fallback failed for {frame}: {exc}")
                continue

            if not detections:
                continue

            texts = []
            confidences = []
            for det in detections:
                txt = self._clean_optional_text(det.get("text"))
                if txt:
                    texts.append(txt)
                conf = det.get("confidence")
                if conf is not None:
                    try:
                        confidences.append(float(conf))
                    except (TypeError, ValueError):
                        pass

            merged = self._clean_optional_text(" ".join(texts))
            if not merged:
                continue

            mean_conf = (
                round(sum(confidences) / len(confidences), 4) if confidences else None
            )
            if best_text is None or len(merged) > len(best_text):
                best_text = merged
                best_conf = mean_conf

        return best_text, best_conf

    @classmethod
    def _fallback_caption_from_signals(
        cls, caption: Optional[str], object_labels: Any, ocr_text: Optional[str]
    ) -> Optional[str]:
        """
        Generate a minimal caption when the vision model doesn't return one.
        Uses available OCR/label signals only; avoids generic noise captions.
        """
        clean_caption = cls._clean_optional_text(caption)
        if clean_caption:
            return clean_caption

        labels = cls._normalize_object_labels(object_labels)
        if labels:
            return f"Scene with {', '.join(labels[:8])}."

        clean_ocr = cls._clean_optional_text(ocr_text)
        if clean_ocr:
            snippet = clean_ocr[:160].strip()
            return f"Scene showing text: {snippet}"

        return None

    def _fallback_caption_from_transcript(
        self, video_id: int, start_time: float, end_time: float
    ) -> Optional[str]:
        """
        Build a light caption from overlapping transcript snippets.
        Keeps no-transcript scenes separate while improving text-rich scenes.
        """
        rows = (
            self.db.query(TranscriptSegment.text)
            .filter(
                TranscriptSegment.video_id == video_id,
                TranscriptSegment.end_time >= start_time,
                TranscriptSegment.start_time <= end_time,
            )
            .order_by(TranscriptSegment.start_time.asc())
            .limit(2)
            .all()
        )
        snippets: List[str] = []
        for row in rows:
            txt = self._clean_optional_text(row[0] if row else None)
            if txt:
                snippets.append(txt)
        if not snippets:
            return None
        joined = " ".join(snippets)
        joined = re.sub(r"\s+", " ", joined).strip()
        if len(joined) > 180:
            joined = joined[:177].rstrip() + "..."
        return f"Scene discussing: {joined}"

    def _count_missing_transcript_embeddings(self, video_id: int) -> int:
        row = self.db.execute(
            text(
                """
                SELECT COUNT(*) AS missing_count
                FROM transcript_segments ts
                LEFT JOIN embeddings e ON e.segment_id = ts.id
                WHERE ts.video_id = :video_id
                  AND e.id IS NULL
                """
            ),
            {"video_id": video_id},
        ).first()
        return int(row[0] if row else 0)

    def _count_scenes_missing_enrichment(self, video_id: int) -> int:
        row = self.db.execute(
            text(
                """
                SELECT COUNT(*) AS missing_count
                FROM scenes s
                WHERE s.video_id = :video_id
                  AND s.keyframe_path IS NOT NULL
                  AND (
                        s.caption IS NULL
                        OR BTRIM(s.caption) = ''
                        OR LOWER(BTRIM(s.caption)) IN ('none', 'null', 'n/a', 'na')
                        OR (
                            (
                                s.ocr_text IS NULL
                                OR BTRIM(s.ocr_text) = ''
                                OR LOWER(BTRIM(s.ocr_text)) IN ('none', 'null', 'n/a', 'na')
                            )
                            AND s.ocr_processed_at IS NULL
                        )
                  )
                """
            ),
            {"video_id": video_id},
        ).first()
        return int(row[0] if row else 0)

    def _count_scenes_missing_text_embeddings(self, video_id: int) -> int:
        row = self.db.execute(
            text(
                """
                SELECT COUNT(*) AS missing_count
                FROM scenes s
                LEFT JOIN embeddings e
                    ON e.scene_id = s.id
                   AND e.segment_id IS NULL
                WHERE s.video_id = :video_id
                  AND (
                        (
                            s.caption IS NOT NULL
                            AND BTRIM(s.caption) <> ''
                            AND LOWER(BTRIM(s.caption)) NOT IN ('none', 'null', 'n/a', 'na')
                        )
                        OR (
                            s.ocr_text IS NOT NULL
                            AND BTRIM(s.ocr_text) <> ''
                            AND LOWER(BTRIM(s.ocr_text)) NOT IN ('none', 'null', 'n/a', 'na')
                        )
                        OR (s.object_labels IS NOT NULL AND s.object_labels::text <> '[]')
                  )
                  AND e.id IS NULL
                """
            ),
            {"video_id": video_id},
        ).first()
        return int(row[0] if row else 0)

    def _link_unmapped_segments(self, video_id: int) -> int:
        """
        Attach transcript segments without scene_id to the closest scene by time.
        Improves keyframe/context availability for transcript hits.
        """
        scenes = (
            self.db.query(Scene)
            .filter(Scene.video_id == video_id)
            .order_by(Scene.start_time.asc())
            .all()
        )
        if not scenes:
            return 0

        segments = (
            self.db.query(TranscriptSegment)
            .filter(
                TranscriptSegment.video_id == video_id,
                TranscriptSegment.scene_id.is_(None),
            )
            .all()
        )
        if not segments:
            return 0

        linked = 0
        for seg in segments:
            seg_start = float(seg.start_time or 0.0)
            seg_end = float(seg.end_time or seg_start)
            seg_mid = (seg_start + seg_end) / 2.0

            match = None
            for scene in scenes:
                start_t = float(scene.start_time or 0.0)
                end_t = float(scene.end_time or start_t)
                if start_t <= seg_mid <= end_t:
                    match = scene
                    break

            if match is None:
                # Choose nearest scene by midpoint distance.
                match = min(
                    scenes,
                    key=lambda s: abs(
                        ((float(s.start_time or 0.0) + float(s.end_time or 0.0)) / 2.0)
                        - seg_mid
                    ),
                )

            if match and match.id != seg.scene_id:
                seg.scene_id = match.id
                linked += 1

        if linked:
            self.db.flush()
        return linked

    def _scenes_missing_visual_coverage(self, video_id: int) -> List[Scene]:
        required_roles = {"start", "mid", "end"}
        scenes = (
            self.db.query(Scene)
            .filter(Scene.video_id == video_id, Scene.keyframe_path.isnot(None))
            .all()
        )
        if not scenes:
            return []

        missing: List[Scene] = []
        for scene in scenes:
            rows = (
                self.db.query(VisualEmbedding.frame_role)
                .filter(VisualEmbedding.scene_id == scene.id)
                .all()
            )
            existing_roles = {str(role[0]) for role in rows if role and role[0]}
            if not required_roles.issubset(existing_roles):
                missing.append(scene)
        return missing

    def _enrich_missing_scenes(
        self, video: Video, scenes_data: Optional[List[Dict]] = None
    ) -> Dict[str, int]:
        """
        Backfill missing caption/object-label/OCR fields for existing DB scenes.
        Uses values from results.json first, then model/OCR/transcript fallbacks.
        """
        stats = {
            "scenes_enriched": 0,
            "scenes_enriched_from_results": 0,
            "scenes_enriched_from_model": 0,
        }
        scene_payload_by_id = self._scene_payload_map(scenes_data or [])

        candidates = (
            self.db.query(Scene)
            .filter(Scene.video_id == video.id, Scene.keyframe_path.isnot(None))
            .all()
        )
        if not candidates:
            return stats

        enricher = None
        for scene in candidates:
            # Normalize pre-existing sentinel values (e.g., literal "None").
            existing_caption = self._clean_optional_text(scene.caption)
            existing_ocr = self._clean_optional_text(scene.ocr_text)
            existing_labels = self._normalize_object_labels(scene.object_labels)
            if scene.caption != existing_caption:
                scene.caption = existing_caption
            if scene.ocr_text != existing_ocr:
                scene.ocr_text = existing_ocr
                scene.ocr_text_norm = self._normalize_ocr_text(existing_ocr)
            if scene.object_labels != existing_labels:
                scene.object_labels = existing_labels

            caption = existing_caption
            ocr_text = existing_ocr
            object_labels = existing_labels

            need_caption = caption is None
            need_ocr = ocr_text is None
            need_labels = len(object_labels) == 0
            if not (need_caption or need_ocr or need_labels):
                continue

            payload = scene_payload_by_id.get(scene.scene_id, {})
            payload_caption = self._clean_optional_text(payload.get("caption"))
            payload_labels = self._normalize_object_labels(payload.get("object_labels"))
            payload_ocr = self._clean_optional_text(payload.get("ocr_text"))

            from_results = False
            from_model = False
            if need_caption and payload_caption:
                caption = payload_caption
                from_results = True
                need_caption = False
            if need_ocr and payload_ocr:
                ocr_text = payload_ocr
                from_results = True
                need_ocr = False
            if need_labels and payload_labels:
                object_labels = payload_labels
                from_results = True
                need_labels = False

            ocr_confidence = None

            keyframe_path = self._resolve_keyframe_path_for_scene(scene, payload)
            if keyframe_path is None:
                # Fallback: extract a mid-frame when cached keyframes are missing.
                video_path = self._resolve_video_path(scene)
                if video_path is not None:
                    mid_t = float(
                        (float(scene.start_time or 0.0) + float(scene.end_time or 0.0))
                        / 2.0
                    )
                    output_dir = Path("processed") / "scenes" / video_path.stem
                    extracted = self._extract_frame_image(
                        video_path,
                        mid_t,
                        output_dir,
                        scene.scene_id,
                        "mid",
                    )
                    if extracted is not None:
                        keyframe_path, _ = extracted
                        scene.keyframe_path = str(keyframe_path)

            ocr_attempted = False
            if keyframe_path is not None and (need_caption or need_ocr or need_labels):
                if need_ocr:
                    ocr_attempted = True
                if enricher is None:
                    enricher = self._get_visual_enricher()

                if enricher is not None:
                    try:
                        model_out = enricher.analyze_image(str(keyframe_path)) or {}
                    except Exception as exc:
                        print(
                            f"Warning: scene enrichment failed for scene {scene.id}: {exc}"
                        )
                        model_out = {}

                    model_caption = self._clean_optional_text(model_out.get("caption"))
                    model_labels = self._normalize_object_labels(
                        model_out.get("object_labels")
                    )
                    model_ocr = self._clean_optional_text(model_out.get("ocr_text"))

                    if need_caption and model_caption:
                        caption = model_caption
                        need_caption = False
                        from_model = True
                    if need_labels and model_labels:
                        object_labels = model_labels
                        need_labels = False
                        from_model = True
                    if need_ocr and model_ocr:
                        ocr_text = model_ocr
                        need_ocr = False
                        from_model = True

            # OCR fallback should run even when Qwen is unavailable.
            if need_ocr and keyframe_path is not None:
                ocr_attempted = True
                fallback_ocr, fallback_conf = self._fallback_ocr_for_scene(
                    scene, keyframe_path
                )
                if fallback_ocr:
                    ocr_text = fallback_ocr
                    ocr_confidence = fallback_conf
                    need_ocr = False

            if need_caption:
                caption = self._fallback_caption_from_signals(
                    caption, object_labels, ocr_text
                )
                need_caption = caption is None

            if need_caption:
                caption = self._fallback_caption_from_transcript(
                    video_id=video.id,
                    start_time=float(scene.start_time or 0.0),
                    end_time=float(scene.end_time or scene.start_time or 0.0),
                )
                need_caption = caption is None

            if need_caption and keyframe_path is not None:
                video_stub = Path(video.filename).stem.replace("_", " ")
                caption = (
                    f"Scene from {video_stub} between "
                    f"{float(scene.start_time or 0.0):.1f}s and "
                    f"{float(scene.end_time or scene.start_time or 0.0):.1f}s."
                )
                need_caption = False

            if not (
                self._clean_optional_text(caption)
                or self._clean_optional_text(ocr_text)
                or self._normalize_object_labels(object_labels)
            ):
                continue

            changed = False
            new_caption = self._clean_optional_text(caption)
            new_labels = self._normalize_object_labels(object_labels)
            new_ocr = self._clean_optional_text(ocr_text)
            new_ocr_norm = self._normalize_ocr_text(new_ocr)
            ocr_text_changed = scene.ocr_text != new_ocr

            if scene.caption != new_caption:
                scene.caption = new_caption
                changed = True
            if scene.object_labels != new_labels:
                scene.object_labels = new_labels
                changed = True
            if scene.ocr_text != new_ocr:
                scene.ocr_text = new_ocr
                changed = True
            if scene.ocr_text_norm != new_ocr_norm:
                scene.ocr_text_norm = new_ocr_norm
                changed = True

            if scene.ocr_text:
                if ocr_confidence is not None:
                    if scene.ocr_confidence != ocr_confidence:
                        scene.ocr_confidence = ocr_confidence
                        changed = True
                elif scene.ocr_confidence is None:
                    scene.ocr_confidence = 0.6
                    changed = True
                if ocr_text_changed or scene.ocr_processed_at is None:
                    scene.ocr_processed_at = datetime.utcnow()
                    changed = True
            elif ocr_attempted and scene.ocr_processed_at is None:
                scene.ocr_processed_at = datetime.utcnow()
                changed = True

            if not changed:
                continue

            stats["scenes_enriched"] += 1
            if from_results:
                stats["scenes_enriched_from_results"] += 1
            elif from_model:
                stats["scenes_enriched_from_model"] += 1

        if stats["scenes_enriched"] > 0:
            self.db.flush()
        return stats

    def _ensure_scene_text_embeddings(self, video_id: int) -> int:
        scenes = self.db.query(Scene).filter(Scene.video_id == video_id).all()
        targets: List[Scene] = []
        texts: List[str] = []

        for scene in scenes:
            text_to_embed = self._compose_scene_text(
                scene.caption, scene.object_labels, scene.ocr_text
            )
            if not text_to_embed:
                continue

            existing = (
                self.db.query(Embedding)
                .filter(Embedding.scene_id == scene.id, Embedding.segment_id.is_(None))
                .first()
            )
            if existing:
                continue

            targets.append(scene)
            texts.append(text_to_embed)

        if not targets:
            return 0

        vectors = self.embedding_gen.encode(texts, batch_size=16, show_progress=False)
        for scene, vec in zip(targets, vectors):
            emb = Embedding(
                scene_id=scene.id,
                segment_id=None,
                embedding=vec.tolist(),
                embedding_model=self.embedding_gen.model_name,
            )
            self.db.add(emb)

        self.db.flush()
        return len(targets)

    def ingest_video(
        self,
        results_file: Path,
        generate_embeddings: bool = True,
        generate_visual_embeddings: bool = True,
        skip_existing: bool = True,
        update_existing: bool = False,
    ) -> Dict:
        """
        Ingest a single video's results into database.

        Args:
            results_file: Path to results.json (e.g., processed/Whisper-Large-v3/AkerBP_1/results.json)
            generate_embeddings: Whether to generate embeddings for transcript segments
            generate_visual_embeddings: Whether to generate visual embeddings for keyframes
            skip_existing: Skip if video already exists in database
            update_existing: If True, checks if file is newer than DB record and updates if so.

        Returns:
            Dict with ingestion statistics
        """
        results_file = Path(results_file)

        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")

        # Load results
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        video_info = results["video"]
        video_name = video_info["filename"]

        # Simple fingerprint, using mtime_iso as a proxy for now
        video_fingerprint_val = video_info.get("mtime_iso")

        # Check existing
        existing_video = (
            self.db.query(Video).filter(Video.filename == video_name).first()
        )

        if existing_video:
            # ── Compare video fingerprints (content-based, not filesystem mtime) ──
            should_update = False
            reason = "no_changes"

            if update_existing:
                new_fingerprint = video_info.get("mtime_iso")
                db_fingerprint = existing_video.video_fingerprint

                if new_fingerprint != db_fingerprint:
                    # Video file itself has changed -> full re-ingest
                    should_update = True
                    reason = "video_changed"
                    print(
                        f"Video content changed for {video_name} "
                        f"(DB: {db_fingerprint} -> New: {new_fingerprint})"
                    )
                else:
                    # Fingerprint matches. Still repair missing enrichment/embeddings.
                    missing_text_emb_count = (
                        self._count_missing_transcript_embeddings(existing_video.id)
                        if generate_embeddings
                        else 0
                    )
                    missing_visual_scenes = (
                        self._scenes_missing_visual_coverage(existing_video.id)
                        if generate_visual_embeddings
                        else []
                    )
                    missing_enrichment_count = self._count_scenes_missing_enrichment(
                        existing_video.id
                    )
                    missing_scene_emb_count = self._count_scenes_missing_text_embeddings(
                        existing_video.id
                    )

                    need_text = missing_text_emb_count > 0
                    need_visual = len(missing_visual_scenes) > 0
                    need_scene_enrichment = missing_enrichment_count > 0
                    need_scene_text_embeddings = missing_scene_emb_count > 0

                    if (
                        need_text
                        or need_visual
                        or need_scene_enrichment
                        or need_scene_text_embeddings
                    ):
                        print(
                            "Repairing existing video data for "
                            f"{video_name} (missing_text={missing_text_emb_count}, "
                            f"missing_visual_scenes={len(missing_visual_scenes)}, "
                            f"missing_enrichment={missing_enrichment_count}, "
                            f"missing_scene_embeddings={missing_scene_emb_count})"
                        )
                        filled = self._fill_missing_embeddings(
                            existing_video,
                            need_text=need_text,
                            need_visual=need_visual,
                            need_scene_enrichment=need_scene_enrichment,
                            need_scene_text_embeddings=need_scene_text_embeddings,
                            scenes_data=results.get("scene_analysis", {}).get(
                                "scenes", []
                            ),
                            precomputed_missing_visual_scenes=missing_visual_scenes,
                        )
                        return {
                            "video": video_name,
                            "status": "filled_embeddings",
                            "video_id": existing_video.id,
                            **filled,
                        }

                    reason = "up_to_date"

            if should_update:
                print(f"Updating existing video: {video_name} (Replacing record)")
                self.db.delete(existing_video)
                self.db.commit()  # Commit deletion to ensure clean slate
            elif skip_existing:
                if update_existing and reason == "up_to_date":
                    print(f"  [OK] Video up to date: {video_name}")
                else:
                    print(f"  [OK] Video already in database: {video_name}")

                return {
                    "video": video_name,
                    "status": "skipped",
                    "video_id": existing_video.id,
                    "reason": reason,
                }
            else:
                # Not updating and not skipping explicitly -> Skip
                print(f"  [OK] Video already in database: {video_name} (Skipping)")
                return {
                    "video": video_name,
                    "status": "skipped",
                    "video_id": existing_video.id,
                    "reason": "already_exists",
                }

        print(f"\n{'=' * 60}")
        print(f"Ingesting: {video_name}")
        print(f"{'=' * 60}")

        # Create video record
        video = Video(
            filename=video_name,
            file_path=video_info["path"],
            file_size_mb=video_info.get("size_mb"),
            duration_seconds=results["scene_analysis"].get("total_duration"),
            whisper_model=results["processing_info"].get("whisper_model"),
            scene_threshold=results["processing_info"].get("scene_threshold"),
            video_fingerprint=video_fingerprint_val,
            processed_at=datetime.fromisoformat(video_info.get("mtime_iso"))
            if video_info.get("mtime_iso")
            else None,
        )

        self.db.add(video)
        self.db.flush()  # Get video.id

        print(f"Video record created (ID: {video.id})")

        # Ingest scenes
        scenes_data = results["scene_analysis"]["scenes"]
        scene_db_objects = []

        for scene_data in scenes_data:
            raw_ocr = scene_data.get("ocr_text")
            raw_caption = scene_data.get("caption")
            raw_labels = self._normalize_object_labels(scene_data.get("object_labels"))
            scene = Scene(
                video_id=video.id,
                scene_id=scene_data["scene_id"],
                start_time=scene_data["start_time"],
                end_time=scene_data["end_time"],
                duration=scene_data["duration"],
                start_frame=scene_data.get("start_frame"),
                end_frame=scene_data.get("end_frame"),
                keyframe_path=scene_data.get("keyframe_path"),
                ocr_text=self._clean_optional_text(raw_ocr),
                ocr_text_norm=self._normalize_ocr_text(self._clean_optional_text(raw_ocr)),
                ocr_confidence=scene_data.get("ocr_confidence"),
                object_labels=raw_labels,
                caption=self._clean_optional_text(raw_caption),
            )
            self.db.add(scene)
            scene_db_objects.append(scene)

        self.db.flush()
        print(f"{len(scenes_data)} scenes ingested")

        # Ingest transcript segments
        segments_data = results["transcription"]["segments"]
        transcript_segments = []

        for idx, seg_data in enumerate(segments_data):
            # Find corresponding scene (if any)
            scene_db_id = None
            for s_db in scene_db_objects:
                if (
                    seg_data["start"] >= s_db.start_time
                    and seg_data["start"] <= s_db.end_time
                ):
                    scene_db_id = s_db.id
                    break

            segment = TranscriptSegment(
                video_id=video.id,
                scene_id=scene_db_id,
                segment_index=idx,
                start_time=seg_data["start"],
                end_time=seg_data["end"],
                text=seg_data["text"],
                language=results["transcription"].get("language", "en"),
            )

            self.db.add(segment)
            transcript_segments.append(segment)

        self.db.flush()
        print(f"{len(transcript_segments)} transcript segments ingested")

        relinked_segments = self._link_unmapped_segments(video.id)
        if relinked_segments:
            print(
                f"[OK] Relinked {relinked_segments} transcript segments to nearest scenes"
            )

        # Text Embeddings
        if generate_embeddings and transcript_segments:
            print("Generating text embeddings (batch mode)...")
            texts = [seg.text for seg in transcript_segments]
            embeddings = self.embedding_gen.encode(
                texts,
                batch_size=32,
                show_progress=True,
            )

            for segment, embedding in zip(transcript_segments, embeddings):
                emb = Embedding(
                    segment_id=segment.id,
                    embedding=embedding.tolist(),
                    embedding_model=self.embedding_gen.model_name,
                )
                self.db.add(emb)

            print(f"[OK] {len(embeddings)} text embeddings generated")

        # Visual Embeddings
        visual_count = 0
        if generate_visual_embeddings:
            visual_count = self.ingest_visual_embeddings(scene_db_objects)

        # Scene text embeddings (caption + object labels + OCR).
        scene_text_embs_added = self._ensure_scene_text_embeddings(video.id)
        if scene_text_embs_added:
            print(f"[OK] {scene_text_embs_added} scene text embeddings generated")

        # Commit all changes
        self.db.commit()
        print(f"Successfully ingested: {video_name}\n")

        return {
            "video": video_name,
            "status": "success",
            "video_id": video.id,
            "scenes_count": len(scenes_data),
            "segments_count": len(transcript_segments),
            "text_embeddings": generate_embeddings,
            "visual_embeddings": visual_count,
        }

    def _fill_missing_embeddings(
        self,
        video: Video,
        need_text: bool,
        need_visual: bool,
        need_scene_enrichment: bool = False,
        need_scene_text_embeddings: bool = False,
        scenes_data: Optional[List[Dict]] = None,
        precomputed_missing_visual_scenes: Optional[List[Scene]] = None,
    ) -> Dict:
        """
        Repair missing data for an existing video without deleting records.
        """
        result = {
            "text_embeddings_added": 0,
            "visual_embeddings_added": 0,
            "scene_text_embeddings_added": 0,
            "segments_relinked": 0,
            "scenes_enriched": 0,
            "scenes_enriched_from_results": 0,
            "scenes_enriched_from_model": 0,
        }

        relinked = self._link_unmapped_segments(video.id)
        if relinked:
            result["segments_relinked"] = relinked
            print(f"  [OK] Relinked {relinked} transcript segments to scenes")

        if need_scene_enrichment:
            enrich_stats = self._enrich_missing_scenes(video, scenes_data=scenes_data)
            result.update(enrich_stats)
            if enrich_stats.get("scenes_enriched", 0):
                print(
                    "  [OK] Scene enrichment backfilled "
                    f"({enrich_stats['scenes_enriched']} scenes, "
                    f"results={enrich_stats['scenes_enriched_from_results']}, "
                    f"model={enrich_stats['scenes_enriched_from_model']})"
                )

        if need_text:
            segments = (
                self.db.query(TranscriptSegment)
                .filter(TranscriptSegment.video_id == video.id)
                .all()
            )
            segments_without_emb = [
                seg
                for seg in segments
                if not self.db.query(Embedding)
                .filter(Embedding.segment_id == seg.id)
                .first()
            ]

            if segments_without_emb:
                print(
                    f"  Generating text embeddings for {len(segments_without_emb)} segments..."
                )
                texts = [seg.text for seg in segments_without_emb]
                embeddings = self.embedding_gen.encode(
                    texts, batch_size=32, show_progress=True
                )
                for seg, emb_vec in zip(segments_without_emb, embeddings):
                    emb = Embedding(
                        segment_id=seg.id,
                        embedding=emb_vec.tolist(),
                        embedding_model=self.embedding_gen.model_name,
                    )
                    self.db.add(emb)
                result["text_embeddings_added"] = len(segments_without_emb)
                print(f"  [OK] {len(segments_without_emb)} text embeddings generated")

        if need_scene_text_embeddings or result["scenes_enriched"] > 0:
            scene_text_added = self._ensure_scene_text_embeddings(video.id)
            if scene_text_added:
                result["scene_text_embeddings_added"] = scene_text_added
                print(f"  [OK] {scene_text_added} scene text embeddings generated")

        if need_visual:
            scenes_without_emb = (
                precomputed_missing_visual_scenes
                if precomputed_missing_visual_scenes is not None
                else self._scenes_missing_visual_coverage(video.id)
            )
            if scenes_without_emb:
                count = self.ingest_visual_embeddings(scenes_without_emb)
                result["visual_embeddings_added"] = count

        self.db.commit()
        print(f"  [OK] Missing data repaired for {video.filename}")
        return result

    def _resolve_video_path(self, scene: Scene) -> Optional[Path]:
        video = self.db.query(Video).filter(Video.id == scene.video_id).first()
        if not video or not video.file_path:
            return None
        p = Path(video.file_path)
        if p.exists():
            return p
        candidate = Path.cwd() / video.file_path
        return candidate if candidate.exists() else None

    @staticmethod
    def _scene_sample_specs(scene: Scene) -> List[tuple]:
        """Return (role, sample_time) targets for a scene."""
        start_t = float(scene.start_time or 0.0)
        end_t = float(scene.end_time or start_t)
        if end_t < start_t:
            end_t = start_t
        duration = max(0.0, end_t - start_t)
        mid_t = (start_t + end_t) / 2.0

        specs = [("start", start_t), ("mid", mid_t), ("end", end_t)]

        # Add extra temporal coverage on long scenes.
        if duration > 20.0:
            extra_count = min(3, int(duration // 20.0))
            for i in range(extra_count):
                frac = (i + 1) / (extra_count + 1)
                specs.append((f"extra_{i + 1}", start_t + (duration * frac)))
        return specs

    def _extract_frame_image(
        self,
        video_path: Path,
        sample_time: float,
        output_dir: Path,
        scene_id: int,
        role: str,
    ) -> Optional[tuple]:
        """Extract or reuse a sampled frame image; returns (path, frame_index)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"scene_{scene_id}_{role}.jpg"
        if image_path.exists():
            return image_path, None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_index = max(0, int(sample_time * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            cv2.imwrite(str(image_path), frame)
            return image_path, frame_index
        finally:
            cap.release()

    def ingest_visual_embeddings(
        self, scenes: List[Scene], batch_size: int = 32
    ) -> int:
        """Generate and store multi-keyframe visual embeddings for scenes."""
        if not scenes:
            return 0

        print(f"Generating visual embeddings for {len(scenes)} scenes (multi-keyframe)...")
        vision_gen = self._get_vision_gen()

        embeddings_created = 0
        samples = []

        for scene in scenes:
            video_path = self._resolve_video_path(scene)
            if video_path is None:
                continue

            keyframe_base = Path(scene.keyframe_path) if scene.keyframe_path else None
            if keyframe_base is not None and not keyframe_base.is_absolute():
                keyframe_base = Path.cwd() / keyframe_base

            output_dir = (
                keyframe_base.parent
                if keyframe_base is not None and keyframe_base.parent.exists()
                else (Path("processed") / "scenes" / video_path.stem)
            )

            for role, sample_time in self._scene_sample_specs(scene):
                if role == "mid" and keyframe_base is not None and keyframe_base.exists():
                    sample_path = keyframe_base
                    frame_index = None
                else:
                    extracted = self._extract_frame_image(
                        video_path,
                        sample_time,
                        output_dir,
                        scene.scene_id,
                        role,
                    )
                    if extracted is None:
                        continue
                    sample_path, frame_index = extracted

                exists = (
                    self.db.query(VisualEmbedding)
                    .filter(
                        VisualEmbedding.scene_id == scene.id,
                        VisualEmbedding.embedding_model == vision_gen.model_name,
                        VisualEmbedding.frame_role == role,
                        VisualEmbedding.sample_time == float(sample_time),
                    )
                    .first()
                )
                if exists:
                    continue
                samples.append((scene, sample_path, role, float(sample_time), frame_index))

        if not samples:
            print("[OK] Visual embeddings already up to date")
            return 0

        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            batch_paths = []
            valid_samples = []

            for scene, sample_path, role, sample_time, frame_index in batch:
                if sample_path.exists():
                    batch_paths.append(str(sample_path))
                    valid_samples.append((scene, sample_path, role, sample_time, frame_index))
                else:
                    print(f"Warning: Keyframe not found: {sample_path}")

            if not batch_paths:
                continue

            try:
                embeddings = vision_gen.encode_images(
                    batch_paths,
                    batch_size=len(batch_paths),
                    show_progress=False,
                    normalize=True,
                )

                for (scene, sample_path, role, sample_time, frame_index), embedding in zip(valid_samples, embeddings):
                    visual_emb = VisualEmbedding(
                        scene_id=scene.id,
                        keyframe_path=str(sample_path),
                        sample_time=sample_time,
                        frame_role=role,
                        frame_index=frame_index,
                        embedding=embedding.tolist(),
                        embedding_model=vision_gen.model_name,
                    )
                    self.db.add(visual_emb)
                    embeddings_created += 1

                # Flush batches to database
                self.db.flush()

            except Exception as e:
                print(f"Error processing visual batch: {e}")
                continue

        print(f"[OK] {embeddings_created} visual embeddings generated")
        return embeddings_created

    def ingest_batch(
        self,
        processed_dir: str = "processed",
        generate_embeddings: bool = True,
        skip_existing: bool = True,
        update_existing: bool = True,  # Default to True to handle updates
        force: bool = False,
    ) -> Dict:
        """
        Batch ingest all processed videos.
        """
        processed_path = Path(processed_dir)

        # Prefer canonical layout: processed/results/<VideoName>/results.json
        # Fall back to legacy layout only when canonical results/ is absent.
        canonical_results_dir = processed_path / "results"
        if canonical_results_dir.exists():
            results_files = sorted(canonical_results_dir.glob("*/results.json"))
            source_hint = str(canonical_results_dir)
        else:
            results_files = sorted(processed_path.glob("*/*/results.json"))
            source_hint = f"{processed_path} (legacy pattern */*/results.json)"

        if not results_files:
            raise FileNotFoundError(
                f"No results.json found under canonical '{canonical_results_dir}' "
                f"or legacy layout in '{processed_path}'"
            )

        print(f"\n{'=' * 60}")
        print(f"BATCH INGESTION")
        print(f"{'=' * 60}")
        print(f"Found {len(results_files)} videos to ingest")
        print(f"Source: {source_hint}")
        print(f"{'=' * 60}\n")

        stats = {
            "total": len(results_files),
            "success": 0,
            "skipped": 0,
            "failed": 0,
            "failed_videos": [],
        }

        for i, results_file in enumerate(results_files, 1):
            print(f"[{i}/{len(results_files)}]")
            try:
                result = self.ingest_video(
                    results_file,
                    generate_embeddings=generate_embeddings,
                    skip_existing=not force,
                    update_existing=update_existing or force,
                )

                if result["status"] in ("success", "filled_embeddings"):
                    stats["success"] += 1
                elif result["status"] == "skipped":
                    stats["skipped"] += 1
                else:
                    stats["failed"] += 1
                    stats["failed_videos"].append(result["video"])

            except Exception as e:
                self.db.rollback()
                print(f"Failed to ingest {results_file.parent.name}: {e}")
                import traceback

                traceback.print_exc()
                stats["failed"] += 1
                stats["failed_videos"].append(
                    {"video": results_file.parent.name, "error": str(e)}
                )

        print("\n" + "=" * 60)
        print("BATCH INGESTION COMPLETE")
        print("=" * 60)
        print(f"Total: {stats['total']}")
        print(f"Success: {stats['success']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"Failed: {stats['failed']}")
        if stats["failed_videos"]:
            print("\nFailed videos:")
            for fail in stats["failed_videos"]:
                # The structure of failed_videos might be a dict or just the name,
                # depending on where the error occurred.
                if isinstance(fail, dict):
                    print(f"  - {fail['video']}: {fail.get('error', 'Unknown error')}")
                else:
                    print(f"  - {fail}: Unknown error")

        return stats

    def verify(self) -> Dict:
        """Print a summary of database contents and flag any gaps."""
        total_videos = self.db.query(Video).count()
        total_scenes = self.db.query(Scene).count()
        total_segs = self.db.query(TranscriptSegment).count()
        total_text_emb = self.db.query(Embedding).count()
        total_vis_emb = self.db.query(VisualEmbedding).count()

        print(f"\n{'=' * 60}")
        print("DATABASE VERIFICATION")
        print(f"{'=' * 60}")
        print(f"Videos:              {total_videos}")
        print(f"Scenes:              {total_scenes}")
        print(f"Transcript segments: {total_segs}")
        print(f"Text embeddings:     {total_text_emb}")
        print(f"Visual embeddings:   {total_vis_emb}")

        issues = []
        videos = self.db.query(Video).all()
        for v in videos:
            scene_count = self.db.query(Scene).filter(Scene.video_id == v.id).count()
            vis_emb = (
                self.db.query(VisualEmbedding)
                .join(Scene)
                .filter(Scene.video_id == v.id)
                .count()
            )
            if vis_emb == 0 and scene_count > 0 and not v.filename.endswith(".wav"):
                issues.append(
                    f"  [{v.id}] {v.filename}: {scene_count} scenes, 0 visual embeddings"
                )

        print("\nRemaining issues:")
        if issues:
            for issue in issues:
                print(issue)
        else:
            print("  None! All videos are fully embedded.")

        return {
            "videos": total_videos,
            "scenes": total_scenes,
            "segments": total_segs,
            "text_embeddings": total_text_emb,
            "visual_embeddings": total_vis_emb,
            "issues": len(issues),
        }


if __name__ == "__main__":
    from database.config import test_connection, init_db

    # Test database connection
    if not test_connection():
        print("Please configure your database connection in .env file")
        exit(1)

    # Initialize database
    init_db()

    # Ingest all processed videos
    with DataIngester() as ingester:
        stats = ingester.ingest_batch(
            processed_dir="processed",
            generate_embeddings=True,
            update_existing=True,
            skip_existing=True,
        )
        print(f"\nIngestion complete: {stats['success']} videos in database")

        # Verify database state
        ingester.verify()
