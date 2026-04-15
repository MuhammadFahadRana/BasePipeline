"""Multi-modal search combining text (transcript) and vision (keyframes) similarity."""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import text

from search.semantic_search import SemanticSearchEngine, SearchResult
from embeddings.vision_embeddings import get_vision_embedding_generator
from llm.query_parser import get_query_parser


@dataclass
class MultiModalSearchResult(SearchResult):
    """Search result with both text and vision scores."""

    text_score: float = 0.0
    vision_score: float = 0.0
    combined_score: float = 0.0
    # keyframe_path is inherited from SearchResult

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "text_score": round(self.text_score, 4),
                "vision_score": round(self.vision_score, 4),
                "combined_score": round(self.combined_score, 4),
            }
        )
        return base_dict


class MultiModalSearchEngine:
    """Search engine combining text and vision embeddings."""

    def __init__(
        self,
        db: Session,
        text_weight: float = 0.5,
        vision_weight: float = 0.5,
        vision_model: str = "google/siglip-base-patch16-224",
        text_search: SemanticSearchEngine = None,
    ):
        """
        Initialize multi-modal search engine.

        Args:
            db: Database session
            text_weight: Default weight for text similarity (0-1)
            vision_weight: Default weight for vision similarity (0-1)
            vision_model: Vision model name (SigLIP)
            text_search: Optional pre-existing SemanticSearchEngine singleton to reuse
        """
        self.db = db
        self.text_weight = text_weight
        self.vision_weight = vision_weight

        # Reuse provided text search engine or create a new one
        self.text_search = text_search if text_search is not None else SemanticSearchEngine(db)

        # Lazy load vision model (only when needed)
        self._vision_gen = None
        self.vision_model_name = vision_model

        # LLM Query Parser (lazy loaded)
        self.query_parser = None

        # Validate weights
        if not np.isclose(text_weight + vision_weight, 1.0):
            raise ValueError(
                f"Weights must sum to 1.0, got {text_weight + vision_weight}"
            )

    def update_db(self, db: Session):
        """Update the database session (for singleton reuse across requests)."""
        self.db = db
        self.text_search.db = db

    @staticmethod
    def _infer_query_intent(query: str) -> Dict[str, bool]:
        q = (query or "").lower()
        visual_tokens = {
            "show",
            "image",
            "picture",
            "frame",
            "screen",
            "scene",
            "looks like",
            "appearance",
            "logo",
            "diagram",
        }
        ocr_tokens = {
            "text on screen",
            "subtitle",
            "slide",
            "caption",
            "written",
            "spelled",
            "wording",
            "what does it say",
            "ocr",
        }
        temporal_start = {"start", "beginning", "intro", "first"}
        temporal_end = {"end", "ending", "outro", "last", "final"}

        has_visual = any(t in q for t in visual_tokens)
        has_ocr = any(t in q for t in ocr_tokens)
        prefers_start = any(t in q for t in temporal_start)
        prefers_end = any(t in q for t in temporal_end)

        return {
            "visual": has_visual,
            "ocr": has_ocr,
            "prefers_start": prefers_start,
            "prefers_end": prefers_end,
        }

    @staticmethod
    def _temporal_boost(
        frame_role: Optional[str], prefers_start: bool, prefers_end: bool
    ) -> float:
        if not frame_role:
            return 0.0
        role = frame_role.lower()
        if prefers_start and role.startswith("start"):
            return 0.12
        if prefers_end and role.startswith("end"):
            return 0.12
        if prefers_start or prefers_end:
            if role.startswith("mid"):
                return -0.04
            return 0.02
        if role.startswith("mid"):
            return 0.03
        return 0.0

    @property
    def vision_gen(self):
        """Lazy load vision embedding generator."""
        if self._vision_gen is None:
            self._vision_gen = get_vision_embedding_generator(self.vision_model_name)
        return self._vision_gen

    def search(
        self,
        query: str,
        top_k: int = 10,
        video_filter: Optional[str] = None,
        use_vision: bool = True,
        deep_search: bool = False,
    ) -> List[MultiModalSearchResult]:
        """
        Perform multi-modal search.

        Args:
            query: Search query text
            top_k: Number of results to return
            video_filter: Optional video filename filter
            use_vision: Whether to combine with visual search
            deep_search: If True, uses the expensive Cross-Encoder reranker

        Returns:
            List of multi-modal search results ranked by combined score
        """
        # 1. Get Text Candidates
        text_resp = self.text_search.search_with_fallback(
            query,
            top_k=top_k * 4 if use_vision else top_k,
            video_filter=video_filter,
            deep_search=deep_search,
        )
        text_results = text_resp["results"]

        if not use_vision or self.vision_weight == 0:
            return [
                MultiModalSearchResult(
                    **result.__dict__,
                    text_score=result.score,
                    vision_score=0.0,
                    combined_score=result.score,
                )
                for result in text_results[:top_k]
            ]

        # 2. Get Visual Candidates (Query Text -> Vision Embeddings)
        visual_candidates = self.search_visual_only(
            query=query, top_k=top_k * 2, video_filter=video_filter
        )

        # 3. Merge Candidates
        candidate_map = {}  # key -> (text_score, vision_score, SearchResult)

        # Add text candidates
        for r in text_results:
            key = (
                f"seg_{r.segment_id}"
                if r.segment_id
                else f"scene_{r.video_id}_{r.start_time}"
            )
            candidate_map[key] = [r.score, 0.0, r]

        # Add visual candidates (or update existing)
        for r in visual_candidates:
            key = (
                f"seg_{r.segment_id}"
                if r.segment_id
                else f"scene_{r.video_id}_{r.start_time}"
            )
            if key in candidate_map:
                candidate_map[key][1] = r.vision_score
            else:
                candidate_map[key] = [0.0, r.vision_score, r]

        # 4. Fill missing vision scores for text candidates
        query_vision_embedding = self.vision_gen.encode_text(query, normalize=True)

        final_results = []
        for key, (t_score, v_score, base_result) in candidate_map.items():
            # If we don't have a vision score yet, fetch it from DB
            current_v_score = v_score
            current_keyframe = base_result.keyframe_path

            if current_v_score == 0.0 and base_result.segment_id:
                vision_data = self._get_vision_embedding_for_segment(
                    base_result.segment_id
                )
                if vision_data:
                    emb, path = vision_data
                    current_v_score = float(np.dot(query_vision_embedding, emb))
                    if not current_keyframe:
                        current_keyframe = path

            # Combine scores
            combined_score = (self.text_weight * t_score) + (
                self.vision_weight * current_v_score
            )

            # Create MultiModal result (safely handling existing dict keys)
            res_data = base_result.__dict__.copy()
            res_data.update(
                {
                    "vision_score": current_v_score,
                    "combined_score": combined_score,
                    "score": combined_score,  # Update base score for sorting
                    "keyframe_path": current_keyframe,
                }
            )

            mm_res = MultiModalSearchResult(**res_data)
            final_results.append(mm_res)

        # 5. Sort and return
        final_results.sort(key=lambda x: x.combined_score, reverse=True)
        return final_results[:top_k]

    def search_with_fallback(
        self,
        query: str,
        top_k: int = 10,
        video_filter: Optional[str] = None,
        use_llm: bool = True,
        facet: str = "auto",
        deep_search: bool = False,
    ) -> dict:
        """
        Multi-modal search with tiered fallback strategy.

        Uses the text search engine's fallback tiers, then enriches
        with vision scores. Auto-adjusts text/vision weights when
        text results are low confidence to avoid score dilution.

        Returns:
            Dict with 'results', 'search_metadata'
        """
        # Use local weight variables so we don't mutate singleton state
        tw = self.text_weight
        vw = self.vision_weight

        # 0. LLM Query Parsing (Intent Understanding)
        parsed_query = None
        search_query = query

        if use_llm:
            # Initialize parser if not done
            if self.query_parser is None:
                self.query_parser = get_query_parser(enabled=True)

            if self.query_parser:
                parsed_query = self.query_parser.parse(query)
            search_query = parsed_query.normalized_query

            # Dynamic weighting based on intent
            if (
                "vision" in parsed_query.targets
                and "transcript" not in parsed_query.targets
            ):
                # Visual search intent (e.g. "orange robot") -> Boost vision
                vw = max(vw, 0.6)
                tw = 1.0 - vw
            elif "ocr" in parsed_query.targets:
                # OCR intent (e.g. "slide about X") -> Boost text (OCR is part of text search)
                tw = max(tw, 0.7)
                vw = 1.0 - tw

        heuristics = self._infer_query_intent(search_query)
        if heuristics["visual"] and not heuristics["ocr"]:
            vw = max(vw, 0.65)
            tw = 1.0 - vw
        elif heuristics["ocr"]:
            tw = max(tw, 0.75)
            vw = 1.0 - tw

        # 1. Get text results with fallback
        fallback_data = self.text_search.search_with_fallback(
            query=search_query,
            top_k=top_k * 3 if vw > 0 else top_k,
            video_filter=video_filter,
            facet=facet or "auto",
            deep_search=deep_search,
        )

        # Attach intent metadata
        if parsed_query:
            fallback_data["search_metadata"]["llm_intent"] = parsed_query.to_dict()

        text_results = fallback_data["results"]
        metadata = fallback_data["search_metadata"]

        # If no vision needed or no text results, return as-is
        if vw == 0 or not text_results:
            mm_results = [
                MultiModalSearchResult(
                    **r.__dict__,
                    text_score=r.score,
                    vision_score=0.0,
                    combined_score=r.score,
                )
                for r in text_results[:top_k]
            ]
            return {"results": mm_results, "search_metadata": metadata}

        # 2. Auto-adjust weights: if text results are weak, favor text more
        #    to avoid vision scores diluting already-weak text matches
        top_text_score = text_results[0].score if text_results else 0
        effective_text_weight = tw
        effective_vision_weight = vw

        if top_text_score < 0.3:
            # Low confidence text — go text-heavy to preserve what we found
            effective_text_weight = 0.7
            effective_vision_weight = 0.3
        elif top_text_score > 0.6:
            # High confidence text — favor text so vision doesn't dilute strong matches
            effective_text_weight = 0.65
            effective_vision_weight = 0.35

        # 3. Enrich with vision scores
        try:
            query_vision_embedding = self.vision_gen.encode_text(query, normalize=True)
        except Exception:
            # Vision unavailable — return text-only results
            mm_results = [
                MultiModalSearchResult(
                    **r.__dict__,
                    text_score=r.score,
                    vision_score=0.0,
                    combined_score=r.score,
                )
                for r in text_results[:top_k]
            ]
            return {"results": mm_results, "search_metadata": metadata}

        # Also get visual-only candidates
        try:
            visual_candidates = self.search_visual_only(
                query=query, top_k=top_k * 2, video_filter=video_filter
            )
        except Exception:
            visual_candidates = []

        # Build candidate map
        candidate_map = {}
        for r in text_results:
            key = (
                f"seg_{r.segment_id}"
                if r.segment_id
                else f"scene_{r.video_id}_{r.start_time}"
            )
            candidate_map[key] = [r.score, 0.0, r]

        for r in visual_candidates:
            key = (
                f"seg_{r.segment_id}"
                if r.segment_id
                else f"scene_{r.video_id}_{r.start_time}"
            )
            if key in candidate_map:
                candidate_map[key][1] = r.vision_score
            else:
                candidate_map[key] = [0.0, r.vision_score, r]

        # Fill missing vision scores
        raw_vision_scores = {}
        for key, (t_score, v_score, base_result) in candidate_map.items():
            current_v_score = v_score
            current_keyframe = base_result.keyframe_path
            current_evidence_time = getattr(base_result, "evidence_time", None)
            current_evidence_role = getattr(base_result, "evidence_frame_role", None)

            # If we don't have a score yet, try to fetch embedding for segment or scene
            if current_v_score == 0.0:
                vision_data = None

                # Extract scene_id or segment_id from key
                target_scene_id = None
                target_segment_id = None

                # Prefer decoding from result_id when available.
                # This matters for OCR/Visual matches where `segment_id` can be an encoded scene id.
                rid = getattr(base_result, "result_id", None)
                if isinstance(rid, str):
                    if rid.startswith("visual_"):
                        try:
                            raw = int(rid.replace("visual_", ""))
                            # visual branch encodes as scene_id + 2000000
                            target_scene_id = raw - 2000000 if raw > 2000000 else raw
                        except Exception:
                            pass
                    elif rid.startswith("ocr_"):
                        try:
                            target_scene_id = int(rid.replace("ocr_", ""))
                        except Exception:
                            pass
                elif isinstance(rid, int):
                    # Fallback to result_id logic for scene_id
                    if rid < -2000000:
                        target_scene_id = abs(rid) - 2000000
                    elif rid < 0:
                        target_scene_id = abs(rid)
                    else:
                        target_segment_id = rid

                if isinstance(key, str):
                    if key.startswith("visual_"):
                        raw = int(key.replace("visual_", ""))
                        target_scene_id = raw - 2000000 if raw > 2000000 else raw
                    elif key.startswith("ocr_"):
                        target_scene_id = int(key.replace("ocr_", ""))
                    elif key.startswith("seg_"):
                        seg_val = int(key.replace("seg_", ""))
                        # If this looks like an encoded scene id, decode it.
                        if seg_val > 2000000:
                            target_scene_id = seg_val - 2000000
                        else:
                            target_segment_id = seg_val
                    elif key.startswith("scene_"):
                        # format scene_VIDEOID_STARTTIME
                        # we might not have scene_id here directly, but search_with_fallback
                        # results usually have result_id which is negative scene_id
                        pass

                # Now fetch vision data
                if target_segment_id:
                    vision_data = self._get_vision_embedding_for_segment(
                        target_segment_id
                    )
                elif target_scene_id:
                    vision_data = self._get_vision_embedding_for_scene(target_scene_id)

                if vision_data:
                    emb, path, sample_time, frame_role = vision_data
                    current_v_score = float(np.dot(query_vision_embedding, emb))
                    if not current_keyframe:
                        current_keyframe = path
                    current_evidence_time = sample_time
                    current_evidence_role = frame_role

            raw_vision_scores[key] = (
                current_v_score,
                current_keyframe,
                current_evidence_time,
                current_evidence_role,
            )

        # Normalize vision scores to [0, 1] for fair combination with text
        # critical fix: Do NOT min-max normalize if all scores are low noise!
        all_v = [v for v, _, _, _ in raw_vision_scores.values()]
        v_min = min(all_v) if all_v else 0
        v_max = max(all_v) if all_v else 0
        
        # If the best vision score is very weak (under 0.22 typical SigLIP noise floor),
        # don't scale it up to 1.0. Anchor the min to at least 0.20 to compress weak signals.
        effective_v_min = max(0.20, v_min) if v_max < 0.3 else v_min
        v_range = v_max - effective_v_min if v_max > effective_v_min else 1.0

        final_results = []
        for key, (t_score, _, base_result) in candidate_map.items():
            (
                current_v_score,
                current_keyframe,
                current_evidence_time,
                current_evidence_role,
            ) = raw_vision_scores[key]

            # Adjust normalization: if score is below noise floor, keep it near 0
            if v_max < 0.22:
                norm_v_score = max(0.0, current_v_score)  # keep raw weak score
            else:
                norm_v_score = (current_v_score - effective_v_min) / v_range if v_range > 0 else 0.0
                norm_v_score = max(0.0, min(1.0, norm_v_score)) # clamp to 0-1

            # Combined score: if text score is literally 0.0, heavily penalize the vision score
            # to prevent purely visual (unrelated) matches from surfacing in exact-term searches
            if t_score == 0.0:
                norm_v_score *= 0.5  # 50% penalty for zero text relevance

            norm_v_score += self._temporal_boost(
                current_evidence_role,
                prefers_start=heuristics["prefers_start"],
                prefers_end=heuristics["prefers_end"],
            )
            norm_v_score = max(0.0, min(1.0, norm_v_score))

            combined_score = (effective_text_weight * t_score) + (
                effective_vision_weight * norm_v_score
            )

            res_data = base_result.__dict__.copy()
            res_data.update(
                {
                    "text_score": t_score,
                    "vision_score": norm_v_score,
                    "combined_score": combined_score,
                    "score": combined_score,
                    "keyframe_path": current_keyframe,
                    "evidence_time": current_evidence_time,
                    "evidence_frame_role": current_evidence_role,
                }
            )

            # Remove any keys that don't belong to MultiModalSearchResult
            for extra_key in list(res_data.keys()):
                if extra_key not in MultiModalSearchResult.__dataclass_fields__:
                    del res_data[extra_key]

            mm_res = MultiModalSearchResult(**res_data)
            final_results.append(mm_res)

        final_results.sort(key=lambda x: x.combined_score, reverse=True)
        return {"results": final_results[:top_k], "search_metadata": metadata}

    def _get_vision_embedding_for_segment(self, segment_id: int) -> Optional[tuple]:
        """
        Get vision embedding for a segment via its scene.

        Returns:
            Tuple of (embedding_array, keyframe_path) or None
        """
        if not segment_id:
            return None

        result = self.db.execute(
            text("""
            SELECT ve.embedding, ve.keyframe_path, ve.sample_time, ve.frame_role
            FROM transcript_segments ts
            -- Most segments may have ts.scene_id = NULL; fall back to time overlap within same video.
            JOIN scenes s ON (
                ts.scene_id = s.id
                OR (
                    ts.scene_id IS NULL
                    AND s.video_id = ts.video_id
                    AND ts.start_time >= s.start_time
                    AND ts.start_time <= s.end_time
                )
            )
            JOIN visual_embeddings ve ON s.id = ve.scene_id
            WHERE ts.id = :segment_id
            AND ve.embedding_model = :model_name
            ORDER BY CASE ve.frame_role
                WHEN 'mid' THEN 0
                WHEN 'start' THEN 1
                WHEN 'end' THEN 2
                ELSE 3
            END,
            COALESCE(ve.sample_time, 0)
            LIMIT 1
        """),
            {"segment_id": segment_id, "model_name": self.vision_model_name},
        )

        row = result.fetchone()
        if row:
            raw_embedding = row[0]
            if isinstance(raw_embedding, str):
                # Handle cases where DB returns vector as string like "[1,2,3]"
                import json

                try:
                    embedding = np.array(json.loads(raw_embedding), dtype=np.float32)
                except Exception:
                    # Alternative parsing if json.loads fails (e.g. pgvector string format)
                    cleaned = raw_embedding.replace("[", "").replace("]", "").split(",")
                    embedding = np.array(
                        [float(x.strip()) for x in cleaned if x.strip()],
                        dtype=np.float32,
                    )
            else:
                embedding = np.array(raw_embedding, dtype=np.float32)

            # Normalize if not already (safeguard)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding, row[1], row[2], row[3]

        return None

    def _get_vision_embedding_for_scene(self, scene_id: int) -> Optional[tuple]:
        """
        Get vision embedding for a scene directly.

        Returns:
            Tuple of (embedding_array, keyframe_path) or None
        """
        result = self.db.execute(
            text("""
            SELECT ve.embedding, ve.keyframe_path, ve.sample_time, ve.frame_role
            FROM visual_embeddings ve
            WHERE ve.scene_id = :scene_id
            AND ve.embedding_model = :model_name
            ORDER BY CASE ve.frame_role
                WHEN 'mid' THEN 0
                WHEN 'start' THEN 1
                WHEN 'end' THEN 2
                ELSE 3
            END,
            COALESCE(ve.sample_time, 0)
            LIMIT 1
        """),
            {"scene_id": scene_id, "model_name": self.vision_model_name},
        )

        row = result.fetchone()
        if row:
            raw_embedding = row[0]
            if isinstance(raw_embedding, str):
                import json

                try:
                    embedding = np.array(json.loads(raw_embedding), dtype=np.float32)
                except Exception:
                    cleaned = raw_embedding.replace("[", "").replace("]", "").split(",")
                    embedding = np.array(
                        [float(x.strip()) for x in cleaned if x.strip()],
                        dtype=np.float32,
                    )
            else:
                embedding = np.array(raw_embedding, dtype=np.float32)

            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding, row[1], row[2], row[3]

        return None

    def search_visual_only(
        self, query: str, top_k: int = 10, video_filter: Optional[str] = None
    ) -> List[MultiModalSearchResult]:
        """
        Perform vision-only search (query text -> find similar keyframes).

        Args:
            query: Search query text
            top_k: Number of results to return
            video_filter: Filter by specific video filename

        Returns:
            List of results ranked by visual similarity
        """
        # Generate vision embedding for query
        query_vision_embedding = self.vision_gen.encode_text(query, normalize=True)

        # Search visual embeddings database
        query_filter = ""
        params = {
            "query_embedding": query_vision_embedding.tolist(),
            "top_k": top_k,
            "model_name": self.vision_model_name,
        }

        if video_filter:
            query_filter = "AND v.filename = :video_filter"
            params["video_filter"] = video_filter

        sql_query = f"""
            WITH ranked AS (
                SELECT
                    ts.id as segment_id,
                    v.id as video_id,
                    v.filename as video_filename,
                    v.file_path as video_path,
                    COALESCE(ts.start_time, s.start_time) as start_time,
                    COALESCE(ts.end_time, s.end_time) as end_time,
                    COALESCE(ts.text, '[Visual match]') as text,
                    ve.keyframe_path,
                    ve.sample_time,
                    ve.frame_role,
                    1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS similarity,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.id
                        ORDER BY ve.embedding <=> CAST(:query_embedding AS vector)
                    ) AS rn
                FROM visual_embeddings ve
                JOIN scenes s ON ve.scene_id = s.id
                JOIN videos v ON s.video_id = v.id
                LEFT JOIN transcript_segments ts ON ts.scene_id = s.id
                WHERE ve.embedding_model = :model_name
                {query_filter}
            )
            SELECT * FROM ranked
            WHERE rn = 1
            ORDER BY similarity DESC
            LIMIT :top_k
        """

        result = self.db.execute(text(sql_query), params)

        results = []
        for row in result.fetchall():
            res = MultiModalSearchResult(
                segment_id=row.segment_id,
                video_id=row.video_id,
                video_filename=row.video_filename,
                video_path=row.video_path,
                start_time=row.start_time,
                end_time=row.end_time,
                text=row.text,
                score=0.0,
                vision_score=float(row.similarity),
                combined_score=float(row.similarity),
                match_type="visual",
                evidence_time=row.sample_time,
                evidence_frame_role=row.frame_role,
            )
            res.keyframe_path = row.keyframe_path
            results.append(res)

        return results


def set_optimal_weights(search_type: str = "balanced") -> tuple:
    """
    Get optimal text/vision weights for different search scenarios.

    Args:
        search_type: One of "balanced", "text_heavy", "vision_heavy", "visual_only"

    Returns:
        Tuple of (text_weight, vision_weight)
    """
    weights = {
        "balanced": (0.5, 0.5),
        "text_heavy": (0.7, 0.3),
        "vision_heavy": (0.3, 0.7),
        "visual_only": (0.0, 1.0),
        "text_only": (1.0, 0.0),
    }

    return weights.get(search_type, (0.5, 0.5))
