from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image

# YOLOv8
from ultralytics import YOLO

# CLIP (OpenCLIP)
import torch
import open_clip


@dataclass
class RefinerConfig:
    yolo_model: str = "yolov8n.pt"              # small + fast; use yolov8s.pt for higher accuracy
    yolo_person_conf: float = 0.35
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    clip_sim_merge_threshold: float = 0.90     # higher = merge only very similar shots
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SemanticSceneRefiner:
    """
    Adds semantic signals on top of shot detection:
      - speaker/person presence using YOLO on keyframes
      - visual similarity using CLIP embeddings to merge near-identical consecutive scenes
    """

    def __init__(self, cfg: Optional[RefinerConfig] = None):
        self.cfg = cfg or RefinerConfig()

        # YOLO
        self.yolo = YOLO(self.cfg.yolo_model)

        # CLIP
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.cfg.clip_model, pretrained=self.cfg.clip_pretrained
        )
        self.clip_model = self.clip_model.to(self.cfg.device).eval()
        self.tokenizer = open_clip.get_tokenizer(self.cfg.clip_model)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def _person_present(self, image_path: str) -> Tuple[bool, float]:
        """
        Returns (person_present, max_confidence_for_person)
        COCO class 0 is 'person' for YOLOv8.
        """
        if not image_path or not Path(image_path).exists():
            return False, 0.0

        results = self.yolo(image_path, verbose=False)
        max_conf = 0.0
        present = False

        for r in results:
            if r.boxes is None:
                continue
            cls = r.boxes.cls.detach().cpu().numpy().astype(int)
            conf = r.boxes.conf.detach().cpu().numpy()
            # person class == 0
            person_confs = conf[cls == 0] if len(cls) else np.array([])
            if person_confs.size > 0:
                max_conf = float(person_confs.max())
                if max_conf >= self.cfg.yolo_person_conf:
                    present = True

        return present, max_conf

    @torch.no_grad()
    def _clip_embed(self, image_path: str) -> Optional[np.ndarray]:
        if not image_path or not Path(image_path).exists():
            return None

        img = self._load_image(image_path)
        x = self.preprocess(img).unsqueeze(0).to(self.cfg.device)

        feats = self.clip_model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).detach().cpu().numpy()

    def _cos_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def refine(self, scenes: List[Dict]) -> List[Dict]:
        """
        Adds:
          - scene["speaker_present"] (bool)
          - scene["speaker_conf"] (float)
          - scene["clip_sim_to_prev"] (float or None)
        Then merges consecutive scenes if:
          - both have speaker_present
          - CLIP similarity is above threshold
        """
        # 1) annotate scenes with YOLO + CLIP
        clip_embs: List[Optional[np.ndarray]] = []
        for i, s in enumerate(scenes):
            kf = s.get("keyframe_path")
            speaker_present, speaker_conf = self._person_present(kf) if kf else (False, 0.0)
            s["speaker_present"] = speaker_present
            s["speaker_conf"] = speaker_conf

            emb = self._clip_embed(kf) if kf else None
            clip_embs.append(emb)

            if i > 0 and emb is not None and clip_embs[i - 1] is not None:
                s["clip_sim_to_prev"] = self._cos_sim(clip_embs[i - 1], emb)
            else:
                s["clip_sim_to_prev"] = None

        # 2) merge consecutive similar speaker scenes
        merged: List[Dict] = []
        i = 0
        while i < len(scenes):
            cur = scenes[i]
            cur_emb = clip_embs[i]

            j = i + 1
            while j < len(scenes):
                nxt = scenes[j]
                nxt_emb = clip_embs[j]

                if not (cur.get("speaker_present") and nxt.get("speaker_present")):
                    break
                if cur_emb is None or nxt_emb is None:
                    break

                sim = self._cos_sim(cur_emb, nxt_emb)
                if sim < self.cfg.clip_sim_merge_threshold:
                    break

                # merge nxt into cur
                cur["end_time"] = nxt["end_time"]
                cur["duration"] = float(cur["end_time"] - cur["start_time"])
                # keep the keyframe of the longer segment or keep the first; your choice:
                # here: keep first keyframe, but you could update based on duration
                j += 1

            merged.append(cur)
            i = j

        # reassign scene ids
        for idx, s in enumerate(merged, 1):
            s["scene_id"] = idx

        return merged
