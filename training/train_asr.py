"""
train_asr.py

Fine-tune Whisper for domain-specific ASR using audio + ground-truth transcript pairs.

Training flow (correct):
    audio chunk + ground-truth text -> sequence loss -> backprop -> update weights
    WER is used only for EVALUATION, not for the loss function.

Usage:
    python -m training.train_asr
    python -m training.train_asr --model openai/whisper-large-v3
    python -m training.train_asr --data-dir training/asr_data
    python -m training.train_asr --max-samples 50 --epochs 1
    python -m training.train_asr --training-technique decoder_only
    python -m training.train_asr --augment-copies 2 --noise-std 0.003
"""

import argparse
import json
import math
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def resolve_training_path(path_value: str, must_exist: bool = False) -> Path:
    """
    Resolve paths robustly whether commands run from repo root or Extras/.
    """
    raw = Path(path_value).expanduser()
    if raw.is_absolute():
        resolved = raw.resolve()
        if not must_exist or resolved.exists():
            return resolved
        return resolved

    script_dir = Path(__file__).resolve().parent          # .../Extras/training
    extras_dir = script_dir.parent                        # .../Extras
    repo_root = extras_dir.parent                         # .../BasePipeline

    candidates = [
        Path.cwd() / raw,
        extras_dir / raw,      # supports "training/asr_data" from repo root
        repo_root / raw,       # supports direct repo-relative paths
        script_dir / raw,      # supports local relative invocation
    ]

    for candidate in candidates:
        resolved = candidate.resolve()
        if not must_exist or resolved.exists():
            return resolved

    return (Path.cwd() / raw).resolve()


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────


@dataclass
class ASRTrainingConfig:
    """Configuration for Whisper fine-tuning."""

    # Model
    model_name: str = "openai/whisper-large-v3"
    language: str = "en"
    task: str = "transcribe"
    training_technique: str = "lora"  # lora | decoder_only | full
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32

    # Training
    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = False
    bf16: Optional[bool] = None
    max_input_length_sec: float = 30.0
    augment_copies: int = 1
    augmentation_probability: float = 0.85
    speed_perturb_min: float = 0.95
    speed_perturb_max: float = 1.05
    noise_std: float = 0.002
    random_seed: int = 42

    # Data
    data_dir: str = "training/asr_data"
    output_dir: str = "training/asr_checkpoints"
    train_file: Optional[str] = None
    eval_file: Optional[str] = None

    # Evaluation
    eval_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 25


def load_asr_config(config_path: Optional[str] = None) -> ASRTrainingConfig:
    """Load ASR training config from YAML or use defaults."""
    config = ASRTrainingConfig()

    if config_path and Path(config_path).exists():
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        asr_cfg = data.get("asr", {})
        for key, value in asr_cfg.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return config


# ──────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────


class WhisperASRDataset(torch.utils.data.Dataset):
    """
    Dataset that loads audio-transcript pairs from the JSONL manifest
    and preprocesses them for Whisper fine-tuning.
    """

    def __init__(
        self,
        jsonl_path: str,
        processor,
        max_input_length_sec: float = 30.0,
        max_samples: Optional[int] = None,
        min_transcript_words: int = 2,
        enable_augmentation: bool = False,
        augment_copies: int = 0,
        augmentation_probability: float = 0.85,
        speed_perturb_range: Tuple[float, float] = (0.95, 1.05),
        noise_std: float = 0.002,
        random_seed: int = 42,
    ):
        self.processor = processor
        self.max_input_length_sec = max_input_length_sec
        self.jsonl_path = resolve_training_path(jsonl_path, must_exist=True)
        self.manifest_dir = self.jsonl_path.parent
        self.min_transcript_words = min_transcript_words
        self.enable_augmentation = bool(enable_augmentation)
        self.augment_copies = max(0, int(augment_copies))
        self.augmentation_probability = max(
            0.0, min(1.0, float(augmentation_probability))
        )
        sp_min, sp_max = speed_perturb_range
        if sp_min <= 0 or sp_max <= 0:
            raise ValueError("Speed perturb range values must be > 0.")
        self.speed_perturb_range = (min(sp_min, sp_max), max(sp_min, sp_max))
        self.noise_std = max(0.0, float(noise_std))
        self.random_seed = int(random_seed)
        self.samples = []
        skipped_short = 0

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"  Warning: Skipping invalid JSONL line {line_no} "
                        f"in {self.jsonl_path}: {e}"
                    )
                    continue

                raw_audio = raw.get("audio") or raw.get("audio_path")
                transcript = (
                    raw.get("sentence")
                    or raw.get("text")
                    or raw.get("transcript")
                    or ""
                )
                transcript = self.clean_transcript(transcript)

                if not raw_audio:
                    print(
                        f"  Warning: Skipping sample with no audio path "
                        f"(line {line_no} in {self.jsonl_path})"
                    )
                    continue

                if len(transcript.split()) < self.min_transcript_words:
                    skipped_short += 1
                    continue

                audio_path = self.normalize_audio_path(raw_audio)

                self.samples.append(
                    {
                        "audio_path": audio_path,
                        "transcript": transcript,
                    }
                )

        if max_samples:
            self.samples = self.samples[:max_samples]

        self.base_samples_count = len(self.samples)
        self.effective_samples_count = self.base_samples_count
        if self.enable_augmentation and self.augment_copies > 0:
            self.effective_samples_count = self.base_samples_count * (
                1 + self.augment_copies
            )

        print(f"  Loaded {len(self.samples)} samples from {self.jsonl_path}")
        if skipped_short:
            print(f"  Skipped {skipped_short} samples with very short transcripts")
        if self.effective_samples_count != self.base_samples_count:
            print(
                f"  Training-set expansion with augmentation: "
                f"{self.base_samples_count} -> {self.effective_samples_count} "
                f"(augment_copies={self.augment_copies})"
            )

    def __len__(self):
        return self.effective_samples_count

    def normalize_audio_path(self, audio_path: str) -> Path:
        """
        Normalize Windows/Linux paths robustly.

        Handles:
        - training\\asr_data\\audio\\file.wav
        - training/asr_data/audio/file.wav
        - audio/file.wav
        """
        raw = Path(str(audio_path).replace("\\", "/"))

        if raw.is_absolute():
            return raw

        candidates = [
            Path.cwd() / raw,
            self.manifest_dir / raw,
            self.manifest_dir.parent / raw,
            self.manifest_dir.parent.parent / raw,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        return candidates[0].resolve()

    @staticmethod
    def clean_transcript(text: str) -> str:
        """
        Normalize transcript text for stable training labels.
        """
        text = unicodedata.normalize("NFKC", str(text or ""))

        # Common punctuation normalization (keeps semantics while avoiding mojibake artifacts)
        text = (
            text.replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("…", "...")
        )

        # Optional ftfy-based repair if available
        try:
            import ftfy

            text = ftfy.fix_text(text)
        except Exception:
            pass

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _load_audio(self, audio_path: Path):
        """
        Load audio safely and return mono 16k waveform as numpy array.
        """
        try:
            import librosa

            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            return audio, sr
        except Exception:
            try:
                from transcriber_utils import load_audio_array

                audio, sr = load_audio_array(audio_path)
                return audio, sr
            except Exception as e:
                print(f"  Warning: Could not load {audio_path}: {e}")
                return None, None

    @staticmethod
    def _speed_perturb(audio: np.ndarray, factor: float) -> np.ndarray:
        """
        Lightweight time-stretch by linear resampling.
        """
        if factor <= 0 or len(audio) < 2 or abs(factor - 1.0) < 1e-3:
            return audio

        old_indices = np.arange(len(audio), dtype=np.float32)
        new_length = max(1, int(len(audio) / factor))
        new_indices = np.linspace(0, len(audio) - 1, new_length, dtype=np.float32)
        return np.interp(new_indices, old_indices, audio).astype(np.float32)

    def _augment_audio(self, audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Apply simple waveform augmentations to create additional training variants.
        """
        augmented = np.asarray(audio, dtype=np.float32)

        if rng.random() <= self.augmentation_probability:
            speed = float(rng.uniform(*self.speed_perturb_range))
            augmented = self._speed_perturb(augmented, speed)

        if self.noise_std > 0 and rng.random() <= self.augmentation_probability:
            rms = float(np.sqrt(np.mean(np.square(augmented))) + 1e-8)
            scaled_std = self.noise_std * max(rms, 0.01)
            noise = rng.normal(0.0, scaled_std, size=augmented.shape).astype(np.float32)
            augmented = augmented + noise

        if rng.random() <= self.augmentation_probability:
            gain_db = float(rng.uniform(-3.0, 3.0))
            gain = 10 ** (gain_db / 20.0)
            augmented = augmented * gain

        return np.clip(augmented, -1.0, 1.0).astype(np.float32)

    def __getitem__(self, idx):
        if self.base_samples_count == 0:
            raise IndexError("Dataset is empty.")

        base_idx = idx % self.base_samples_count
        augmentation_slot = idx // self.base_samples_count
        use_augmented_variant = self.enable_augmentation and augmentation_slot > 0

        sample = self.samples[base_idx]
        audio_path = sample["audio_path"]
        transcript = sample["transcript"]

        if not audio_path.exists():
            print(f"  Warning: Missing audio file: {audio_path}")
            audio = torch.zeros(16000, dtype=torch.float32).numpy()
            transcript = ""
        else:
            audio, sr = self._load_audio(audio_path)
            if audio is None:
                audio = torch.zeros(16000, dtype=torch.float32).numpy()
                transcript = ""

        audio = np.asarray(audio, dtype=np.float32)

        if use_augmented_variant and transcript:
            aug_seed = self.random_seed + (idx * 1009) + (augmentation_slot * 37)
            rng = np.random.default_rng(aug_seed)
            audio = self._augment_audio(audio, rng)

        max_samples = int(self.max_input_length_sec * 16000)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
        )

        labels = self.processor.tokenizer(
            transcript,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=448,
        ).input_ids[0]

        return {
            "input_features": input_features.input_features[0],
            "attention_mask": input_features.attention_mask[0],
            "labels": labels,
        }


# ──────────────────────────────────────────────
# Data collator
# ──────────────────────────────────────────────


@dataclass
class WhisperDataCollator:
    """
    Custom data collator for Whisper that:
    - Pads input features to the same length
    - Pads labels and replaces padding with -100
    """

    processor: Any

    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features:
            raise ValueError("Received an empty batch after filtering invalid samples.")

        input_features = [f["input_features"] for f in features]
        input_attention_masks = [f["attention_mask"] for f in features]
        label_features = [f["labels"] for f in features]

        batch = self.processor.feature_extractor.pad(
            {
                "input_features": input_features,
                "attention_mask": input_attention_masks,
            },
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_features},
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (
            labels.shape[1] > 0
            and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ──────────────────────────────────────────────
# WER metric
# ──────────────────────────────────────────────


def compute_wer_metric(pred, processor):
    """Compute WER for evaluation during training."""

    def _word_error_rate(predictions, references):
        def _edit_distance(a_words, b_words):
            rows = len(a_words) + 1
            cols = len(b_words) + 1
            dp = [[0] * cols for _ in range(rows)]

            for i in range(rows):
                dp[i][0] = i
            for j in range(cols):
                dp[0][j] = j

            for i in range(1, rows):
                for j in range(1, cols):
                    cost = 0 if a_words[i - 1] == b_words[j - 1] else 1
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1,
                        dp[i - 1][j - 1] + cost,
                    )

            return dp[-1][-1]

        total_words = 0
        total_errors = 0

        for hyp, ref in zip(predictions, references):
            ref_words = ref.split()
            hyp_words = hyp.split()

            total_words += len(ref_words)
            total_errors += _edit_distance(ref_words, hyp_words)

        if total_words == 0:
            return 0.0

        return total_errors / total_words

    try:
        import importlib

        evaluate = importlib.import_module("evaluate")
        wer_metric = evaluate.load("wer")
        has_evaluate = True
    except ImportError:
        has_evaluate = False

    pred_ids = pred.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Normalize punctuation/case so WER reflects recognition quality rather than styling.
    def normalize_for_wer(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    pred_str = [normalize_for_wer(t) for t in pred_str]
    label_str = [normalize_for_wer(t) for t in label_str]

    if has_evaluate:
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
    else:
        wer = _word_error_rate(pred_str, label_str)

    return {"wer": round(wer * 100, 2)}


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────


def train(config: ASRTrainingConfig, max_samples: Optional[int] = None):
    """
    Main training function for Whisper fine-tuning.
    """
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    training_technique = (
        str(config.training_technique or "lora").strip().lower().replace("-", "_")
    )
    if training_technique not in {"lora", "decoder_only", "full"}:
        raise ValueError(
            f"Unsupported training_technique='{config.training_technique}'. "
            "Use one of: lora, decoder_only, full."
        )
    if training_technique == "lora" and not config.use_lora:
        training_technique = "full"

    config.training_technique = training_technique
    config.use_lora = training_technique == "lora"

    print(f"\n{'=' * 60}")
    print("ATLAS Whisper ASR Fine-Tuning")
    print(f"{'=' * 60}")
    print(f"Model:     {config.model_name}")
    print(f"Technique: {config.training_technique}")
    print(f"Epochs:    {config.epochs}")
    print(f"Batch:     {config.batch_size}")
    print(f"Data:      {config.data_dir}")
    if config.augment_copies > 0:
        print(f"Augment:   +{config.augment_copies} copies/sample")
    print(f"{'=' * 60}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    bf16_enabled = False
    fp16_enabled = False
    if device == "cuda":
        bf16_supported = torch.cuda.is_bf16_supported()
        bf16_enabled = config.bf16 if config.bf16 is not None else bf16_supported
        fp16_enabled = config.fp16 and not bf16_enabled

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    print(f"Mixed precision -> bf16: {bf16_enabled}, fp16: {fp16_enabled}")

    # 1. Load model and processor
    print(f"\nLoading Whisper: {config.model_name}")
    processor = WhisperProcessor.from_pretrained(config.model_name)

    # Load in float32 — the Trainer's bf16/fp16 autocast handles mixed precision
    # during both training AND evaluation/generate(). Loading in float16 directly
    # causes dtype mismatches when generate() runs outside autocast during eval.
    model = WhisperForConditionalGeneration.from_pretrained(
        config.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )

    forced_decoder_ids = (
        processor.get_decoder_prompt_ids(language=config.language, task=config.task)
        if config.language
        else None
    )
    model.generation_config.update(
        language=config.language,
        task=config.task,
        forced_decoder_ids=forced_decoder_ids,
    )

    # 2. Configure trainable parameters
    if training_technique == "lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
            )

            model = get_peft_model(model, lora_config)

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(
                f"LoRA applied: {trainable:,} trainable / {total:,} total "
                f"({100 * trainable / total:.2f}%)"
            )
        except ImportError:
            print(
                "Warning: `peft` is not installed. Falling back from LoRA to full fine-tuning."
            )
            training_technique = "full"
            config.training_technique = "full"
            config.use_lora = False

    if training_technique == "decoder_only":
        if hasattr(model, "freeze_encoder"):
            model.freeze_encoder()
        elif hasattr(model, "model") and hasattr(model.model, "encoder"):
            for param in model.model.encoder.parameters():
                param.requires_grad = False

        if hasattr(model, "model") and hasattr(model.model, "decoder"):
            for param in model.model.decoder.parameters():
                param.requires_grad = True
        if hasattr(model, "proj_out"):
            for param in model.proj_out.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"Decoder-only fine-tuning: {trainable:,} trainable / {total:,} total "
            f"({100 * trainable / total:.2f}%)"
        )

    if training_technique == "full":
        for param in model.parameters():
            param.requires_grad = True

    # 3. Load datasets
    data_dir = resolve_training_path(config.data_dir, must_exist=True)
    train_jsonl = (
        resolve_training_path(config.train_file, must_exist=True)
        if config.train_file
        else (data_dir / "train.jsonl").resolve()
    )
    eval_jsonl = (
        resolve_training_path(config.eval_file, must_exist=True)
        if config.eval_file
        else (data_dir / "eval.jsonl").resolve()
    )

    if not train_jsonl.exists():
        print(f"ERROR: Training data not found at {train_jsonl}")
        print("Run `python -m training.prepare_asr_data` first.")
        return

    print("\nLoading training data...")
    train_dataset = WhisperASRDataset(
        str(train_jsonl),
        processor,
        max_input_length_sec=config.max_input_length_sec,
        max_samples=max_samples,
        enable_augmentation=config.augment_copies > 0,
        augment_copies=config.augment_copies,
        augmentation_probability=config.augmentation_probability,
        speed_perturb_range=(config.speed_perturb_min, config.speed_perturb_max),
        noise_std=config.noise_std,
        random_seed=config.random_seed,
    )
    if len(train_dataset) == 0:
        print("ERROR: Training dataset is empty after filtering.")
        return
    base_train_samples = getattr(train_dataset, "base_samples_count", len(train_dataset))
    effective_train_samples = len(train_dataset)
    if effective_train_samples != base_train_samples:
        print(
            f"Effective train samples per epoch: {effective_train_samples} "
            f"(base={base_train_samples}, augmented={effective_train_samples - base_train_samples})"
        )

    eval_dataset = None
    if eval_jsonl.exists():
        print("Loading eval data...")
        eval_dataset = WhisperASRDataset(
            str(eval_jsonl),
            processor,
            max_input_length_sec=config.max_input_length_sec,
            max_samples=max_samples,
        )
        if len(eval_dataset) == 0:
            print("Warning: Eval dataset is empty after filtering. Disabling eval.")
            eval_dataset = None

    # 4. Training arguments
    output_dir = resolve_training_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / effective_batch_size))
    total_train_steps = max(1, int(steps_per_epoch * config.epochs))
    warmup_steps = max(0, int(total_train_steps * config.warmup_ratio))

    print(
        f"Warmup: {warmup_steps} steps "
        f"(~{config.warmup_ratio:.0%} of {total_train_steps} total steps)"
    )

    if eval_dataset and total_train_steps >= max(10, config.eval_steps):
        eval_strategy = "steps"
        eval_steps = max(10, min(config.eval_steps, total_train_steps))
        save_strategy = "steps"
        save_steps = eval_steps
    else:
        eval_strategy = "epoch" if eval_dataset else "no"
        eval_steps = None
        save_strategy = "epoch"
        save_steps = None

    print(
        f"Eval/save strategy -> eval: {eval_strategy}"
        + (f" ({eval_steps} steps)" if eval_steps else "")
        + f", save: {save_strategy}"
        + (f" ({save_steps} steps)" if save_steps else "")
    )

    training_args_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,
        fp16=fp16_enabled,
        bf16=bf16_enabled,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        logging_steps=config.logging_steps,
        predict_with_generate=True,
        generation_max_length=448,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="wer" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,
    )
    if eval_steps is not None:
        training_args_kwargs["eval_steps"] = eval_steps
    if save_steps is not None:
        training_args_kwargs["save_steps"] = save_steps

    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)

    data_collator = WhisperDataCollator(processor=processor)

    compute_metrics_fn = (
        (lambda pred: compute_wer_metric(pred, processor)) if eval_dataset else None
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
        compute_metrics=compute_metrics_fn,
    )

    # 7. Train
    print("\nStarting training...")
    train_result = trainer.train()

    # 8. Save
    print("\nSaving model...")
    if training_technique == "lora":
        adapter_dir = output_dir / "lora_adapter"
        trainer.model.save_pretrained(str(adapter_dir))
        print(f"  LoRA adapter saved to: {adapter_dir}")
    else:
        full_model_dir = output_dir / "full_model"
        trainer.save_model(str(full_model_dir))
        print(f"  Full model saved to: {full_model_dir}")

    if eval_dataset:
        best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
        print(f"  Best checkpoint: {best_ckpt or 'N/A'}")

    processor_dir = output_dir / "processor"
    processor.save_pretrained(str(processor_dir))

    # 9. Final evaluation
    final_metrics = {}
    if eval_dataset:
        print("\nFinal evaluation...")
        eval_results = trainer.evaluate()
        final_metrics = eval_results
        print(f"  Final WER: {eval_results.get('eval_wer', 'N/A')}%")

    # 10. Save training results
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config.model_name,
            "training_technique": training_technique,
            "use_lora": config.use_lora,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "language": config.language,
            "bf16": bf16_enabled,
            "fp16": fp16_enabled,
            "augment_copies": config.augment_copies,
            "augmentation_probability": config.augmentation_probability,
            "speed_perturb_min": config.speed_perturb_min,
            "speed_perturb_max": config.speed_perturb_max,
            "noise_std": config.noise_std,
            "random_seed": config.random_seed,
            "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
        },
        "train_samples": effective_train_samples,
        "train_samples_base": base_train_samples,
        "eval_samples": len(eval_dataset) if eval_dataset else 0,
        "train_loss": float(train_result.training_loss),
        "final_metrics": final_metrics,
    }

    results_path = output_dir / "training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("ASR TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Train loss:  {train_result.training_loss:.4f}")
    if final_metrics:
        print(f"  Final WER:   {final_metrics.get('eval_wer', 'N/A')}%")
    print(f"  Results:     {results_path}")
    print(f"{'=' * 60}")

    return results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper ASR on domain-specific audio"
    )
    parser.add_argument(
        "--model",
        default="openai/whisper-large-v3",
        help="Whisper model to fine-tune",
    )
    parser.add_argument(
        "--data-dir",
        default="training/asr_data",
        help="Path to prepared ASR data",
    )
    parser.add_argument(
        "--train-file",
        default=None,
        help="Optional explicit path to train JSONL",
    )
    parser.add_argument(
        "--eval-file",
        default=None,
        help="Optional explicit path to eval JSONL",
    )
    parser.add_argument(
        "--output",
        default="training/asr_checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device batch size",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--language",
        default="en",
        help="Language for Whisper decoder prompt",
    )
    parser.add_argument(
        "--training-technique",
        choices=["lora", "decoder_only", "full"],
        default="lora",
        help="Training strategy: LoRA adapters, decoder-only fine-tuning, or full fine-tuning",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Backward-compatible alias for --training-technique full",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Force fp16 instead of bf16",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable bf16 auto-detection",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples for quick test runs",
    )
    parser.add_argument(
        "--augment-copies",
        type=int,
        default=1,
        help="How many augmented variants to add per original training sample",
    )
    parser.add_argument(
        "--augmentation-prob",
        type=float,
        default=0.85,
        help="Probability of applying each augmentation transform (0..1)",
    )
    parser.add_argument(
        "--speed-perturb-min",
        type=float,
        default=0.95,
        help="Minimum speed perturbation factor for augmentation",
    )
    parser.add_argument(
        "--speed-perturb-max",
        type=float,
        default=1.05,
        help="Maximum speed perturbation factor for augmentation",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.002,
        help="Relative Gaussian noise level used in augmentation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.augment_copies < 0:
        parser.error("--augment-copies must be >= 0.")
    if args.speed_perturb_min <= 0 or args.speed_perturb_max <= 0:
        parser.error("--speed-perturb-min and --speed-perturb-max must be > 0.")
    if args.speed_perturb_min > args.speed_perturb_max:
        parser.error("--speed-perturb-min must be <= --speed-perturb-max.")
    if not (0.0 <= args.augmentation_prob <= 1.0):
        parser.error("--augmentation-prob must be between 0 and 1.")

    training_technique = args.training_technique
    if args.no_lora and training_technique == "lora":
        training_technique = "full"
    elif args.no_lora and training_technique != "lora":
        print(
            "Warning: --no-lora is ignored because --training-technique "
            f"is explicitly set to '{training_technique}'."
        )

    if args.max_samples is not None and args.max_samples < 32:
        print(
            f"Warning: --max-samples={args.max_samples} is very small and may hurt ASR quality. "
            "Use this only for quick debug runs."
        )

    bf16_value = None
    if args.no_bf16:
        bf16_value = False
    elif args.fp16:
        bf16_value = False

    config = ASRTrainingConfig(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output,
        train_file=args.train_file,
        eval_file=args.eval_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        language=args.language,
        training_technique=training_technique,
        use_lora=training_technique == "lora",
        fp16=args.fp16,
        bf16=bf16_value,
        augment_copies=args.augment_copies,
        augmentation_probability=args.augmentation_prob,
        speed_perturb_min=args.speed_perturb_min,
        speed_perturb_max=args.speed_perturb_max,
        noise_std=args.noise_std,
        random_seed=args.seed,
    )

    train(config, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
