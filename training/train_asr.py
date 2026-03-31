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
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


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
    ):
        self.processor = processor
        self.max_input_length_sec = max_input_length_sec
        self.jsonl_path = Path(jsonl_path).resolve()
        self.manifest_dir = self.jsonl_path.parent
        self.samples = []

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

                if not raw_audio:
                    print(
                        f"  Warning: Skipping sample with no audio path "
                        f"(line {line_no} in {self.jsonl_path})"
                    )
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

        print(f"  Loaded {len(self.samples)} samples from {self.jsonl_path}")

    def __len__(self):
        return len(self.samples)

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
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        return candidates[0].resolve()

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

    def __getitem__(self, idx):
        sample = self.samples[idx]
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

    print(f"\n{'=' * 60}")
    print("ATLAS Whisper ASR Fine-Tuning")
    print(f"{'=' * 60}")
    print(f"Model:     {config.model_name}")
    print(f"LoRA:      {config.use_lora}")
    print(f"Epochs:    {config.epochs}")
    print(f"Batch:     {config.batch_size}")
    print(f"Data:      {config.data_dir}")
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

    model.generation_config.update(
        language=config.language,
        task=config.task,
        forced_decoder_ids=None,
        suppress_tokens=None,
        begin_suppress_tokens=None,
    )

    # 2. Apply LoRA if configured
    if config.use_lora:
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
            print("Warning: peft not installed, training full model")
            config.use_lora = False

    # 3. Load datasets
    data_dir = Path(config.data_dir)
    train_jsonl = (
        Path(config.train_file).resolve()
        if config.train_file
        else (data_dir / "train.jsonl").resolve()
    )
    eval_jsonl = (
        Path(config.eval_file).resolve()
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

    # 4. Training arguments
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / effective_batch_size))
    total_train_steps = max(1, int(steps_per_epoch * config.epochs))
    warmup_steps = max(0, int(total_train_steps * config.warmup_ratio))

    print(
        f"Warmup: {warmup_steps} steps "
        f"(~{config.warmup_ratio:.0%} of {total_train_steps} total steps)"
    )

    training_args = Seq2SeqTrainingArguments(
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
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.eval_steps if eval_dataset else None,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="wer" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,
    )

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
    if config.use_lora:
        adapter_dir = output_dir / "lora_adapter"
        model.save_pretrained(str(adapter_dir))
        print(f"  LoRA adapter saved to: {adapter_dir}")
    else:
        full_model_dir = output_dir / "full_model"
        trainer.save_model(str(full_model_dir))
        print(f"  Full model saved to: {full_model_dir}")

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
            "use_lora": config.use_lora,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "language": config.language,
            "bf16": bf16_enabled,
            "fp16": fp16_enabled,
        },
        "train_samples": len(train_dataset),
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
        "--no-lora",
        action="store_true",
        help="Disable LoRA and train the full model",
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
    args = parser.parse_args()

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
        use_lora=not args.no_lora,
        fp16=args.fp16,
        bf16=bf16_value,
    )

    train(config, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
