"""
train_asr.py

Fine-tune Whisper for domain-specific ASR using audio + ground-truth transcript pairs.

Training flow (correct):
    audio chunk + ground-truth text -> sequence loss -> backprop -> update weights
    WER is used only for EVALUATION, not for the loss function.

Usage:
    python -m training.train_asr                                    # train with defaults
    python -m training.train_asr --model openai/whisper-base        # specific model
    python -m training.train_asr --data-dir training/asr_data       # custom data path
    python -m training.train_asr --max-samples 50 --epochs 1        # quick test
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

@dataclass
class ASRTrainingConfig:
    """Configuration for Whisper fine-tuning."""
    # Model
    model_name: str = "openai/whisper-base"
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
    fp16: bool = True
    max_input_length_sec: float = 30.0

    # Data
    data_dir: str = "training/asr_data"
    output_dir: str = "training/asr_checkpoints"

    # Evaluation
    eval_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 25


def load_asr_config(config_path: Optional[str] = None) -> ASRTrainingConfig:
    """Load ASR training config from YAML or use defaults."""
    config = ASRTrainingConfig()

    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

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
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"  Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample["audio"]
        transcript = sample["sentence"]

        # Load audio
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception:
            try:
                from transcriber_utils import load_audio_array
                audio, sr = load_audio_array(Path(audio_path))
            except Exception as e:
                print(f"  Warning: Could not load {audio_path}: {e}")
                # Return a dummy sample
                audio = torch.zeros(16000).numpy()  # 1 second of silence
                transcript = ""

        # Truncate if too long
        max_samples = int(self.max_input_length_sec * 16000)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Process audio -> mel spectrogram features
        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features[0]

        # Process text -> token IDs (these are the labels / targets)
        labels = self.processor.tokenizer(
            transcript,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=448,
        ).input_ids[0]

        return {
            "input_features": input_features,
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
    - Pads labels and replaces padding with -100 (ignored by loss)
    """
    processor: object

    def __call__(self, features):
        # Separate input features and labels
        input_features = [f["input_features"] for f in features]
        label_features = [f["labels"] for f in features]

        # Pad input features (mel spectrograms)
        batch = self.processor.feature_extractor.pad(
            {"input_features": input_features},
            return_tensors="pt",
        )

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_features},
            return_tensors="pt",
        )

        # Replace padding token id with -100 so it's ignored by the loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If the beginning-of-sentence token is appended, remove it
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ──────────────────────────────────────────────
# WER metric
# ──────────────────────────────────────────────

def compute_wer_metric(pred, processor):
    """Compute WER for evaluation during training."""
    try:
        import evaluate
        wer_metric = evaluate.load("wer")
    except ImportError:
        print("  Warning: 'evaluate' package not installed. Skipping WER computation.")
        return {"wer": -1}

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": round(wer * 100, 2)}


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train(config: ASRTrainingConfig, max_samples: Optional[int] = None):
    """
    Main training function for Whisper fine-tuning.

    Flow:
        1. Load Whisper model + processor
        2. Optionally apply LoRA adapters
        3. Load audio-transcript pairs from JSONL manifest
        4. Train with sequence-to-sequence loss (cross-entropy on decoder output)
        5. Evaluate with WER on held-out data
        6. Save model/adapter checkpoint
    """
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )

    print(f"\n{'=' * 60}")
    print(f"ATLAS Whisper ASR Fine-Tuning")
    print(f"{'=' * 60}")
    print(f"Model:     {config.model_name}")
    print(f"LoRA:      {config.use_lora}")
    print(f"Epochs:    {config.epochs}")
    print(f"Batch:     {config.batch_size}")
    print(f"Data:      {config.data_dir}")
    print(f"{'=' * 60}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load model and processor
    print(f"\nLoading Whisper: {config.model_name}")
    processor = WhisperProcessor.from_pretrained(config.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)

    # Set forced decoder IDs for language and task
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=config.language, task=config.task
    )
    model.config.suppress_tokens = []

    # 2. Apply LoRA if configured
    if config.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType

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
            print(f"LoRA applied: {trainable:,} trainable / {total:,} total "
                  f"({100 * trainable / total:.2f}%)")
        except ImportError:
            print("Warning: peft not installed, training full model")
            config.use_lora = False

    # 3. Load datasets
    data_dir = Path(config.data_dir)
    train_jsonl = data_dir / "train.jsonl"
    eval_jsonl = data_dir / "eval.jsonl"

    if not train_jsonl.exists():
        print(f"ERROR: Training data not found at {train_jsonl}")
        print(f"Run `python -m training.prepare_asr_data` first.")
        return

    print(f"\nLoading training data...")
    train_dataset = WhisperASRDataset(
        str(train_jsonl), processor,
        max_input_length_sec=config.max_input_length_sec,
        max_samples=max_samples,
    )

    eval_dataset = None
    if eval_jsonl.exists():
        print(f"Loading eval data...")
        eval_dataset = WhisperASRDataset(
            str(eval_jsonl), processor,
            max_input_length_sec=config.max_input_length_sec,
        )

    # 4. Training arguments
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        fp16=config.fp16 and device == "cuda",
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.eval_steps if eval_dataset else None,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="wer" if eval_dataset else None,
        greater_is_better=False,
        report_to="none",  # disable wandb/tensorboard by default
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # 5. Data collator
    data_collator = WhisperDataCollator(processor=processor)

    # 6. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
        compute_metrics=lambda pred: compute_wer_metric(pred, processor)
            if eval_dataset else None,
    )

    # 7. Train!
    print(f"\nStarting training...")
    train_result = trainer.train()

    # 8. Save
    print(f"\nSaving model...")
    if config.use_lora:
        # Save only the LoRA adapter
        adapter_dir = output_dir / "lora_adapter"
        model.save_pretrained(str(adapter_dir))
        print(f"  LoRA adapter saved to: {adapter_dir}")
    else:
        trainer.save_model(str(output_dir / "full_model"))
        print(f"  Full model saved to: {output_dir / 'full_model'}")

    # Save processor too (needed for inference)
    processor.save_pretrained(str(output_dir / "processor"))

    # 9. Final evaluation
    final_metrics = {}
    if eval_dataset:
        print(f"\nFinal evaluation...")
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
        },
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset else 0,
        "train_loss": train_result.training_loss,
        "final_metrics": final_metrics,
    }

    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"ASR TRAINING COMPLETE")
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
        "--model", default="openai/whisper-base",
        help="Whisper model to fine-tune"
    )
    parser.add_argument(
        "--data-dir", default="training/asr_data",
        help="Path to prepared ASR data (from prepare_asr_data.py)"
    )
    parser.add_argument(
        "--output", default="training/asr_checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--language", default="en",
        help="Language for Whisper decoder prompt"
    )
    parser.add_argument(
        "--no-lora", action="store_true",
        help="Disable LoRA (train full model)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit training samples (for quick test runs)"
    )
    args = parser.parse_args()

    config = ASRTrainingConfig(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        language=args.language,
        use_lora=not args.no_lora,
    )

    train(config, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
