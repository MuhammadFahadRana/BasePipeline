"""
train_embeddings.py

Fine-tune Qwen3-Embedding-0.6B for contrastive retrieval using LoRA adapters
and MultipleNegativesRankingLoss.

Usage:
    python -m training.train_embeddings                                # train with defaults
    python -m training.train_embeddings --config training/config.yaml  # custom config
    python -m training.train_embeddings --max-samples 50 --epochs 1    # quick test run
"""

import json
import random
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


# ──────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────

def load_config(config_path: str = "training/config.yaml") -> Dict:
    """Load training configuration from YAML."""
    import yaml
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────

def load_dataset(dataset_path: str, enrichment_mode: str = "transcript_only"):
    """
    Load the retrieval dataset and convert to sentence-transformers InputExamples.

    Args:
        dataset_path: Path to retrieval_dataset.json
        enrichment_mode: How to represent text for training
            - "transcript_only": just segment.text
            - "transcript+ocr": segment.text + OCR
            - "transcript+caption+ocr": segment.text + caption + OCR + labels

    Returns:
        (train_examples, eval_examples) — lists of InputExample objects
    """
    from sentence_transformers import InputExample

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    entries = dataset.get("entries", [])
    if not entries:
        raise ValueError(f"No entries found in {dataset_path}")

    examples: List[InputExample] = []

    for entry in entries:
        query = entry["query"]
        positive = entry["positive"]
        pos_text = _enrich_text(
            positive.get("segment_text", ""),
            entry.get("metadata", {}),
            enrichment_mode,
        )

        # For MNRL: each InputExample has texts=[query, positive]
        # The loss function treats all other positives in the batch as negatives
        examples.append(InputExample(texts=[query, pos_text]))

        # Explicit hard negatives: each creates a (query, positive, negative) triplet
        # MNRL with hard negatives: texts=[anchor, positive, neg1, neg2, ...]
        hard_negs = entry.get("hard_negatives", [])
        if hard_negs:
            neg_texts = [neg.get("segment_text", "") for neg in hard_negs if neg.get("segment_text")]
            if neg_texts:
                examples.append(InputExample(texts=[query, pos_text] + neg_texts))

    return examples


def _enrich_text(text: str, metadata: Dict, mode: str) -> str:
    """Apply enrichment mode to create the searchable text representation."""
    if mode == "transcript_only":
        return text

    parts = [text]

    if mode in ("transcript+ocr", "transcript+caption+ocr"):
        ocr = metadata.get("ocr_text")
        if ocr:
            parts.append(ocr)

    if mode == "transcript+caption+ocr":
        caption = metadata.get("caption")
        if caption:
            parts.append(caption)
        labels = metadata.get("object_labels", [])
        if labels:
            parts.append(" ".join(str(lbl) for lbl in labels))

    return " ".join(parts)


def train_eval_split(
    examples: List, eval_ratio: float = 0.15, seed: int = 42
):
    """Split examples into train and eval sets."""
    random.seed(seed)
    shuffled = list(examples)
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - eval_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


# ──────────────────────────────────────────────
# LoRA adapter application
# ──────────────────────────────────────────────

def apply_lora(model, config: Dict):
    """
    Apply LoRA adapters to a SentenceTransformer model using PEFT.

    Args:
        model: SentenceTransformer model
        config: model section of config.yaml

    Returns:
        model with LoRA adapters applied
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError("peft is required for LoRA training. Run: pip install peft")

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("lora_target_modules", ["q_proj", "v_proj"]),
        bias="none",
    )

    # SentenceTransformer wraps a Transformer model in model[0].auto_model
    base_transformer = model[0].auto_model
    peft_model = get_peft_model(base_transformer, lora_config)

    # Print trainable params summary
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"LoRA applied: {trainable:,} trainable params / {total:,} total "
          f"({100 * trainable / total:.2f}%)")

    model[0].auto_model = peft_model
    return model


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate_model(
    model,
    eval_examples: List,
    batch_size: int = 32,
) -> Dict:
    """
    Evaluate the model on held-out examples.
    Computes Recall@1, Recall@5, Recall@10, and MRR.
    """
    if not eval_examples:
        return {"recall@1": 0, "recall@5": 0, "recall@10": 0, "mrr": 0}

    queries = []
    positives = []

    for ex in eval_examples:
        if len(ex.texts) >= 2:
            queries.append(ex.texts[0])
            positives.append(ex.texts[1])

    if not queries:
        return {"recall@1": 0, "recall@5": 0, "recall@10": 0, "mrr": 0}

    # Encode all
    q_embs = model.encode(queries, batch_size=batch_size, normalize_embeddings=True)
    p_embs = model.encode(positives, batch_size=batch_size, normalize_embeddings=True)

    # Compute similarity matrix: queries × positives
    sim_matrix = np.dot(q_embs, p_embs.T)  # shape: (N, N)

    recall_at = {1: 0, 5: 0, 10: 0}
    mrr_sum = 0.0

    for i in range(len(queries)):
        # Rank all positives by similarity to query i
        scores = sim_matrix[i]
        ranked = np.argsort(-scores)  # descending

        # Find rank of the correct positive (index i)
        rank = np.where(ranked == i)[0][0] + 1  # 1-indexed

        for k in recall_at:
            if rank <= k:
                recall_at[k] += 1

        mrr_sum += 1.0 / rank

    n = len(queries)
    metrics = {
        "recall@1": round(recall_at[1] / n, 4),
        "recall@5": round(recall_at[5] / n, 4),
        "recall@10": round(recall_at[10] / n, 4),
        "mrr": round(mrr_sum / n, 4),
        "eval_samples": n,
    }

    return metrics


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────

def train(config: Dict, max_samples: Optional[int] = None):
    """
    Main training function.
    Loads data, applies LoRA, trains with MNRL, evaluates, and saves.
    """
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})

    # ── Load model ──
    base_model_name = model_cfg.get("base", "Qwen/Qwen3-Embedding-0.6B")
    print(f"\n{'=' * 60}")
    print(f"ATLAS Embedding Fine-Tuning")
    print(f"{'=' * 60}")
    print(f"Base model: {base_model_name}")
    print(f"Method:     {model_cfg.get('method', 'lora')}")
    print(f"Loss:       {train_cfg.get('loss', 'mnrl')}")
    print(f"Epochs:     {train_cfg.get('epochs', 3)}")
    print(f"{'=' * 60}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = SentenceTransformer(base_model_name, device=device, trust_remote_code=True)

    # ── Apply LoRA if configured ──
    if model_cfg.get("method") == "lora":
        model = apply_lora(model, model_cfg)

    # ── Load dataset ──
    dataset_path = data_cfg.get("dataset_path", "training/retrieval_dataset.json")
    enrichment_mode = data_cfg.get("enrichment_mode", "transcript_only")

    print(f"\nLoading dataset: {dataset_path}")
    print(f"Enrichment mode: {enrichment_mode}")

    all_examples = load_dataset(dataset_path, enrichment_mode)

    if max_samples and len(all_examples) > max_samples:
        all_examples = random.sample(all_examples, max_samples)
        print(f"Reduced to {max_samples} samples (--max-samples)")

    # ── Train/eval split ──
    eval_ratio = data_cfg.get("eval_split", 0.15)
    seed = train_cfg.get("seed", 42)
    train_examples, eval_examples = train_eval_split(all_examples, eval_ratio, seed)

    print(f"Train samples: {len(train_examples)}")
    print(f"Eval samples:  {len(eval_examples)}")

    # ── DataLoader ──
    batch_size = train_cfg.get("batch_size", 16)
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
    )

    # ── Loss ──
    loss_name = train_cfg.get("loss", "mnrl")
    if loss_name == "mnrl":
        train_loss = losses.MultipleNegativesRankingLoss(model)
    elif loss_name == "infonce":
        # MNRL is effectively InfoNCE, but with a temperature parameter
        train_loss = losses.MultipleNegativesRankingLoss(
            model, scale=1.0 / train_cfg.get("temperature", 0.05)
        )
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    # ── Evaluate baseline (before training) ──
    print("\nBaseline evaluation (before training)...")
    baseline_metrics = evaluate_model(model, eval_examples, batch_size=batch_size)
    print(f"  Recall@1:  {baseline_metrics['recall@1']}")
    print(f"  Recall@5:  {baseline_metrics['recall@5']}")
    print(f"  Recall@10: {baseline_metrics['recall@10']}")
    print(f"  MRR:       {baseline_metrics['mrr']}")

    # ── Train ──
    epochs = train_cfg.get("epochs", 3)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.1)
    warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)
    lr = train_cfg.get("learning_rate", 2e-5)

    model_dir = Path(output_cfg.get("model_dir", "training/checkpoints"))
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training...")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Learning rate: {lr}")
    print(f"  Total batches: {len(train_dataloader) * epochs}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        weight_decay=train_cfg.get("weight_decay", 0.01),
        output_path=str(model_dir),
        use_amp=train_cfg.get("fp16", True) and device == "cuda",
        show_progress_bar=True,
    )

    # ── Evaluate after training ──
    print("\nPost-training evaluation...")
    trained_metrics = evaluate_model(model, eval_examples, batch_size=batch_size)
    print(f"  Recall@1:  {trained_metrics['recall@1']}  (was {baseline_metrics['recall@1']})")
    print(f"  Recall@5:  {trained_metrics['recall@5']}  (was {baseline_metrics['recall@5']})")
    print(f"  Recall@10: {trained_metrics['recall@10']} (was {baseline_metrics['recall@10']})")
    print(f"  MRR:       {trained_metrics['mrr']}       (was {baseline_metrics['mrr']})")

    # ── Save results ──
    eval_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "base_model": base_model_name,
            "method": model_cfg.get("method"),
            "loss": loss_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "enrichment_mode": enrichment_mode,
        },
        "train_samples": len(train_examples),
        "eval_samples": len(eval_examples),
        "baseline": baseline_metrics,
        "trained": trained_metrics,
        "improvement": {
            k: round(trained_metrics[k] - baseline_metrics[k], 4)
            for k in ["recall@1", "recall@5", "recall@10", "mrr"]
        },
    }

    eval_path = Path(output_cfg.get("eval_results", "training/eval_results.json"))
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"  Model saved to:  {model_dir}")
    print(f"  Eval results:    {eval_path}")
    print(f"{'=' * 60}")

    # ── Save LoRA adapter separately if applicable ──
    if model_cfg.get("method") == "lora":
        adapter_dir = model_dir / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        try:
            model[0].auto_model.save_pretrained(str(adapter_dir))
            print(f"  LoRA adapter:    {adapter_dir}")
        except Exception as e:
            print(f"  Warning: Could not save LoRA adapter separately: {e}")

    return eval_results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-Embedding for ATLAS retrieval"
    )
    parser.add_argument(
        "--config", default="training/config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit training samples (for quick test runs)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs

    train(config, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
