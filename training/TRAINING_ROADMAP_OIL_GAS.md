# Oil & Gas Maximum-Performance Training Roadmap

Date: 2026-04-20

This roadmap is LoRA-first, then non-LoRA when gains plateau. It is tailored to your current pipeline:
- Embedding model: `Qwen/Qwen3-Embedding-0.6B`
- Scene pipeline: PySceneDetect + CLIP merge, with optional supervised TransNetV2 fallback
- Retrieval: hybrid search + reranking
- QA: `Qwen/Qwen2.5-1.5B-Instruct` RAG answer generation

## 1) Success Criteria (Per Model Type)

Do not use one global “error rate”. Track these targets separately:

- Embedding / Retrieval candidate generation:
  - `Recall@10 >= 0.99`
  - `Recall@1 >= 0.80` (stretch; depends on label quality)
  - `MRR >= 0.88`
- Reranker:
  - `nDCG@10 >= 0.92`
  - `MRR >= 0.90`
  - `P@1 >= 0.85`
- Scene detection:
  - Boundary `F1 >= 0.95` (at +/- 0.5s tolerance)
  - Over-segmentation rate `< 5%`
- QA:
  - Grounded answer accuracy `>= 90%`
  - Unsupported-claim rate `<= 3%`
  - Citation span hit rate `>= 95%`

## 2) Promotion Rule (When To Move Beyond LoRA)

Treat performance as plateaued if BOTH are true for 2 consecutive runs:
- Absolute gain in primary metric is < 0.5 percentage points
- Relative gain is < 1.0%

When plateaued, move from LoRA/QLoRA to full fine-tuning for that model.

## 3) Data Policy (Required Before Any Training)

- Freeze 3 splits by source-video groups (not random row split):
  - Train: 70%
  - Dev: 15%
  - Test: 15%
- Keep a hard Oil & Gas holdout set with rare terminology and diagram-heavy segments.
- Keep retrieval and QA evaluation sets disjoint from training.
- For embedding/retrieval examples, use enriched text:
  - `transcript + caption + ocr + object_labels`

---

## 4) Embedding Model (Qwen) Roadmap

### Stage E1 - LoRA (First)

Use this config as `training/config.embedding.stage1.lora.yaml`:

```yaml
model:
  base: "Qwen/Qwen3-Embedding-0.6B"
  method: "lora"
  lora_r: 32
  lora_alpha: 64
  lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  lora_dropout: 0.05

training:
  loss: "mnrl"
  epochs: 5
  batch_size: 24
  learning_rate: 1.5e-5
  warmup_ratio: 0.1
  weight_decay: 0.01
  hard_negatives_per_query: 12
  temperature: 0.03
  fp16: true
  gradient_accumulation_steps: 2
  seed: 42

data:
  dataset_path: "training/retrieval_dataset.json"
  eval_split: 0.15
  enrichment_mode: "transcript+caption+ocr"
  negatives:
    easy_count: 2
    medium_count: 3
    hard_temporal_count: 3
    hard_keyword_count: 3
    hard_visual_count: 1

output:
  model_dir: "training/checkpoints/embedding_stage1_lora"
  eval_results: "training/eval_results.embedding_stage1_lora.json"
  logs_dir: "training/logs/embedding_stage1_lora"
```

Commands:

```powershell
python -m training.build_training_dataset --config training/config.embedding.stage1.lora.yaml --processed-dir processed --from-judgments training/relevance_judgments.csv --output training/retrieval_dataset.json --seed 42
python -m training.train_embeddings --config training/config.embedding.stage1.lora.yaml
python -m training.evaluate_retrieval --model Qwen/Qwen3-Embedding-0.6B --adapter training/checkpoints/embedding_stage1_lora/lora_adapter --dataset training/retrieval_dataset.json --processed-dir processed --output training/eval_results.embedding_stage1_lora.eval.json
```

### Stage E2 - Full Fine-Tuning (After LoRA Plateau)

Use this config as `training/config.embedding.stage2.full.yaml`:

```yaml
model:
  base: "Qwen/Qwen3-Embedding-0.6B"
  method: "full"

training:
  loss: "mnrl"
  epochs: 3
  batch_size: 8
  learning_rate: 2.0e-6
  warmup_ratio: 0.1
  weight_decay: 0.01
  hard_negatives_per_query: 14
  temperature: 0.03
  fp16: true
  gradient_accumulation_steps: 8
  seed: 42

data:
  dataset_path: "training/retrieval_dataset.json"
  eval_split: 0.15
  enrichment_mode: "transcript+caption+ocr"
  negatives:
    easy_count: 1
    medium_count: 3
    hard_temporal_count: 4
    hard_keyword_count: 4
    hard_visual_count: 2

output:
  model_dir: "training/checkpoints/embedding_stage2_full"
  eval_results: "training/eval_results.embedding_stage2_full.json"
  logs_dir: "training/logs/embedding_stage2_full"
```

Commands:

```powershell
python -m training.train_embeddings --config training/config.embedding.stage2.full.yaml
python -m training.evaluate_retrieval --model training/checkpoints/embedding_stage2_full --dataset training/retrieval_dataset.json --processed-dir processed --output training/eval_results.embedding_stage2_full.eval.json
```

---

## 5) Scene Detection Roadmap

LoRA is not appropriate for your current scene stack because scene cuts are mostly heuristic + thresholded post-processing.

### Stage S1 - Tune Existing Detector (No LoRA)

Grid to run:

```yaml
scene_tuning_grid:
  threshold: [16.0, 18.0, 20.0, 22.0, 24.0]
  min_scene_len: [12, 15, 18]
  max_scene_duration: [45.0, 60.0]
  clip_sim_merge_threshold: [0.86, 0.90, 0.93]
```

Current runnable command (threshold is CLI-supported):

```powershell
python basic_pipeline.py --video videos/Risk management.mp4 --threshold 18.0 --force
```

For each run, log:
- boundary F1 (+/- 0.5s and +/- 1.0s)
- number of scenes
- over/under segmentation rate

### Stage S2 - Supervised Scene Model (TransNetV2)

If S1 cannot meet target, train supervised shot-boundary detection.

1) Prepare mapping files:
- `training/scene/train_mapping.csv`
- `training/scene/val_mapping.csv`
- `training/scene/test_mapping.csv`

Each line format:
`/abs/path/video.mp4,/abs/path/scenes.txt`

2) Build datasets:

```powershell
python Extras/models/TransNetV2/training/create_dataset.py train --mapping_fn training/scene/train_mapping.csv --target_dir training/scene/tfrecords/train --target_fn trainset --w 48 --h 27
python Extras/models/TransNetV2/training/create_dataset.py test  --mapping_fn training/scene/val_mapping.csv   --target_dir training/scene/tfrecords/val --w 48 --h 27
python Extras/models/TransNetV2/training/create_dataset.py test-npy --mapping_fn training/scene/test_mapping.csv --target_dir training/scene/test_npy --w 48 --h 27
```

3) Train with a gin config (save as `training/scene/transnetv2_oilgas.gin`):

```gin
options.log_dir = "training/scene/logs"
options.log_name = "transnetv2_oilgas_stage2"
options.n_epochs = 50
options.trn_files = ["training/scene/tfrecords/train/trainset-*.tfrecord"]
options.tst_files = {"val": ["training/scene/tfrecords/val/*.tfrecord"]}
options.input_shape = [100, 27, 48, 3]

training.log_freq = 100
training.grad_clipping = 10.0
training.n_batches_per_epoch = 2000

loss.transition_weight = 5.0
loss.many_hot_loss_weight = 0.3
loss.l2_loss_weight = 1e-6
```

4) Train and evaluate:

```powershell
python Extras/models/TransNetV2/training/training.py training/scene/transnetv2_oilgas.gin
python Extras/models/TransNetV2/training/evaluate.py training/scene/logs/transnetv2_oilgas_stage2_<timestamp> 50 training/scene/test_npy --thr 0.5
```

---

## 6) Retrieval Reranker Roadmap

Goal: replace zero-shot LLM judging with a trained reranker.

### Stage R1 - Cross-Encoder LoRA/QLoRA

Training config (framework-agnostic; use your preferred trainer):

```yaml
reranker_stage1:
  model_name: "BAAI/bge-reranker-base"
  method: "lora_or_qlora"
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  quantization:
    load_in_4bit: true
    bnb_4bit_compute_dtype: "bfloat16"
    bnb_4bit_quant_type: "nf4"
  data:
    train_pairs: "training/reranker/train_pairs.jsonl"
    dev_pairs: "training/reranker/dev_pairs.jsonl"
    max_query_chars: 256
    max_doc_chars: 1024
    negatives_per_query: 8
  optimization:
    lr: 2.0e-5
    batch_size: 16
    grad_accum: 2
    epochs: 3
    weight_decay: 0.01
    warmup_ratio: 0.06
```

### Stage R2 - Full Cross-Encoder Fine-Tuning (After Plateau)

```yaml
reranker_stage2:
  model_name: "BAAI/bge-reranker-base"
  method: "full"
  data:
    train_pairs: "training/reranker/train_pairs.jsonl"
    dev_pairs: "training/reranker/dev_pairs.jsonl"
    negatives_per_query: 12
  optimization:
    lr: 1.0e-5
    batch_size: 8
    grad_accum: 4
    epochs: 2
    weight_decay: 0.01
    warmup_ratio: 0.1
```

Deployment policy:
- Use reranker on top `k=50` candidates from fast retrieval.
- Keep lexical+OCR branches; rerank final merged candidates only.

---

## 7) QA Model Roadmap

### Stage Q1 - Grounded SFT with LoRA/QLoRA

Use supervised examples with explicit citation spans:

```yaml
qa_stage1_sft:
  base_model: "Qwen/Qwen2.5-1.5B-Instruct"
  method: "qlora"
  lora:
    r: 32
    alpha: 64
    dropout: 0.05
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  quantization:
    load_in_4bit: true
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_compute_dtype: "bfloat16"
  data:
    train_file: "training/qa/train_grounded.jsonl"
    dev_file: "training/qa/dev_grounded.jsonl"
    input_format: "question + retrieved_context + required_citation_schema"
    output_format: "answer + citations[{video,start,end}]"
  optimization:
    lr: 1.0e-4
    batch_size: 8
    grad_accum: 4
    epochs: 3
    max_seq_len: 4096
    warmup_ratio: 0.03
    weight_decay: 0.01
```

### Stage Q2 - Preference Tuning (After SFT Plateau)

```yaml
qa_stage2_preference:
  base_from: "qa_stage1_sft_best_checkpoint"
  method: "dpo_or_orpo"
  data:
    preference_pairs: "training/qa/preference_pairs.jsonl"
    labels: ["grounded_preferred", "hallucinated_rejected"]
  optimization:
    lr: 5.0e-6
    batch_size: 4
    grad_accum: 8
    epochs: 1
    beta: 0.1
```

Scoring gates:
- Keep only responses with valid citations.
- Penalize unsupported claims and citation mismatches.

---

## 8) Execution Order (Recommended)

1. Embedding Stage E1 (LoRA) with richer enrichment and harder negatives
2. Retrieval Reranker Stage R1 (LoRA/QLoRA cross-encoder)
3. QA Stage Q1 (grounded SFT)
4. Scene Stage S1 (threshold tuning), then S2 only if needed
5. Move to E2/R2/Q2 only when plateau rule is met

## 9) Immediate Next Run (Concrete)

Run this first:

```powershell
python -m training.build_training_dataset --config training/config.embedding.stage1.lora.yaml --processed-dir processed --from-judgments training/relevance_judgments.csv --output training/retrieval_dataset.json --seed 42
python -m training.train_embeddings --config training/config.embedding.stage1.lora.yaml
python -m training.evaluate_retrieval --model Qwen/Qwen3-Embedding-0.6B --adapter training/checkpoints/embedding_stage1_lora/lora_adapter --dataset training/retrieval_dataset.json --processed-dir processed --output training/eval_results.embedding_stage1_lora.eval.json
```

Then compare against your current baseline before deciding E2.
