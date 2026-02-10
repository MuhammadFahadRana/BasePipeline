# Fine-Tuning BGE-M3 for Domain Specificity

To get the **absolute best results** for specific domain terms (like "Omega Well", "Drilling logs"), you can fine-tune the BGE-M3 embedding model.

## 1. Concepts

- **Contrastive Learning**: The model learns to pull "positive" pairs (query matched with correct transcript) closer and push "negative" pairs (query vs random transcript) apart.
- **Hard Negatives**: The most effective training happens when you use negatives that verify *similar* but *incorrect* answers (e.g. "Alpha Well" is a hard negative for "Omega Well").

## 2. Dynamic Training Loop

To make the program dynamic and seemingly "learn" from user interactions, you can implement a **Feedback Loop**:

1.  **Capture Feedback**: Add a "Thumps Up/Down" in the UI.
2.  **Store Pairs**: Save `(query, transcript_segment, is_relevant)` to a dataset.
3.  **Retrain**: Periodically run a fine-tuning script.

## 3. Implementation Steps

### A. Install Requirements
```bash
pip install sentence-transformers datasets accelerate
```

### B. Prepare Data
Create a JSONL file `train_data.jsonl`:
```json
{"query": "where is omega well", "pos": ["Omega well is located in sector..."], "neg": ["Alpha well is 5km north..."]}
{"query": "drilling depth", "pos": ["We drilled to 5000ft..."], "neg": ["The weather is nice..."]}
```

### C. Fine-Tuning Script
Create `finetune.py`:

```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

# 1. Load Model
model = SentenceTransformer('BAAI/bge-m3')

# 2. Load Data (Dynamic part: load from database of feedback)
train_examples = [
    InputExample(texts=['where is omega well', 'Omega well is located...'], label=1),
    InputExample(texts=['where is omega well', 'Alpha well is...'], label=0)
]

# 3. Define Loss
train_loss = losses.ContrastiveLoss(model=model)

# 4. Train
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    output_path='models/bge-m3-finetuned'
)
```

### D. Use New Model
Update `text_embeddings.py` to point to your new customized model path:
```python
def __init__(self, model_name="models/bge-m3-finetuned", ...):
```
