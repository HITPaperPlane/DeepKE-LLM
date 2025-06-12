# OpenAPIKG

This module provides a lightweight pipeline to construct knowledge graphs with any large language model exposed through an OpenAPI-style interface. CPM-Bee related codes have been removed.

The workflow contains three main stages:

1. **Dataset Preparation** - convert raw JSON data into instruction-style examples for remote models (`src/dataset_preparation.py`).
2. **Knowledge Graph Generation** - call a remote API with an API key to obtain triple predictions (`src/generate_kg.py`).
3. **Knowledge Graph Evaluation** - evaluate the predicted triples against gold data (`src/evaluate_kg.py`).

## Requirements
```bash
pip install -r requirements.txt
```

## Usage
### 1. Prepare Dataset
```bash
python src/dataset_preparation.py --input raw.json --train_out train.jsonl --eval_out eval.jsonl
```

### 2. Generate Triples with a Remote Model
```bash
python src/generate_kg.py --input eval.jsonl --output pred.jsonl \
    --api_url https://api.example.com/v1/chat/completions --api_key YOUR_API_KEY
```

### 3. Evaluate
```bash
python src/evaluate_kg.py --ref eval.jsonl --pred pred.jsonl
```
