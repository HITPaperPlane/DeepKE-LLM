# DeepKE-LLM Workflow

This folder contains a streamlined pipeline for building knowledge graphs with large language models.

1. **Data augmentation** using Qwen API.
2. **P-Tuning** on the augmented data.
3. **Instruction driven KG construction** with the tuned model.
4. **Evaluation** on the test split.

Example scripts are under `workflow/`.

## Setup
```bash
pip install -r ../requirements.txt
pip install openai transformers peft datasets
```

## Usage
```bash
# Step1: augment training data
python workflow/data_augmentation.py \
  --input dataset/InstructIE/train_zh.json \
  --output dataset/InstructIE/train_zh_aug.json

# Step2: p-tuning Qwen
python workflow/ptuning_qwen.py \
  --train dataset/InstructIE/train_zh_aug.json \
  --model Qwen/Qwen1.5-7B \
  --output qwen_pt

# Step3: build knowledge graph
python workflow/build_kg.py \
  --input dataset/InstructIE/test_zh.json \
  --output result.json

# Step4: evaluate
python workflow/evaluate.py \
  --pred result.json \
  --gold dataset/InstructIE/test_zh.json
```
