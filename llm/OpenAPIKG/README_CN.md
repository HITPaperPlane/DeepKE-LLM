# OpenAPIKG

该模块展示如何使用遵循 OpenAPI 风格的远程大模型完成知识图谱构建，已完全移除 CPM-Bee 相关代码。

整体流程包含三个核心步骤：

1. **数据集准备** - 将原始 JSON 数据转换为指令格式样例（`src/dataset_preparation.py`）。
2. **知识图谱生成** - 通过 API key 调用远程模型获得三元组预测（`src/generate_kg.py`）。
3. **知识图谱评估** - 对预测结果进行评估（`src/evaluate_kg.py`）。

## 环境依赖
```bash
pip install -r requirements.txt
```

## 使用示例
### 1. 数据准备
```bash
python src/dataset_preparation.py --input raw.json --train_out train.jsonl --eval_out eval.jsonl
```

### 2. 远程模型生成三元组
```bash
python src/generate_kg.py --input eval.jsonl --output pred.jsonl \
    --api_url https://api.example.com/v1/chat/completions --api_key YOUR_API_KEY
```

### 3. 评估
```bash
python src/evaluate_kg.py --ref eval.jsonl --pred pred.jsonl
```
