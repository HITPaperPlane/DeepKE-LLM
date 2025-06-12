# DeepKE-LLM 工作流

本目录提供基于千问大语言模型的知识图谱构建示例。整体流程如下：

1. **数据增强**：利用千问 API 对训练集进行扩充；
2. **P-Tuning 微调**：在增强后的数据上对 Qwen 进行软提示微调；
3. **指令驱动构建**：使用微调后的模型抽取知识图谱三元组；
4. **效果评估**：在测试集上计算三元组抽取的 F1 值。

示例脚本位于 `workflow/` 文件夹中。

## 运行环境
```bash
pip install -r ../requirements.txt
pip install openai transformers peft datasets
```

## 使用方法
### 1. 数据增强
```bash
python workflow/data_augmentation.py \
  --input dataset/InstructIE/train_zh.json \
  --output dataset/InstructIE/train_zh_aug.json
```

### 2. P-Tuning 微调 Qwen
```bash
python workflow/ptuning_qwen.py \
  --train dataset/InstructIE/train_zh_aug.json \
  --model Qwen/Qwen1.5-7B \
  --output qwen_pt
```

### 3. 指令驱动的知识图谱构建
```bash
python workflow/build_kg.py \
  --input dataset/InstructIE/test_zh.json \
  --output result.json
```

### 4. 评估
```bash
python workflow/evaluate.py \
  --pred result.json \
  --gold dataset/InstructIE/test_zh.json
```

如需在 `IEPile` 数据集上实验，只需将 `--input` 与 `--gold` 参数换成对应的文件即可。
