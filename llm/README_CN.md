# 知识图谱构建流程

本目录展示如何使用部署好的 **Qwen** API 构建知识图谱。流程包含四个阶段：

1. **数据增强**：利用大语言模型扩充 `InstructIE` 与 `IEPile` 两个数据集的训练集；
2. **P-Tuning 微调**：使用增强后的数据对 Qwen 进行黑盒 P-Tuning；
3. **指令式知识图谱构建**：调用 Qwen 按指令抽取实体关系；
4. **评估**：分别在给定 Schema 与自动识别 Schema 两种模式下评估效果。

## 1. 数据增强

运行 `pipeline/data_augmentation.py` 即可对两个数据集进行增强，生成的文件保存在原数据集目录下：

```bash
python pipeline/data_augmentation.py
```

脚本会调用 Qwen API 生成新的标注样例，具体参数可在脚本中调整。

## 2. P-Tuning 微调

利用增强后的数据启动 P-Tuning 训练：

```bash
python pipeline/ptuning.py
```

该脚本内部调用 `InstructKGC/src/finetuning_chatglm_pt.py`，并以 Qwen 模型路径和训练集路径作为输入。请根据实际环境修改 `model_dir` 等参数。

## 3. 指令驱动的知识图谱构建

完成微调后，可使用以下命令在测试集上进行知识图谱构建：

```bash
python pipeline/kg_construction.py
```

脚本会读取测试集，调用 Qwen 按给定指令抽取三元组，结果保存在 `kg_output.json`。

## 4. 评估

使用 `pipeline/evaluate.py` 评估模型输出：

```bash
python pipeline/evaluate.py
```

默认评估关系抽取任务（`RE`），同时可通过修改脚本参数支持不同任务或按数据源拆分评估。

## 数据集结构

- `dataset/InstructIE`：按照 `train_zh.json`、`train_en.json`、`valid_zh.json`、`test_zh.json` 等文件组织；
- `dataset/iepile`：统一格式文件 `train.json`、`dev.json` 以及 `IE-en`、`IE-zh` 目录。

有关格式详情请参阅仓库根目录 `agent.md` 中的说明。
