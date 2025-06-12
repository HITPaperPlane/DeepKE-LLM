# Knowledge Graph Construction Pipeline

This folder contains a simplified workflow for building knowledge graphs with a deployed **Qwen** model. The process consists of four steps:

1. **Data Augmentation** using the language model on the `InstructIE` and `IEPile` training sets;
2. **P-Tuning** of Qwen with the augmented data (black-box fine-tuning);
3. **Instruction-based KG Construction** on the test sets;
4. **Evaluation** of the extraction results.

## 1. Data Augmentation

```bash
python pipeline/data_augmentation.py
```

The script calls the Qwen API to generate additional labeled examples.

## 2. P-Tuning

```bash
python pipeline/ptuning.py
```

This wraps `InstructKGC/src/finetuning_chatglm_pt.py` and trains a soft prompt for Qwen. Adjust model paths and training parameters as needed.

## 3. Instruction-driven KG Construction

```bash
python pipeline/kg_construction.py
```

The script invokes Qwen to extract triples from the test sets according to a given instruction.

## 4. Evaluation

```bash
python pipeline/evaluate.py
```

By default the relation extraction task is evaluated. Modify the script to compute other metrics or evaluation modes.
