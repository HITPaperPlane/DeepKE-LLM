import json
from pathlib import Path
from openai import OpenAI

API_BASE = "http://192.168.11.218:8192/v1"
MODEL_NAME = "novel"

client = OpenAI(api_key="EMPTY", base_url=API_BASE)

INSTRUCT_TEMPLATE = "请根据以下句子和给定的关系类型生成新的标注样例。关系类型:{relation}\n句子:{text}\n返回JSON格式: {\"text\":..., \"triple\": [...] }"


def augment_file(input_file: Path, output_file: Path, relation: str):
    data = json.load(open(input_file))
    augmented = []
    for item in data:
        prompt = INSTRUCT_TEMPLATE.format(relation=relation, text=item["text"])
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
        )
        aug = resp.choices[0].message.content.strip()
        augmented.append({"origin": item, "augmented": aug})
    json.dump(augmented, open(output_file, "w"), ensure_ascii=False, indent=2)


def main():
    base = Path("dataset")
    # InstructIE
    augment_file(base / "InstructIE" / "train_zh.json", base / "InstructIE" / "train_zh_aug.json", "relation")
    augment_file(base / "InstructIE" / "train_en.json", base / "InstructIE" / "train_en_aug.json", "relation")
    # IEPile
    augment_file(base / "iepile" / "train.json", base / "iepile" / "train_aug.json", "relation")


if __name__ == "__main__":
    main()
