import json
from pathlib import Path
from openai import OpenAI

API_BASE = "http://192.168.11.218:8192/v1"
MODEL_NAME = "novel"

client = OpenAI(api_key="EMPTY", base_url=API_BASE)

PROMPT = "指令:{instruction}\n文本:{text}\n请抽取其中的三元组, 按JSON数组格式返回"


def construct(input_file: Path, output_file: Path, instruction: str):
    data = json.load(open(input_file))
    outputs = []
    for item in data:
        prompt = PROMPT.format(instruction=instruction, text=item["input"])
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )
        outputs.append({"id": item.get("id", ""), "output": resp.choices[0].message.content.strip()})
    json.dump(outputs, open(output_file, "w"), ensure_ascii=False, indent=2)


def main():
    construct(Path("dataset/InstructIE/test_zh.json"), Path("kg_output.json"), "抽取文本中的三元组")


if __name__ == "__main__":
    main()
