import json
import argparse
from openai import OpenAI


def augment_sample(client, prompt, instruction):
    messages = [
        {"role": "system", "content": "你是一个数据增强助手"},
        {"role": "user", "content": instruction + prompt}
    ]
    response = client.chat.completions.create(model="novel", messages=messages)
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to training json")
    parser.add_argument("--output", required=True, help="path to save augmented json")
    parser.add_argument("--api_base", default="http://192.168.11.218:8192/v1")
    args = parser.parse_args()
    client = OpenAI(api_key="EMPTY", base_url=args.api_base)
    # load dataset. Some files use jsonl format but keep `.json` suffix
    if args.input.endswith('.jsonl'):
        data = [json.loads(l) for l in open(args.input)]
    else:
        try:
            data = json.load(open(args.input))
        except json.JSONDecodeError:
            # fall back to json lines
            data = [json.loads(l) for l in open(args.input)]
    augmented = []
    for item in data:
        text = item.get("text", item.get("input", ""))
        inst = "请仿照给定样本风格重新编写一条新的训练样本。"
        new_text = augment_sample(client, text, inst)
        item["augmented_text"] = new_text
        augmented.append(item)
    json.dump(augmented, open(args.output, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
