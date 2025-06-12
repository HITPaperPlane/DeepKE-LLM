import json
import argparse
from openai import OpenAI


def build_triples(client, text, instruction):
    messages = [
        {"role": "system", "content": "你是知识图谱构建助手"},
        {"role": "user", "content": instruction + text}
    ]
    response = client.chat.completions.create(model="novel", messages=messages)
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--api_base', default='http://192.168.11.218:8192/v1')
    args = parser.parse_args()
    client = OpenAI(api_key='EMPTY', base_url=args.api_base)
    data = json.load(open(args.input))
    results = []
    for item in data:
        triples = build_triples(client, item['text'], '抽取知识图谱三元组：')
        results.append({'id': item.get('id'), 'triples': triples})
    json.dump(results, open(args.output, 'w'), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
