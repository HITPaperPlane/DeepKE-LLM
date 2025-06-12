import argparse
import json
import os
import requests
from typing import List, Dict
from tqdm import tqdm

def load_data(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def save_jsonl(data: List[Dict], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            json.dump(d, f, ensure_ascii=False)
            f.write('\n')


def call_api(api_url: str, api_key: str, prompt: str) -> str:
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    payload = {
        'messages': [{'role': 'user', 'content': prompt}]
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get('choices', [{}])[0].get('message', {}).get('content', '')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='evaluation dataset file')
    parser.add_argument('--output', required=True, help='prediction output file')
    parser.add_argument('--api_url', required=True, help='OpenAPI endpoint')
    parser.add_argument('--api_key', required=True, help='API key')
    args = parser.parse_args()

    samples = load_data(args.input)
    results = []
    for sample in tqdm(samples, desc='Generating'):
        prompt = f"{sample['prompt']}\n{sample['input']}"
        prediction = call_api(args.api_url, args.api_key, prompt)
        results.append({
            'input': sample['input'],
            'prompt': sample['prompt'],
            'prediction': prediction
        })

    save_jsonl(results, args.output)

if __name__ == '__main__':
    main()
