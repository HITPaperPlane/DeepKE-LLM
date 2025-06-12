import argparse
import json
import random
from typing import List, Dict

def load_data(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    return data

def save_jsonl(data: List[Dict], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            json.dump(d, f, ensure_ascii=False)
            f.write('\n')


def format_sample(sample: Dict) -> Dict:
    return {
        "input": sample.get("input", ""),
        "prompt": sample.get("instruction", ""),
        "answer": sample.get("output", "")
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='raw dataset path')
    parser.add_argument('--train_out', required=True, help='output training file')
    parser.add_argument('--eval_out', required=True, help='output evaluation file')
    parser.add_argument('--eval_ratio', type=float, default=0.2, help='ratio for evaluation set')
    args = parser.parse_args()

    data = load_data(args.input)
    random.shuffle(data)
    n_eval = int(len(data) * args.eval_ratio)
    eval_data = [format_sample(d) for d in data[:n_eval]]
    train_data = [format_sample(d) for d in data[n_eval:]]

    save_jsonl(train_data, args.train_out)
    save_jsonl(eval_data, args.eval_out)

if __name__ == '__main__':
    main()
