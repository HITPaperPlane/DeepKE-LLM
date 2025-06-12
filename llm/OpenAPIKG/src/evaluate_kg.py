import argparse
import json
import re
from typing import List, Tuple, Dict


def load_data(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def parse_triples(text: str) -> List[Tuple[str, str, str]]:
    pattern = re.compile(r'\(([^,]+),([^,]+),([^\)]+)\)')
    triples = []
    for m in pattern.finditer(text):
        triples.append(tuple(t.strip() for t in m.groups()))
    return triples

def evaluate(refs: List[Dict], preds: List[Dict]) -> Tuple[float, float, float]:
    assert len(refs) == len(preds)
    correct = 0
    pred_total = 0
    ref_total = 0
    for r, p in zip(refs, preds):
        ref_triples = set(parse_triples(r.get('answer', '')))
        pred_triples = set(parse_triples(p.get('prediction', '')))
        correct += len(ref_triples & pred_triples)
        pred_total += len(pred_triples)
        ref_total += len(ref_triples)
    precision = correct / pred_total if pred_total else 0
    recall = correct / ref_total if ref_total else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', required=True, help='ground truth file')
    parser.add_argument('--pred', required=True, help='prediction file')
    args = parser.parse_args()

    refs = load_data(args.ref)
    preds = load_data(args.pred)
    p, r, f1 = evaluate(refs, preds)
    print(f'Precision: {p:.4f}')
    print(f'Recall: {r:.4f}')
    print(f'F1: {f1:.4f}')

if __name__ == '__main__':
    main()
