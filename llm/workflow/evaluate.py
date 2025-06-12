import json
import argparse
from collections import Counter


def load_gold(path):
    data = json.load(open(path))
    gold = {}
    for item in data:
        gold[item['id']] = set(tuple(t) for t in item.get('relation', []))
    return gold


def load_pred(path):
    data = json.load(open(path))
    pred = {}
    for item in data:
        triples = item['triples']
        if isinstance(triples, str):
            triples = [t.strip() for t in triples.split(')') if t.strip()]
            triples = set(tuple(x.strip('()').split(',')) for x in triples)
        else:
            triples = set(tuple(t) for t in triples)
        pred[item['id']] = triples
    return pred


def compute_f1(gold, pred):
    tp = fp = fn = 0
    for k in gold:
        g = gold.get(k, set())
        p = pred.get(k, set())
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True)
    parser.add_argument('--gold', required=True)
    args = parser.parse_args()
    gold = load_gold(args.gold)
    pred = load_pred(args.pred)
    p, r, f1 = compute_f1(gold, pred)
    print(f'Precision: {p:.4f} Recall: {r:.4f} F1: {f1:.4f}')


if __name__ == '__main__':
    main()
