import torch
from data import LangData
from train import get_language_opts
import pathlib
import json

def find_minimal_pairs(dataset):
    input2utterance = {}

    for bit_vector, utterance in zip(dataset.inputs, dataset.codes):
        as_tuple = tuple(bit_vector.tolist())
        input2utterance[as_tuple] = utterance

    paired_utterances = []

    for k, u in input2utterance.items():
        k = list(k)

        for i in range(len(k)):
            if k[i] == 0:
                continue

            k[i] = 0
            paired_key = tuple(k)
            paired_utt = input2utterance[paired_key]
            paired_utterances.append((paired_utt, u, i))

            k[i] = 1

    return paired_utterances

def find_diff_positions(pairs):
    max_len = pairs[0][0].size(0)
    n_bits = max(j for _1, _2, j in pairs) + 1

    feature_changed_positions = torch.zeros((n_bits, max_len))
    total = torch.zeros(n_bits)

    for a, b, feature_id in pairs:
        assert a.size(0) == b.size(0)

        total[feature_id] += 1
        feature_changed_positions[feature_id, :] += (a != b).float()

    return (feature_changed_positions.t() / total).t()

def min_max(table):
    n = table.size(0)
    k = table.size(1)

    distances = []

    for i in range(n):
        for j in range(i+1, n):
            d = (table[i, :] - table[j, :]).abs().sum()
            distances.append((d / k, i, j))

    distances.sort()

    return distances[0], distances[-1]

def iterate_dir(root):
    root = pathlib.Path(root).absolute()

    collected = {}
    for fname in root.glob('*'):
        yield fname

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', action='append')
    parser.add_argument('--dir', action='append')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    assert args.output
    languages = []
    if args.dir:
        for d in args.dir:
            languages.extend(iterate_dir(d))

    if args.language:
        languages.extend([pathlib.Path(p) for p in args.language])

    results = []

    for l in languages:
        print(l)
        data = LangData(l)

        inp, code = data[0]
        if len(inp) != 8: 
            print('non 8 bit language, skipping')
            continue

        pairs = find_minimal_pairs(data)

        change_freq = find_diff_positions(pairs)
        (min_d, min_i, min_j), (max_d, max_i, max_j) = min_max(change_freq)

        results.append({
            'name': l.stem,
            'change_freq': change_freq.tolist(),
            'min': (min_d.item(), min_i, min_j),
            'max': (max_d.item(), max_i, max_j)
        })

    results = json.dumps(results, sort_keys=True, indent=2)

    with open(args.output, 'w') as f:
        f.write(results)
        f.write('\n')

