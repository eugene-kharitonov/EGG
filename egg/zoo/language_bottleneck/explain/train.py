# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import pathlib

import egg.core as core
from egg.zoo.language_bottleneck.explain.data import LangData
from egg.zoo.language_bottleneck.explain.archs import Explainer, Masker, Game, ReverseGame
from egg.zoo.language_bottleneck.explain.archs import GsMinimalCoverGame
from egg.zoo.language_bottleneck.explain.illustrate import CallbackEvaluator

from egg.zoo.language_bottleneck.intervention import mutual_info, entropy

def get_params(params):
    print(params)
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=float, default=0.0)

    parser.add_argument('--lr_e', type=float, default=1e-3)
    parser.add_argument('--lr_a', type=float, default=1e-3)
    parser.add_argument('--lr_m', type=float, default=1e-4)
    parser.add_argument('--coeff', type=float, default=1e-1)

    parser.add_argument('--scaler', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=10,
                        help='')
    #parser.add_argument('--entropy_coeff', type=float, default=1e-2,
    #                    help="")
    parser.add_argument('--prediction_mask', type=str, default=None)
    parser.add_argument('--source_mask', type=str, default=None)

    parser.add_argument('--language', type=str, action='append', default=None)
    parser.add_argument('--languages', type=str, action='append', default=None)

    parser.add_argument('--preference_x', type=float, default=2)
    parser.add_argument('--output', type=str, default=None)

    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--mode', choices=['explain_symbol', 'intersection_game', 
                                           'minimal_support', 'information_gap'])

    args = core.init(arg_parser=parser, params=params)
    assert args.output

    return args

def pruned_mask(probs, old_mask):
    """
    >>> p = torch.tensor([0.5, 0.4, 0.3])
    >>> pruned_mask(p, "x??")
    'xx?'
    >>> pruned_mask(p, "???")
    'x??'
    """

    ranked = torch.argsort(probs, descending=True).tolist()
    old_mask = list(old_mask)
    for r in ranked:
        if old_mask[r] == '?':
            old_mask[r] = 'x'
            break
    return ''.join(old_mask)

def get_language_opts(path):
    with open(path, 'r') as f:
        h = f.readline()[1:]
        h = json.loads(h)
    return h

def iterate_dir(root):
    root = pathlib.Path(root).absolute()
    for fname in root.glob('*'):
        yield fname

def main(params):
    opts = get_params(params)
    device = opts.device
    output = open(opts.output, 'w')

    languages = []
    if opts.languages:
        for d in opts.languages:
            languages.extend(list(pathlib.Path(d).absolute().glob('*')))

    if opts.language:
        languages.extend([pathlib.Path(p) for p in opts.language])

    for language in languages:
        language_opts = get_language_opts(language)

        opts.max_len = language_opts['max_len']
        opts.n_bits = language_opts['n_bits']
        opts.vocab_size = language_opts['vocab_size']

        prediction_mask = ''.join(['?'] * opts.n_bits) if opts.prediction_mask is None else opts.prediction_mask
        source_mask = (''.join(['?'] * (opts.max_len - 1) + ['x'])) if opts.source_mask is None else opts.source_mask

        train_data = LangData(language, scale_factor=100, transpose=True, prediction_mask=prediction_mask, source_mask=source_mask)
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=opts.batch_size)
        test_data = LangData(language, scale_factor=1, transpose=True, prediction_mask=prediction_mask, source_mask=source_mask)
        test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=1)


        if opts.mode == 'intersection_game':
            train_utterance_to_bits(opts, train_loader, test_loader)
        if opts.mode == 'explain_symbol':
            try:
                result = quick_explain_symbol(opts, train_loader, test_loader)
            except:
                result = explain_symbol(opts, train_loader, test_loader)
        if opts.mode == 'minimal_support':
            result = minimal_support(opts, train_loader, test_loader)
        if opts.mode == 'information_gap':
            result = information_gap(opts, train_loader, test_loader)

        print(result)

        for k, v in result.items():
            language_opts[k] = v
        language_opts['name'] = language.stem
        language_opts = json.dumps(language_opts)

        output.write(language_opts)
        output.write('\n')

    output.close()
    core.close()


def quick_explain_symbol(opts, train_loader, test_loader):

    def get_bit_utterance(mask_bits, tgt, loader):
        x, y = [], []
        ind = [i for i, m in enumerate(mask_bits) if m == '?']

        for utterance, meaning in loader:
            x.append(meaning[:, ind])
            y.append(utterance[:, tgt])
        return torch.cat(x, dim=0), torch.cat(y, dim=0)

    n_bits = opts.n_bits
    max_len = opts.max_len
    vocab_size = opts.vocab_size
    device = opts.device

    for target in range(max_len - 1):
        for b in range(n_bits):
            prediction_mask = ['?'] * n_bits
            prediction_mask[b] = 'x'
            prediction_mask = ''.join(prediction_mask)

            meaning, utt = get_bit_utterance(prediction_mask, target, test_loader)
            symbol_entr = entropy(utt)
            meaning_utt_info = mutual_info(meaning, utt)
            print(f'# symbol entropy {symbol_entr}, info {meaning_utt_info}')
            if meaning_utt_info / symbol_entr >= 0.99:
                raise ValueError()
                #assert False, "Actually position {target} does not need bit {b}"

    return {'ave_bits_needed': n_bits}

def explain_symbol(opts, train_loader, test_loader):
    def get_bit_utterance(mask_bits, tgt, loader):
        x, y = [], []
        ind = [i for i, m in enumerate(mask_bits) if m == '?']

        for utterance, meaning in loader:
            x.append(meaning[:, ind])
            y.append(utterance[:, tgt])
        return torch.cat(x, dim=0), torch.cat(y, dim=0)

    n_bits = opts.n_bits
    max_len = opts.max_len
    vocab_size = opts.vocab_size
    device = opts.device
    bits_required = 0
    normalizer = 0.0

    for target in range(max_len - 1):
        prediction_mask = ''.join(['?'] * n_bits)
        for b in range(n_bits):
            print('# starting with a mask', prediction_mask)
            meaning, utt = get_bit_utterance(prediction_mask, target, test_loader)
            symbol_entr = entropy(utt)
            meaning_utt_info = mutual_info(meaning, utt)
            print(f'# symbol entropy {symbol_entr}, info {meaning_utt_info}')
            if symbol_entr == 0 or meaning_utt_info / symbol_entr < 0.99:
                break

            game = ReverseGame(target_position=target, n_bits=n_bits, vocab_size=vocab_size, mask=prediction_mask, prior=opts.prior, l=opts.coeff)
            stopper = core.EarlyStopperAccuracy(threshold=0.99, field_name="acc_X")

            optimizer = torch.optim.Adam(game.parameters(), opts.lr)
            callback = CallbackEvaluator(test_loader, device)

            trainer = core.Trainer(
                game=game, optimizer=optimizer,
                train_data=train_loader, validation_data=test_loader,
                callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=True), callback, stopper])

            trainer.train(n_epochs=opts.n_epochs)

            probs = game.masker.prob_mask_logits.detach().sigmoid()
            prediction_mask = pruned_mask(probs, prediction_mask)
        if symbol_entr > 0:
            bits_required += n_bits - b + 1
            normalizer += 1

    return {'ave_bits_needed': bits_required / normalizer}


def minimal_support(opts, train_loader, test_loader):

    def get_bit_utterance(mask_utt, loader):
        x, y = [], []
        ind = [i for i, m in enumerate(mask_utt) if m == '?']

        for utterance, meaning in loader:
            x.append(meaning)
            y.append(utterance[:, ind])
        return torch.cat(x, dim=0), torch.cat(y, dim=0)

    n_bits = opts.n_bits
    max_len = opts.max_len
    vocab_size = opts.vocab_size
    device = opts.device

    source_mask = (''.join(['?'] * (max_len - 1) + ['x'])) if not opts.source_mask else opts.source_mask
    prev_mask = source_mask

    for r in range(max_len - 2):
        print('# starting with a mask', source_mask)
        meaning, utt = get_bit_utterance(source_mask, test_loader)
        meaning_entr = entropy(meaning)
        meaning_utt_info = mutual_info(meaning, utt)
        print(f'# meaning entropy {meaning_entr}, info {meaning_utt_info}')
        if meaning_utt_info / meaning_entr < 0.99:
            break

        game = GsMinimalCoverGame(vocab_size, max_len, n_bits, opts.coeff)
        stopper = core.EarlyStopperAccuracy(threshold=0.99, field_name="acc_X_mean")

        optimizer = torch.optim.Adam(game.parameters())

        callback = CallbackEvaluator(test_loader, device)

        trainer = core.Trainer(
            game=game, optimizer=optimizer,
            train_data=train_loader, validation_data=test_loader,
            callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=False), callback, stopper])

        trainer.train(n_epochs=opts.n_epochs)

        probs = game.masker.prob_mask_logits.detach().sigmoid()
        prev_mask = source_mask
        source_mask = pruned_mask(probs, source_mask)

    table = {'x': '?', '?': 'x'}
    prev_mask_inverse = ''.join([table[x] for x in prev_mask])

    meaning, utt = get_bit_utterance(prev_mask_inverse, test_loader)
    reverse_info = mutual_info(meaning, utt)

    return {'min_positions': r, 'meaning_utt_info': meaning_utt_info, 'mask': source_mask, 'reverse_info': reverse_info}

def train_utterance_to_bits(opts, train_loader, test_loader):
    n_bits = opts.n_bits
    max_len = opts.max_len
    vocab_size = opts.vocab_size
    device = opts.device

    source_mask = (''.join(['?'] * (max_len - 1) + ['x'])) if not opts.source_mask else opts.source_mask

    for _ in range(max_len - 2):
        print('# starting with a mask', source_mask)
        masker = Masker(replace_id=vocab_size, max_len=max_len, prior=opts.prior, mask=source_mask)
        explainer = Explainer(vocab_size=vocab_size, max_len=max_len, n_bits=1)
        adv_explainer = Explainer(vocab_size=vocab_size, max_len=max_len, n_bits=n_bits)
        game = Game(masker, explainer, adv_explainer, opts.target, opts.coeff, opts.preference_x)

        optimizer = torch.optim.Adam(
            [
                dict(params=explainer.parameters(), lr=opts.lr_e),
                dict(params=adv_explainer.parameters(), lr=opts.lr_a),
                dict(params=masker.parameters(), lr=opts.lr_m)
            ])

        callback = CallbackEvaluator(test_loader, device)

        trainer = core.Trainer(
            game=game, optimizer=optimizer,
            train_data=train_loader, validation_data=test_loader,
            callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=False), callback])

        trainer.train(n_epochs=opts.n_epochs)

        probs = masker.prob_mask_logits.detach().sigmoid()
        source_mask = pruned_mask(probs, source_mask)


def information_gap(opts, train_loader, test_loader):

    def get_bit_utterance(i, j, loader):
        x, y = [], []

        for utterance, meaning in loader:
            x.append(meaning[:, i])
            y.append(utterance[:, j])

        return torch.cat(x, dim=0), torch.cat(y, dim=0)


    n_bits = opts.n_bits
    max_len = opts.max_len
    vocab_size = opts.vocab_size
    device = opts.device

    gaps = torch.zeros(max_len)
    non_constant_positions = 0.0

    for j in range(max_len):
        symbol_mi = []
        h_j = None
        for i in range(n_bits):
            x, y = get_bit_utterance(i, j, test_loader)
            info =  mutual_info(x, y)
            symbol_mi.append(info)
            
            if h_j is None:
                h_j = entropy(y)

        symbol_mi.sort(reverse=True)

        if h_j > 0.0:
            gaps[j] = (symbol_mi[0] - symbol_mi[1]) / h_j
            non_constant_positions += 1

    score = gaps.sum() / non_constant_positions

    return dict(
        mean_information_gap=score.item()
    )

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
