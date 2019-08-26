# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import torch.utils.data
import torch.nn.functional as F
import numpy as np

import egg.core as core
from egg.zoo.language_bottleneck.explain.data import LangData
from egg.zoo.language_bottleneck.explain.archs import Explainer, Masker, Game
from egg.zoo.language_bottleneck.explain.illustrate import CallbackEvaluator


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
    parser.add_argument('--entropy_coeff', type=float, default=1e-2,
                        help="")
    parser.add_argument('--prediction_mask', type=str, default='?xxxxxxx')
    parser.add_argument('--source_mask', type=str, default=None)
    parser.add_argument('--language', type=str, default='vocab8_language_1.txt')
    parser.add_argument('--preference_x', type=float, default=2)

    parser.add_argument('--n_bits', type=int, default=8)
    parser.add_argument('--bit_x', type=int, default=0)

    args = core.init(arg_parser=parser, params=params)
    assert len(args.prediction_mask) == 8

    assert 0 <= args.bit_x < args.n_bits

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

def main(params):
    opts = get_params(params)
    device = opts.device

    prediction_mask = ''.join(['?'] * 8)
    source_mask = (''.join(['?'] * 9 + ['x'])) if not opts.source_mask else opts.source_mask

    train_data = LangData(opts.language, scale_factor=100, transpose=True, prediction_mask=prediction_mask, source_mask=source_mask)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=opts.batch_size)
    test_data = LangData(opts.language, scale_factor=1, transpose=True, prediction_mask=prediction_mask, source_mask=source_mask)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=1)

    n_bits = opts.n_bits
    max_len = opts.max_len
    vocab_size = opts.vocab_size

    for _ in range(n_bits - 1):
        print('# starting with a mask', source_mask)
        explainer = Explainer(vocab_size=vocab_size, max_len=max_len, n_bits=1)
        adv_explainer = Explainer(vocab_size=vocab_size, max_len=max_len, n_bits=n_bits)
        masker = Masker(vocab_size=vocab_size, max_len=max_len, prior=opts.prior, mask=source_mask)

        game = Game(masker, explainer, adv_explainer, opts.bit_x, opts.coeff, opts.preference_x)

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

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])