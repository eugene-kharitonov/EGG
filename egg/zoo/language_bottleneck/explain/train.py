# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import torch.utils.data
import torch.nn.functional as F
import egg.core as core
from egg.zoo.language_bottleneck.explain.data import LangData
from egg.zoo.language_bottleneck.explain.archs import Explainer, Masker, Game
from egg.zoo.language_bottleneck.explain.illustrate import CallbackEvaluator


def get_params(params):
    print(params)
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=float, default=0.0)

    parser.add_argument('--lr_e', type=float, default=1e-3)
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

    args = core.init(arg_parser=parser, params=params)
    assert len(args.prediction_mask) == 8

    return args

def main(params):
    opts = get_params(params)
    device = opts.device

    prediction_mask = ''.join(['?'] * 8)
    source_mask = ''.join(['?'] * 10)

    train_data = LangData(opts.language, scale_factor=100, transpose=True, prediction_mask=prediction_mask, source_mask=source_mask)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=opts.batch_size)
    test_data = LangData(opts.language, scale_factor=1, transpose=True, prediction_mask=prediction_mask, source_mask=source_mask)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=1)

    n_bits = 8
    max_len = 10

    explainer = Explainer(vocab_size=8, max_len=max_len, n_bits=1)
    adv_explainer = Explainer(vocab_size=8, max_len=max_len, n_bits=7)
    masker = Masker(vocab_size=8, max_len=max_len, prior=opts.prior)

    game = Game(masker, explainer, adv_explainer, opts.coeff)

    optimizer = torch.optim.Adam(
        [
            dict(params=explainer.parameters(), lr=opts.lr_e),
            dict(params=adv_explainer.parameters(), lr=opts.lr_e),
            dict(params=masker.parameters(), lr=opts.lr_m)
        ])

    callback = CallbackEvaluator(test_loader, device)

    trainer = core.Trainer(
        game=game, optimizer=optimizer,
        train_data=train_loader, validation_data=test_loader,
        callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=True), callback])

    trainer.train(n_epochs=opts.n_epochs)
    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
