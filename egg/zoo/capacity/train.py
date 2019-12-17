# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from egg.zoo.capacity.dataset import SphereData
from egg.zoo.capacity.archs import PositionalSender, Receiver, RotatorLenses

import json
import argparse
import torch.utils.data
import torch.nn.functional as F
import egg.core as core
from torch.utils.data import DataLoader


def get_params(params):
    print(params)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_examples', type=int, default=1000,
                        help='Number of examples seen in an epoch (default: 1000)')
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')
    parser.add_argument('--theta', type=float, default=None)

    parser.add_argument('--receiver_cell', type=str, default='rnn')
    parser.add_argument('--receiver_emb', type=int, default=10,
                        help='Size of the embeddings of Receiver (default: 10)')

    args = core.init(arg_parser=parser, params=params)
    return args


def diff_loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    loss = F.mse_loss(receiver_output, sender_input)
    return loss, {}


def main(params):
    import math

    opts = get_params(params)
    print(opts)#json.dumps(vars(opts)))

    device = opts.device

    train_data = SphereData(n_points=opts.n_examples, n_dim=2)
    train_loader = DataLoader(train_data, batch_size=opts.batch_size)

    test_data = SphereData(n_points=opts.n_examples, n_dim=2)
    test_loader = DataLoader(train_data, batch_size=opts.batch_size)

    if opts.theta is None:
        lense = None
    else:
        lense = RotatorLenses(theta=opts.theta * math.pi)

    sender = PositionalSender(vocab_size=opts.vocab_size, lense=lense)
    receiver = Receiver(n_hidden=opts.receiver_hidden, n_dim=2)

    receiver = core.RnnReceiverDeterministic(
                receiver, opts.vocab_size, opts.receiver_emb, opts.receiver_hidden, cell=opts.receiver_cell)
    game = core.SenderReceiverRnnReinforce(sender, receiver, diff_loss, receiver_entropy_coeff=0.0, sender_entropy_coeff=0.0)
       
    optimizer = core.build_optimizer(game.parameters())
    loss = game.loss

    #intervention = CallbackEvaluator(test_loader, device=device, is_gs=opts.mode == 'gs', loss=loss, var_length=opts.variable_length,
    #                                 input_intervention=True)

    trainer = core.Trainer(
        game=game, optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=[core.ConsoleLogger(as_json=True)]) #, EarlyStopperAccuracy(opts.early_stopping_thr)])#, intervention])

    trainer.train(n_epochs=opts.n_epochs)

    #if opts.dump_language:
    #    dump(game, test_loader, device, is_gs=opts.mode ==
    #         'gs', is_var_length=opts.variable_length)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
