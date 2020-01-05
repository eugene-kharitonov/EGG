# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from egg.zoo.capacity.dataset import SphereData
from egg.zoo.capacity.archs import PositionalSender, Receiver, RotatorLenses, PlusOneWrapper, SubspaceSwapLenses, Mixer2d, WrapperModule

import json
import argparse
import torch.utils.data
import torch.nn.functional as F
import egg.core as core
from torch.utils.data import DataLoader
import math


def get_params(params):
    print(params)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_examples', type=int, default=1000,
                        help='Number of examples seen in an epoch (default: 1000)')
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')
    parser.add_argument('--receiver_cell', type=str, default='rnn')
    parser.add_argument('--receiver_emb', type=int, default=10,
                        help='Size of the embeddings of Receiver (default: 10)')
    parser.add_argument('--no_mixer', action='store_true')
    parser.add_argument('--inner_layers', type=int, default=-1)
    parser.add_argument('--mixer_epochs', type=int, default=10)

    args = core.init(arg_parser=parser, params=params)
    assert args.inner_layers >= -1

    return args


def diff_loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    loss = F.mse_loss(receiver_output, sender_input)
    return loss, {}


def train_mixer(train_loader, mixer, unmixer, use_cuda, n_epochs):
    wrapper = WrapperModule(mixer, unmixer)

    if use_cuda:
        wrapper.cuda()

    params = wrapper.parameters()
    optimizer = torch.optim.Adam(params, lr=1e-2)
    
    for epoch in range(n_epochs):
        for points, _ in train_loader:
            n_batch = points.size(0)

            if use_cuda: points = points.cuda()

            optimizer.zero_grad()
            loss = wrapper(points)

            loss.backward()
            optimizer.step()
        print(f"# Mixer train: epoch {epoch}, loss {loss.detach()}")

def main(params):
    opts = get_params(params)
    print(opts)

    device = opts.device

    train_data = SphereData(n_points=opts.n_examples, n_dim=2)
    train_loader = DataLoader(train_data, batch_size=opts.batch_size)

    test_data = SphereData(n_points=opts.n_examples, n_dim=2)
    test_loader = DataLoader(train_data, batch_size=opts.batch_size)

    all_data = torch.utils.data.ConcatDataset([train_data, test_data])
    all_loader = DataLoader(all_data, batch_size=opts.batch_size)

    mixer = None#RotatorLenses(math.pi * 0.25)
    if not opts.no_mixer:
        mixer, unmixer = Mixer2d(inner_layers=opts.inner_layers), Mixer2d(inner_layers=opts.inner_layers)
        train_mixer(all_loader, mixer, unmixer, opts.cuda, n_epochs=opts.mixer_epochs)
        print(mixer)

    sender = PositionalSender(vocab_size=opts.vocab_size, lense=mixer)
    sender = PlusOneWrapper(sender)

    receiver = Receiver(n_hidden=opts.receiver_hidden, n_dim=2)

    receiver = core.RnnReceiverDeterministic(
                receiver, opts.vocab_size + 1,  # exclude eos = 0
                opts.receiver_emb, opts.receiver_hidden, cell=opts.receiver_cell)
    game = core.SenderReceiverRnnReinforce(sender, receiver, diff_loss, receiver_entropy_coeff=0.0, sender_entropy_coeff=0.0)
       
    optimizer = core.build_optimizer(game.parameters())
    loss = game.loss

    #intervention = CallbackEvaluator(test_loader, device=device, is_gs=opts.mode == 'gs', loss=loss, var_length=opts.variable_length,
    #                                 input_intervention=True)

    trainer = core.Trainer(
        game=game, optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=True)]) #, EarlyStopperAccuracy(opts.early_stopping_thr)])#, intervention])

    trainer.train(n_epochs=opts.n_epochs)

    #if opts.dump_language:
    #    dump(game, test_loader, device, is_gs=opts.mode ==
    #         'gs', is_var_length=opts.variable_length)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
