# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import json
import numpy as np
import pathlib

from knockknock import slack_sender
import torch
import torch.nn.functional as F
import torch.nn as nn

import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.zoo.multitask.features import OneHotLoader
from egg.zoo.multitask.archs import Sender, Receiver, SenderTaskEmbedding
from egg.zoo.multitask.metrics import EvaluateAccuracy, EvaluateTopSim
from egg.zoo.multitask.util import compute_binomial, get_output_file


webhook_url = ''
with open('/private/home/rdessi/stuff_egg/knockknock_key.txt') as f:
    webhook_url = f.readlines()[0]

assert webhook_url


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_attributes', type=int, default=10,
                        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument('--baseline', type=str, choices=['no', 'mean', 'builtin'], default='mean')
    parser.add_argument('--n_values', type=int, default=10,
                        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument('--samples_per_epoch', type=int, default=5e5,
                        help='Number of batches per epoch (default: 5e5)')

    parser.add_argument('--shuffle_train_data', default=False, action='store_true')
    parser.add_argument('--n_validation_samples', type=int, default=5000)
    parser.add_argument('--val_batch_size', type=int, default=500)

    parser.add_argument('--sender_hidden', type=int, default=10,
                        help='Size of the hidden layer of Sender (default: 10)')

    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')
    parser.add_argument('--receiver_num_layers', type=int, default=1,
                        help='Number hidden layers of receiver. Only in reinforce (default: 1)')
    parser.add_argument('--sender_num_layers', type=int, default=1,
                        help='Number hidden layers of receiver. Only in reinforce (default: 1)')
    parser.add_argument('--sender_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Sender (default: 10)')
    parser.add_argument('--receiver_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Receiver (default: 10)')

    parser.add_argument('--sender_cell', type=str, default='rnn',
                        help='Type of the cell used for Sender {rnn, gru, lstm, transformer} (default: rnn)')
    parser.add_argument('--receiver_cell', type=str, default='rnn',
                        help='Type of the model used for Receiver {rnn, gru, lstm, transformer} (default: rnn)')

    parser.add_argument('--sender_entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient for Sender (default: 1e-1)')
    parser.add_argument('--receiver_entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient for Receiver (default: 1e-1)')

    parser.add_argument('--early_stopping_thr', default=0.9, type=float,
                        help='stop training when accuracy reacehs  early_stopping_thr')

    parser.add_argument('--multitask', default=False, action='store_true',
                        help='if set a multitask referential game is played')
    parser.add_argument('--same_task', default=False, action='store_true',
                        help='if set a multitask referential game with the same task is played')

    parser.add_argument('--pdb', default=False, action='store_true',
                        help='if set, run with pdb enabled')
    args = core.init(parser, params)

    return args


class Loss(nn.Module):
    def __init__(self, n_attributes, n_values, skip_attributes=None):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        # empty list means compute loss on all attributes
        self.skip_attributes = [] if not skip_attributes else skip_attributes

        for attr in self.skip_attributes:
            assert attr >= 0 and attr < self.n_attributes

    def forward(self, sender_input, _message, _receiver_input, receiver_output, _labels):
        batch_size = sender_input.shape[0]
        sender_input = sender_input.view(batch_size, self.n_attributes, self.n_values)
        receiver_output = receiver_output.view(batch_size, self.n_attributes, self.n_values)

        receiver_output_loss = receiver_output.view(batch_size * self.n_attributes, self.n_values)
        labels = sender_input.view(batch_size * self.n_attributes, self.n_values).argmax(dim=-1)
        for attr in self.skip_attributes:
            labels[attr::self.n_attributes] = -100  # -100 is default cross_entropy_loss ignore_index value
        loss = F.cross_entropy(receiver_output_loss, labels, reduction="none").view(batch_size, self.n_attributes).mean(dim=-1)

        n_attr = self.n_attributes - len(self.skip_attributes)
        s_inp = sender_input.argmax(dim=-1)
        for idx in  self.skip_attributes:
            # setting argmax idx to -1 so we can ignore some attributes
            s_inp[:, idx] = -1

        correct_samples = (receiver_output.argmax(dim=-1) == s_inp).detach()
        acc = (torch.sum(correct_samples, dim=-1) == n_attr).float().mean()
        #soft_acc = torch.div(torch.sum(correct_samples.float(), dim=-1), n_attr).mean()

        return loss, {'acc': acc}  #, 'soft_acc': soft_acc}


def loss_wrapper(n_attributes, n_values):
    def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
        batch_size = sender_input.size(0)
        sender_input = sender_input.view(batch_size, n_attributes, n_values)

        receiver_output = receiver_output.view(batch_size, n_attributes, n_values)

        acc = (torch.sum((receiver_output.argmax(dim=-1) == sender_input.argmax(dim=-1)).detach(), dim=1) == n_attributes).float().mean()

        receiver_output = receiver_output.view(batch_size * n_attributes, n_values)
        labels = sender_input.argmax(dim=-1).view(batch_size * n_attributes)
        loss = F.cross_entropy(receiver_output, labels, reduction="none").view(batch_size, n_attributes).mean(dim=-1)

        return loss, {'acc': acc}

    return loss


@slack_sender(webhook_url=webhook_url, channel="knockknock", send_notification=True)
def main(params):
    opts = get_params(params)
    print(opts, flush=True)
    device = opts.device
    if opts.pdb:
        import pdb; pdb.set_trace()

    dataloader = OneHotLoader(n_attributes=opts.n_attributes,
                                    n_values=opts.n_values,
                                    samples_per_epoch=opts.samples_per_epoch,
                                    batch_size=opts.batch_size,
                                    val_batch_size=opts.val_batch_size,
                                    n_validation_samples=opts.n_validation_samples,
                                    shuffle_train_data=opts.shuffle_train_data,
                                    seed=opts.random_seed)

    train_it, val_it = dataloader.get_train_iterator(), dataloader.get_validation_iterator()

    features = opts.n_attributes*opts.n_values

    num_tasks = 1  # starting from 1 counting the all attributes tasks
    if opts.multitask:
        for comb in range(1, opts.n_attributes):
            num_tasks += compute_binomial(opts.n_attributes, comb)

    sender = core.RnnSenderReinforce(sender,
                               opts.vocab_size, opts.sender_embedding, sender_embed_dim,
                               cell=opts.sender_cell, max_len=opts.max_len, num_layers=opts.sender_num_layers)

    receivers = [core.RnnReceiverDeterministic(Receiver(n_features=features, n_hidden=opts.receiver_hidden),
                                               opts.vocab_size,
                                               opts.receiver_embedding,
                                               opts.receiver_hidden,
                                               cell=opts.receiver_cell,
                                               num_layers=opts.receiver_num_layers) for _ in range(num_tasks)]

    skip_attributes_list = [None]
    if opts.multitask:
        if opts.same_task:
            skip_attributes_list * num_tasks
        else:
            attr_list = list(range(opts.n_attributes))
            for attr_n in range (1, opts.n_attributes):
                skip_attributes_list.extend(map(lambda x: list(x), list(itertools.combinations(attr_list, attr_n))))

    losses = [Loss(n_attributes=opts.n_attributes, n_values=opts.n_values, skip_attributes=attributes) for attributes in skip_attributes_list]

    game_mechanism = core.CommunicationRnnReinforce(sender_entropy_coeff=opts.sender_entropy_coeff,
                                                    receiver_entropy_coeff=0.0, length_cost=0.0, baseline_type=baseline)

    assert len(senders) == len(receivers) and len(receivers) == len(losses)
    agent_loss_sampler = FullSweepAgentSampler(senders, receivers, losses)
    game = PopulationGame(game_mechanism, agent_loss_sampler)
    optimizer = torch.optim.Adam(game.parameters(), lr=opts.lr)

    callbacks = [core.ConsoleLogger(as_json=True, print_train_loss=True, output_file=output_file),
                 core.EarlyStopperAccuracy(threshold=opts.early_stopping_thr, validation=False),
                 EvaluateAccuracy(games, val_it, 'test_set', 'acc',  opts.device, output_file, opts.task_embedding)]

    trainer = core.Trainer(games=game, optimizers=optimizer, train_data=train_it, validation_data=None, callbacks=callbacks)
    trainer.train(n_epochs=opts.n_epochs, task_embedding=opts.task_embedding)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
