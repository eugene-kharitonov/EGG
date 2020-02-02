# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random

from egg.zoo.capacity.dataset import SphereData, AttributeValueData
from egg.zoo.capacity.archs import (Receiver, MixerDiscrete, DiscreteWrapperModule, ArithmeticSender2, ReceiverRandomized,
    IdentitySender, LinearReceiver, NonLinearReceiver, ArithmeticSender, PositionalScrambler, HashSender, RandomizedIdentitySender,
    MultiHashSender, VocabScrambler, UnfactorizedHashSender, RandomizedHashSender)

from egg.zoo.capacity.intervention import Evaluator, Metrics
from .dataset import one_hotify

SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

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
    parser.add_argument('--receiver_layers', type=int, default=-1)
    parser.add_argument('--cell_layers', type=int, default=1)

    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')
    parser.add_argument('--receiver_cell', type=str, default='rnn')
    parser.add_argument('--receiver_emb', type=int, default=10,
                        help='Size of the embeddings of Receiver (default: 10)')

    parser.add_argument('--mixers', type=int, default=1)
    parser.add_argument('--n_a', type=int, default=2)
    parser.add_argument('--n_v', type=int, default=10)

    parser.add_argument('--language', type=str, choices= \
        ['identity', 'scrambled-identity', 'arithmetic', 'scrambled-arithmetic', 'hash', 'scrambled-hash', 'arithmetic2',
        'unfactorized-hash', 'random-hash', 'scrambled-random-hash', 'scrambled-random-identity', 'random-identity'])
    parser.add_argument('--base', type=int, default=2)

    parser.add_argument('--scramble_positions', choices=['0', '1'])
    parser.add_argument('--predict', type=int, default=-1)
    parser.add_argument('--scramble_vocab', action='store_true')
    parser.add_argument('--loss_type', choices=['autoenc', 'mixed', 'linear', 'diff-1'], default='autoenc')

    args = core.init(arg_parser=parser, params=params)

    assert args.base >= 2
    return args


class _DiffLoss(torch.nn.Module):
    def __init__(self, n_attributes, n_values):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values

    def forward(self, sender_input, _message, _receiver_input, receiver_output, _labels):
        batch_size = sender_input.size(0)
        receiver_output = receiver_output.view(batch_size, self.n_attributes, self.n_values)

        acc = (torch.sum((receiver_output.argmax(dim=-1) == sender_input).detach(), dim=1) == self.n_attributes).float().mean()
        acc_or = (receiver_output.argmax(dim=-1) == sender_input).float().mean()

        receiver_output = receiver_output.view(batch_size * self.n_attributes, self.n_values)
        labels = sender_input.view(batch_size * self.n_attributes)
        loss = F.cross_entropy(receiver_output, labels, reduction="none").view(batch_size, self.n_attributes).mean(dim=-1)

        return loss, {'acc': acc, 'acc_or': acc_or}

class DiffLoss(torch.nn.Module):
    def __init__(self, n_attributes, n_values, loss_type, predict=-1):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.predict = predict if predict > 0 else n_attributes

        self.mixer = MixerDiscrete(n_attributes=n_attributes, n_values=n_values)
        self.loss_type = loss_type
        self.a, self.b, self.c, self.d = random.sample(SMALL_PRIMES, 4)

    def forward(self, sender_input, _message, _receiver_input, receiver_output, _labels):
        batch_size = sender_input.size(0)
        receiver_output = receiver_output.view(batch_size, self.n_attributes, self.n_values)

        receiver_output = receiver_output[:, :self.predict, :].contiguous()
        sender_input = sender_input[:, :self.predict].contiguous()

        if self.loss_type == 'mixed':
            sender_input = self.mixer(sender_input)
        elif self.loss_type == 'linear':
            a, b, c, d = self.a, self.b, self.c, self.d  
            left = (a * sender_input[:, 0] + c * sender_input[:, 1]).fmod(self.n_values)
            right = (b * sender_input[:, 0] + d * sender_input[:, 1]).fmod(self.n_values)
            sender_input[:, 0], sender_input[:, 1] = left, right
        elif self.loss_type == 'autoenc':
            pass
        else:
            assert False

        acc = (torch.sum((receiver_output.argmax(dim=-1) == sender_input).detach(), dim=1) == self.predict).float().mean()
        acc_or = (receiver_output.argmax(dim=-1) == sender_input).float().mean()

        receiver_output = receiver_output.view(batch_size * self.predict, self.n_values)
        labels = sender_input.view(batch_size * self.predict)

        loss = F.cross_entropy(receiver_output, labels, reduction="none").view(batch_size, self.predict).mean(dim=-1)

        return loss, {'acc': acc, 'acc_or': acc_or}


class DiffLoss1(torch.nn.Module):
    def __init__(self, n_attributes, n_values, loss_type):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values

    def forward(self, sender_input, _message, _receiver_input, receiver_output, _labels):
        batch_size = sender_input.size(0)
        receiver_output = receiver_output.view(batch_size, self.n_attributes, self.n_values)[:, 0, :]

        acc = (receiver_output.argmax(dim=-1) == sender_input[:, 0]).detach().float().mean()

        labels = sender_input[:, 0]
        loss = F.cross_entropy(receiver_output, labels, reduction="mean")

        return loss, {'acc': acc}

def main(params):
    opts = get_params(params)
    print(opts)

    device = opts.device

    n_a, n_v = opts.n_a, opts.n_v 
    assert opts.vocab_size > n_v

    train_data = AttributeValueData(n_attributes=n_a, n_values=n_v, mul=1)
    train_loader = DataLoader(train_data, batch_size=opts.batch_size)

    if opts.language == 'identity' or opts.language == 'scrambled-identity':
        sender = IdentitySender(n_attributes=n_a, n_values=n_v)
        opts.base = n_v
    elif opts.language == 'arithmetic' or opts.language == 'scrambled-arithmetic':
        sender = ArithmeticSender(n_a, n_v, base=opts.base)
    elif opts.language == 'arithmetic2':
        sender = ArithmeticSender2(n_a, n_v, base=opts.base)
    elif opts.language == 'hash' or opts.language == 'scrambled-hash':
        sender = HashSender(n_v, opts.base)
    elif opts.language == 'multi-hash':
        sender = MultiHashSender(n_a, n_v, opts.base)
    elif opts.language == 'unfactorized-hash':
        sender = UnfactorizedHashSender(n_a, n_v, opts.base)
    elif opts.language == 'random-hash' or opts.language == 'scrambled-random-hash':
        sender = RandomizedHashSender(n_v, opts.base)
    elif opts.language in ['scrambled-random-identity', 'random-identity']:
        sender = RandomizedIdentitySender(n_values=n_v)
    else:
        assert False

    mixer = None
    if opts.mixers > 0:
        mixer = torch.nn.Sequential(
            *(MixerDiscrete(n_attributes=n_a, n_values=n_v) for _ in range(opts.mixers))
        )
        sender = torch.nn.Sequential(mixer, sender)

    if opts.scramble_positions == 1:
        scrambler = PositionalScrambler()
        sender = torch.nn.Sequential(sender, scrambler)

    if 'scrambled-' in opts.language:
        scrambler = VocabScrambler(opts.base)
        sender = torch.nn.Sequential(sender, scrambler)

    """for k, _ in train_data:
        k = k.unsqueeze(0)
        print(k, sender(k)[0])
    exit(0)"""

    if opts.receiver_cell == 'transformer':
            receiver = Receiver(n_hidden=opts.receiver_emb, n_dim=n_a * n_v, inner_layers=opts.receiver_layers)
            receiver = core.TransformerReceiverDeterministic(receiver, 
                opts.vocab_size, n_v + 2, opts.receiver_emb, num_heads=10, hidden_size=opts.receiver_hidden, num_layers=opts.cell_layers,
                causal=True)#False)
    elif opts.receiver_cell == 'linear': 
        receiver = LinearReceiver(n_outputs=n_a * n_v, vocab_size=n_v + 1, max_length=3)#10)
    elif opts.receiver_cell == 'non-linear': 
        receiver = NonLinearReceiver(n_outputs=n_a * n_v, vocab_size=n_v + 1, max_length=2, n_hidden=opts.receiver_hidden)
    else:
        receiver = Receiver(n_hidden=opts.receiver_hidden, n_dim=n_a * n_v, inner_layers=opts.receiver_layers)
        #receiver = ReceiverRandomized(n_hidden=opts.receiver_hidden, n_a=n_a, n_v=n_v, inner_layers=opts.receiver_layers)
        receiver = core.RnnReceiverDeterministic(#Reinforce( #Deterministic(
                receiver, opts.vocab_size + 1,  # exclude eos = 0
                opts.receiver_emb, opts.receiver_hidden, cell=opts.receiver_cell,
                num_layers=opts.cell_layers)

    diff_loss = DiffLoss(n_a, n_v, opts.loss_type, opts.predict)


    game = core.SenderReceiverRnnReinforce(sender, receiver, diff_loss, receiver_entropy_coeff=0.05, sender_entropy_coeff=0.0)
    #game = core.SenderReceiverRnnDeterministic(sender, receiver, diff_loss, receiver_entropy_coeff=0.05, sender_entropy_coeff=0.0)
       
    optimizer = core.build_optimizer(receiver.parameters())
    loss = game.loss


    metrics_evaluator = Metrics(train_data.data, opts.device, n_a, n_v, opts.vocab_size + 1, freq=1)
    early_stopper = core.EarlyStopperAccuracy(1.0, validation=False)

    trainer = core.Trainer(
        game=game, optimizer=optimizer,
        train_data=train_loader,
        callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=True), metrics_evaluator, early_stopper],
        grad_norm=1.0)

    trainer.train(n_epochs=opts.n_epochs)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
