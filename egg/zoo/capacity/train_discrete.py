# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from egg.zoo.capacity.dataset import SphereData, AttributeValueData
from egg.zoo.capacity.archs import (Receiver, MixerDiscrete, DiscreteWrapperModule, ArithmeticSender2,
    IdentitySender, LinearReceiver, NonLinearReceiver, ArithmeticSender, PositionalScrambler, HashSender,
    MultiHashSender, VocabScrambler)

from egg.zoo.capacity.intervention import Evaluator, Metrics
from .dataset import one_hotify


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

    parser.add_argument('--language', type=str, choices=['identity', 'scrambled-identity', 'arithmetic', 'scrambled-arithmetic', 'hash', 'scrambled-hash', 'arithmetic2'])
    parser.add_argument('--base', type=int, default=2)

    parser.add_argument('--scramble_positions', choices=['0', '1'])
    parser.add_argument('--scramble_vocab', action='store_true')

    args = core.init(arg_parser=parser, params=params)

    assert args.base >= 2
    return args


class DiffLoss(torch.nn.Module):
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


class DiffLoss2(torch.nn.Module):
    def __init__(self, n_attributes, n_values):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values

    def forward(self, sender_input, _message, _receiver_input, receiver_output, _labels):
        batch_size = sender_input.size(0)
        one_hot_sender_input = torch.zeros(batch_size, self.n_attributes, self.n_values, device=sender_input.device)
        for b in range(batch_size):
            for i in range(self.n_attributes):
                #one_hot_sender_input.scatter_(src=1, dim=-1, index=sender_input[:, i])
                #one_hot_sender_input[:, i, :].scatter_(1, sender_input[:, i], 1)
                one_hot_sender_input[b, i, sender_input[b, i]] = 1

        one_hot_sender_input = one_hot_sender_input.view(batch_size, -1)
        #one_hot_sender_input = one_hotify(sender_input, self.n_attributes, self.n_values)
        #one_hot_sender_input = torch.cat(one_hot_sender_input, dim=0)

        loss = F.binary_cross_entropy_with_logits(receiver_output, one_hot_sender_input)

        receiver_output = receiver_output.view(batch_size, self.n_attributes, self.n_values)
        acc = (torch.sum((receiver_output.argmax(dim=-1) == sender_input).detach(), dim=1) == self.n_attributes).float().mean()
        acc_or = (receiver_output.argmax(dim=-1) == sender_input).float().mean()


        return loss, {'acc': acc, 'acc_or': acc_or}


def main(params):
    opts = get_params(params)
    print(opts)

    device = opts.device

    n_a, n_v = opts.n_a, opts.n_v 
    assert opts.vocab_size > n_v

    train_data = AttributeValueData(n_attributes=n_a, n_values=n_v)
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

    #for k, _ in train_data:
        #k = train_data.data[0]
        #print(k, sender(k.unsqueeze(0))[0].squeeze())

    if opts.receiver_cell == 'transformer':
            receiver = Receiver(n_hidden=opts.receiver_emb, n_dim=n_a * n_v, inner_layers=opts.receiver_layers)
            receiver = core.TransformerReceiverDeterministic(receiver, 
                opts.vocab_size, n_v + 2, opts.receiver_emb, num_heads=10, hidden_size=opts.receiver_hidden, num_layers=opts.cell_layers,
                causal=True)#False)
    elif opts.receiver_cell == 'linear': 
        receiver = LinearReceiver(n_outputs=n_a * n_v, vocab_size=n_v + 1, max_length=10)
    elif opts.receiver_cell == 'non-linear': 
        receiver = NonLinearReceiver(n_outputs=n_a * n_v, vocab_size=n_v + 1, max_length=n_a + 1, n_hidden=opts.receiver_hidden)
    else:
        receiver = Receiver(n_hidden=opts.receiver_hidden, n_dim=n_a * n_v, inner_layers=opts.receiver_layers)
        receiver = core.RnnReceiverDeterministic(
                receiver, opts.vocab_size + 1,  # exclude eos = 0
                opts.receiver_emb, opts.receiver_hidden, cell=opts.receiver_cell,
                num_layers=opts.cell_layers)

    diff_loss = DiffLoss2(n_a, n_v)

    game = core.SenderReceiverRnnReinforce(sender, receiver, diff_loss, receiver_entropy_coeff=0.0, sender_entropy_coeff=0.0)
       
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
