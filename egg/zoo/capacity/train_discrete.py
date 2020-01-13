# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from egg.zoo.capacity.dataset import SphereData, AttributeValueData
from egg.zoo.capacity.archs import PositionalSender, Receiver, ReflectorLenses, \
    RotatorLenses, PlusOneWrapper, SubspaceSwapLenses, Mixer2d, WrapperModule, MixerDiscrete, DiscreteWrapperModule, UnMixerDiscrete, \
    DiscretePositionalSender, DiagonalSwapDiscrete, LinearReceiver, NonLinearReceiver

from egg.zoo.capacity.intervention import Evaluator, Metrics



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

    parser.add_argument('--mixer_epochs', type=int, default=10)

    parser.add_argument('--unmixer_inner', type=int, default=-1)
    parser.add_argument('--mixers', type=int, default=1)
    parser.add_argument('--n_a', type=int, default=2)
    parser.add_argument('--n_v', type=int, default=10)

    args = core.init(arg_parser=parser, params=params)

    return args

"""
class Loss(torch.nn.Module):

    def __init__(self, n_a, n_v):
        self.n_a = n_a
        self.n_v = n_v

    def forward(self, sender_input, _message, _receiver_input, receiver_output, _labels):
        batch_size = sender_input.size(0)
        sender_input = sender_input.view(batch_size, self.n_a, self.n_v).argmax(dim=-1)
        receiver_output = receiver_output.view(batch_size, self.n_a, self.n_v)

        loss = F.binary_cross_entropy(receiver_output, sender_input)
        return loss, {}
"""

class DiffLoss(torch.nn.Module):
    def __init__(self, n_attributes, n_values):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values

    def forward(self, sender_input, _message, _receiver_input, receiver_output, _labels):
        batch_size = sender_input.size(0)
        #sender_input = sender_input.view(batch_size, self.n_attributes, self.n_values)
        receiver_output = receiver_output.view(batch_size, self.n_attributes, self.n_values)

        acc = (torch.sum((receiver_output.argmax(dim=-1) == sender_input).detach(), dim=1) == self.n_attributes).float().mean()
        acc_or = (receiver_output.argmax(dim=-1) == sender_input).float().mean()

        receiver_output = receiver_output.view(batch_size * self.n_attributes, self.n_values)
        labels = sender_input.view(batch_size * self.n_attributes)
        loss = F.cross_entropy(receiver_output, labels, reduction="none").view(batch_size, self.n_attributes).mean(dim=-1)

        return loss, {'acc': acc, 'acc_or': acc_or}

def train_mixer(train_loader, mixer, unmixer, use_cuda, n_epochs):
    wrapper = DiscreteWrapperModule(mixer, unmixer)

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

    n_a, n_v = opts.n_a, opts.n_v #2, 5#$2, 2 #3, 5

    assert opts.vocab_size > n_v

    train_data = AttributeValueData(n_attributes=n_a, n_values=n_v)
    train_loader = DataLoader(train_data, batch_size=opts.batch_size)

    #mixer = DiagonalSwapDiscrete(n_a, n_v)#None

    mixer = None
    if opts.mixers > 0:
        #mixer = DiagonalSwapDiscrete(n_a, n_v)#
        mixer = torch.nn.Sequential(
            *(MixerDiscrete(n_attributes=n_a, n_values=n_v) for _ in range(opts.mixers))
        )
        #unmixer = UnMixerDiscrete(n_attributes=n_a, n_values=n_v, inner_layers=opts.unmixer_inner)
        #train_mixer(train_loader, mixer, unmixer, opts.cuda, n_epochs=opts.mixer_epochs)
        #for d, _ in train_loader:
        #    print(d)
        #    print(mixer(d))
        #    #exit(0)


    sender = DiscretePositionalSender(lense=mixer, n_attributes=n_a, n_values=n_v)
    #sender = PlusOneWrapper(sender)



    if opts.receiver_cell == 'transformer':
            receiver = Receiver(n_hidden=opts.receiver_emb, n_dim=n_a * n_v, inner_layers=opts.receiver_layers)
            receiver = core.TransformerReceiverDeterministic(receiver, 
                opts.vocab_size, n_v + 2, opts.receiver_emb, num_heads=10, hidden_size=opts.receiver_hidden, num_layers=opts.cell_layers,
                causal=True)#False)
    elif opts.receiver_cell == 'linear': 
        receiver = LinearReceiver(n_outputs=n_a * n_v, vocab_size=n_v + 1, max_length=n_a + 1)
    elif opts.receiver_cell == 'non-linear': 
        receiver = NonLinearReceiver(n_outputs=n_a * n_v, vocab_size=n_v + 1, max_length=n_a + 1, n_hidden=opts.receiver_hidden)
    else:
        receiver = Receiver(n_hidden=opts.receiver_hidden, n_dim=n_a * n_v, inner_layers=opts.receiver_layers)
        receiver = core.RnnReceiverDeterministic(
                receiver, opts.vocab_size + 1,  # exclude eos = 0
                opts.receiver_emb, opts.receiver_hidden, cell=opts.receiver_cell,
                num_layers=opts.cell_layers)

    diff_loss = DiffLoss(n_a, n_v)

    game = core.SenderReceiverRnnReinforce(sender, receiver, diff_loss, receiver_entropy_coeff=0.0, sender_entropy_coeff=0.0)
       
    optimizer = core.build_optimizer(game.parameters())
    loss = game.loss


    metrics_evaluator = Metrics(train_data.data, opts.device, n_a, n_v, opts.vocab_size + 1, freq=1)
    early_stopper = core.EarlyStopperAccuracy(0.5, validation=False)

    #intervention = Evaluator(train_loader, device=device, is_gs=opts.mode == 'gs', loss=loss, var_length=opts.variable_length,
    #                                 input_intervention=True)

    trainer = core.Trainer(
        game=game, optimizer=optimizer,
        train_data=train_loader,
        #validation_data=train_loader,
        callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=True), metrics_evaluator, early_stopper],
        grad_norm=1.0)

    trainer.train(n_epochs=opts.n_epochs)

    #if opts.dump_language:
    #    dump(game, test_loader, device, is_gs=opts.mode ==
    #         'gs', is_var_length=opts.variable_length)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
