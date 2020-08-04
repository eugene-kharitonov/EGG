# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import torch
import torch.utils.data
from torchvision import datasets, transforms
from torch import nn
from torch.nn import functional as F
import torch.distributions
import egg.core as core
from egg.core.population import UniformAgentSampler, PopulationGame
from egg.core.baselines import BuiltInBaseline

import argparse

class Sender(nn.Module):
    def __init__(self, hidden):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Receiver(nn.Module):
    def __init__(self, hidden):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(hidden, 784)

    def forward(self, x, _input):
        x = F.leaky_relu(x)  
        x = self.fc(x)
        return torch.sigmoid(x)


def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    """
    The autoencoder's loss function; cross-entropy between the original and restored images.
    """
    loss = F.binary_cross_entropy(receiver_output, sender_input.view(-1, 784), reduction='none').mean(dim=1)
    return loss, {}


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sender_hidden', type=int, default=50,
                        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument('--receiver_hidden', type=int, default=50,
                        help='Size of the hidden layer of Receiver (default: 10)')
    parser.add_argument('--sender_entropy_coeff', type=float, default=1e-2,
                        help="Entropy regularisation coeff for Sender (default: 1e-2)")
    parser.add_argument('--sender_emb', type=int, default=10,
                        help='Size of the embeddings of Sender (default: 10)')
    parser.add_argument('--receiver_emb', type=int, default=10,
                        help='Size of the embeddings of Receiver (default: 10)')
    parser.add_argument('--n_senders', type=int, default=1,
                        help='Number of Senders (default: 1)')
    parser.add_argument('--n_receivers', type=int, default=1,
                        help='Number of Receivers (default: 1)')
    args = core.init(arg_parser=parser, params=params)
    return args


def main(params):
    # initialize the egg lib
    opts = get_params(params)

    kwargs = {'num_workers': 1, 'pin_memory': True} if opts.cuda else {}
    transform = transforms.ToTensor()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
           transform=transform),
           batch_size=opts.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
           batch_size=opts.batch_size, shuffle=True, **kwargs)

    def get_sender():
        sender = Sender(opts.sender_hidden)
        sender = core.RnnSenderReinforce(sender,
                                         opts.vocab_size, opts.sender_emb, opts.sender_hidden,
                                         cell='lstm', max_len=2, force_eos=False)
        return sender

    def get_receiver():
        receiver = Receiver(opts.receiver_hidden)
        receiver = core.RnnReceiverDeterministic(receiver, opts.vocab_size, opts.receiver_emb,
                                                 opts.receiver_hidden, cell='lstm')

        return receiver

    senders = [get_sender() for _ in range(1)]
    receivers = [get_receiver() for _ in range(1)]
    losses = [loss]

    game = core.CommunicationRnnReinforce(opts.sender_entropy_coeff, receiver_entropy_coeff=0.0, length_cost=0.0, 
            baseline_type=BuiltInBaseline)
    sampler = UniformAgentSampler(senders, receivers, losses)

    game = PopulationGame(game, sampler)
    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader, validation_data=test_loader,
                           callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=True)])
    trainer.train(n_epochs=opts.n_epochs)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

