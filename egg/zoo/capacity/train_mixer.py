# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from egg.zoo.capacity.dataset import SphereData, AttributeValueData
from egg.zoo.capacity.archs import PositionalSender, Receiver, RotatorLenses, \
    PlusOneWrapper, Mixer2d, Predictor, WrapperModule, MixerDiscrete, DiscreteWrapperModule, UnMixerDiscrete

import json
import argparse
import torch.utils.data
import torch.nn.functional as F
import egg.core as core
from torch.utils.data import DataLoader


def get_params(params):
    print(params)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_examples', type=int, default=10000,
                        help='Number of examples seen in an epoch (default: 10000)')
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')

    parser.add_argument('--receiver_cell', type=str, default='rnn')
    parser.add_argument('--receiver_emb', type=int, default=10,
                        help='Size of the embeddings of Receiver (default: 10)')

    args = core.init(arg_parser=parser, params=params)
    return args


def train_mixer(train_loader, mixer, unmixer, use_cuda, n_epochs):
    wrapper = DiscreteWrapperModule(mixer, unmixer)

    if use_cuda:
        wrapper.cuda()
    wrapper.train()

    params = wrapper.parameters()
    optimizer = core.build_optimizer(params)
    
    for epoch in range(n_epochs):
        for points in train_loader:
            n_batch = points.size(0)

            if use_cuda: points = points.cuda()

            optimizer.zero_grad()
            loss = wrapper(points)

            loss.backward()
            optimizer.step()

        print(loss.detach())

def main(params):
    import math

    opts = get_params(params)
    print(opts)#json.dumps(vars(opts)))

    train_data = AttributeValueData(n_attributes=2, n_values=3)
    train_loader = DataLoader(train_data, batch_size=opts.batch_size)

    mixer = torch.nn.Sequential(
        MixerDiscrete(n_attributes=2, n_values=3),
        MixerDiscrete(n_attributes=2, n_values=3),
        #MixerDiscrete(n_attributes=2, n_values=3),
        #MixerDiscrete(n_attributes=2, n_values=3),
        #MixerDiscrete(n_attributes=2, n_values=3)
    )

    unmixer = UnMixerDiscrete(n_attributes=2, n_values=3)

    train_mixer(train_loader, mixer, unmixer, opts.cuda, opts.n_epochs)

    mixer.eval()

    for points in train_loader:
        print(points)
        print(mixer(points))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
