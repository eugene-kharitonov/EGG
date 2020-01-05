# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from egg.zoo.capacity.dataset import SphereData
from egg.zoo.capacity.archs import PositionalSender, Receiver, RotatorLenses, \
    PlusOneWrapper, Mixer2d, Discriminator, grad_reverse, Predictor

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


def diff_loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    loss = F.mse_loss(receiver_output, sender_input)
    return loss, {}


class WrapperModule(torch.nn.Module):
    def __init__(self, mixer, unmixer, n_dim=2):
        super().__init__()

        self.mixer = mixer
        self.unmixer = unmixer

        self.predictors = torch.nn.ModuleList(
            Predictor(n_dim) for _ in range(n_dim)
        )

    def forward(self, points):
        mixed = self.mixer(points)
        unmixed = self.unmixer(mixed)
        recovery_loss = F.mse_loss(points, unmixed)

        norm = mixed.pow(2.0).sum(-1)
        norm_loss = (norm - 1.0).clamp(min=0).sum()

        mixing_loss = 0
        for i, p in enumerate(self.predictors):
            predicted = p(mixed[:, i].unsqueeze(-1))
            # NB: only 2D
            loss_predict_0 = F.mse_loss(points[:, 0], predicted[:, 0])
            loss_predict_1 = F.mse_loss(points[:, 1], predicted[:, 1])
            mixing_loss = mixing_loss + (loss_predict_0 - loss_predict_1).abs()

        loss = recovery_loss + norm_loss + mixing_loss
        return loss

def train_epoch(train_loader, optimizer, model, use_cuda):
    pass

def main(params):
    import math

    opts = get_params(params)
    print(opts)#json.dumps(vars(opts)))

    device = opts.device

    train_data = SphereData(n_points=opts.n_examples, n_dim=2)
    train_loader = DataLoader(train_data, batch_size=opts.batch_size)

    mixer = Mixer2d()
    unmixer = Mixer2d()

    wrapper = WrapperModule(mixer, unmixer)

    if opts.cuda:
        wrapper.cuda()

    params = wrapper.parameters()
    optimizer = core.build_optimizer(params)
    
    for epoch in range(opts.n_epochs):

        for points, _ in train_loader:
            n_batch = points.size(0)

            if opts.cuda:
                points = points.cuda()

            optimizer.zero_grad()
            loss = wrapper(points)

            loss.backward()
            optimizer.step()

        print(loss.detach())

    core.close()
    import math

    print(mixer.fc.weight)
    print(unmixer.fc.weight)

    w = mixer.fc.weight
    print(math.atan2(w[0,1], w[0,0]) / math.pi)

    print(torch.matmul(mixer.fc.weight, unmixer.fc.weight))

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
