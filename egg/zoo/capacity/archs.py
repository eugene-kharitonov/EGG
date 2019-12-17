# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

import egg.core as core


class Receiver(nn.Module):
    def __init__(self, n_hidden, n_dim):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(n_hidden, n_dim)

    def forward(self, x, _):
        x = self.fc(x)
        return x

class PositionalSender(nn.Module):
    def __init__(self, vocab_size, lense=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.lense = lense

    def forward(self, x):
        batch_size = x.size(0)

        if self.lense:
            x = self.lense(x)

        assert (x >= -1).all() and (x <= 1).all()
        message = ((x + 1) / 2 * (self.vocab_size - 1)).round().long()
        assert (message < self.vocab_size).all()

        zeros = torch.zeros(x.size(0), x.size(1), device=x.device)
        return message, zeros, zeros


class RotatorLenses(nn.Module):
    def __init__(self, theta):
        super().__init__()

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        self.rotation_matrix = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]], requires_grad=False)

    def __call__(self, examples):
        with torch.no_grad():
            r = examples.matmul(self.rotation_matrix)
        return r

if __name__ == '__main__':
    from .dataset import SphereData
    from torch.utils.data import DataLoader
    sender = PositionalSender(vocab_size=100)

    for example in DataLoader(SphereData(n_points=10, n_dim=2), batch_size=1):
        print(example, sender(example)[0])