# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.utils.data as data
import torch.nn.parallel
import torch
import numpy as np


class SphereData:
    def __init__(self, n_points, n_dim):
        data = torch.randn(size=(n_points, n_dim))
        data = data / data.pow(2.0).sum(dim=-1, keepdim=True).sqrt()

        assert data[0].pow(2.0).sum().isclose(torch.tensor(1.0))

        self.data = data

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, k):
        return self.data[k, :], torch.zeros(1)


class RotatorLenses:
    def __init__(self, theta):
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        self.rotation_matrix = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    def __call__(self, examples):
        return examples.matmul(self.rotation_matrix)#.matmul(examples)

if __name__ == '__main__':
    s = SphereData(n_points=4, n_dim=2)