# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.utils.data as data
import torch.nn.parallel
import torch
import numpy as np
import math


class SphereData:
    def __init__(self, n_points, n_dim):
        assert n_dim == 2

        radii = torch.FloatTensor(n_points, 1).uniform_(0, 1)
        angle = torch.FloatTensor(n_points, 1).uniform_(0, 2 * math.pi)

        data_xy = torch.cat([torch.cos(angle), torch.sin(angle)], dim=1) * radii
        self.data_xy = data_xy
        self.data_ar = torch.cat([angle, radii], dim=1)

    def __len__(self):
        return self.data_xy.size(0)

    def __getitem__(self, k):
        return self.data_xy[k, :], self.data_ar[k, :]



if __name__ == '__main__':
    s = SphereData(n_points=4, n_dim=2)