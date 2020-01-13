# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.utils.data as data
import torch.nn.parallel
import torch
import numpy as np
import math
import itertools


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




def enumerate_attribute_value(n_attributes, n_values):
    iters = [range(n_values) for _ in range(n_attributes)]

    return list(itertools.product(*iters))


def one_hotify(data, n_attributes, n_values):
    r = []
    for config in data:
        z = torch.zeros((n_attributes, n_values))
        for i in range(n_attributes):
            z[i, config[i]] = 1
        r.append(z.view(-1))
    return r

class AttributeValueData:
    def __init__(self, n_attributes, n_values, one_hot=False):
        self.data = [torch.LongTensor(k) for k in enumerate_attribute_value(n_attributes, n_values)]
        if one_hot:
            self.data = one_hotify(self.data, n_attributes, n_values)

        for k in self.data:
            k[0] = (k[0] + 7).fmod(n_values)
            k[1] = (k[1] + 5).fmod(n_values)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, k):
        return self.data[k], torch.zeros(1)
    

if __name__ == '__main__':
    #s = SphereData(n_points=4, n_dim=2)

    for b in AttributeValueData(4, 4):
        print(b)