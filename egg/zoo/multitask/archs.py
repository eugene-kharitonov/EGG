# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class Receiver(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(Receiver, self).__init__()
        self.output = nn.Linear(n_hidden, n_features)

    def forward(self, x, _input):
        return self.output(x)


class Sender(nn.Module):
    def __init__(self, n_hidden_features, n_features):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden_features)

    def forward(self, x):
        x = self.fc1(x)
        return x


class SenderTaskEmbedding(nn.Module):
    def __init__(self, n_hidden_features, n_features, n_features_task, n_hidden_task):
        super(SenderTaskEmbedding, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden_features)
        self.fc2 = nn.Linear(n_features_task, n_hidden_task)

    def forward(self, x, task_id):
        x = self.fc1(x)
        y = self.fc2(task_id.float()).squeeze()
        y = torch.stack([y] * x.shape[0], dim=0)
        return torch.cat([x, y], dim=1)
