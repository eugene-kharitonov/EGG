# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import random


class UniformAgentSampler(nn.Module):
    # NB: only a module to facilitate checkpoint persistance
    def __init__(self, senders, receivers, losses):
        super().__init__()

        self.senders = nn.ModuleList(senders)
        self.receivers = nn.ModuleList(receivers)
        self.losses = list(losses)

    def forward(self):
        return random.choice(self.senders), random.choice(self.receivers), random.choice(self.losses)


class PopulationGame(nn.Module):
    def __init__(self, game, agents_loss_sampler):
        super().__init__()

        self.game = game
        self.agents_loss_sampler = agents_loss_sampler

    def forward(self, *args, **kwargs):
        sender, receiver, loss = self.agents_loss_sampler()

        return self.game(sender, receiver, loss, *args, **kwargs)

