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
            with torch.no_grad():
                x = self.lense(x)

        assert (x >= -1).all() and (x <= 1).all(), f'max {x.max()}, min {x.min()}'
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
        self.rotation_matrix = nn.Parameter(self.rotation_matrix)

    def __call__(self, examples):
        with torch.no_grad():
            r = examples.matmul(self.rotation_matrix)
        return r

class SubspaceSwapLenses(nn.Module):
    def __call__(self, examples):
        mask_1 = (examples[:, 0] > 0) & (examples[:, 1] > 0)
        mask_2 = (examples[:, 0] < 0) & (examples[:, 1] < 0)

        examples[mask_1, :].mul_(-1)
        examples[mask_2, :].mul_(-1)

        return examples

class PlusOneWrapper(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, *input):
        r1, r2, r3 = self.wrapped(*input)
        return r1 + 1, r2, r3



class Mixer2d(nn.Module):
    def __init__(self, inner_layers=-1):
        super().__init__()
        if inner_layers == -1:
            self.fc = nn.Linear(2, 2, bias=False)
        else:
            l = [nn.Linear(2, 10)]
            for _ in range(inner_layers):
                l.extend([
                    nn.Tanh(),
                    nn.Linear(10, 10)
                ])

            l += [nn.Tanh(), nn.Linear(10, 2)]
            self.fc = nn.Sequential(*l)

    def forward(self, examples):
        return self.fc(examples)


class Predictor(nn.Module):
    def __init__(self, n_dim=2):
        super().__init__()
        self.fc = nn.Linear(1, n_dim)
    def forward(self, examples):
        return self.fc(examples)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = nn.Linear(2, 20)
        self.fc_2 = nn.Linear(20, 20)
        self.fc_3 = nn.Linear(20, 2)

    def forward(self, examples):
        x = self.fc_1(examples)
        x = F.tanh(x)
        x = self.fc_2(x)
        x = F.tanh(x)
        x = self.fc_3(x)
        return x



class GradientReverse(torch.autograd.Function):
    scale = 1.0
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()
    
def grad_reverse(x, scale=1.0):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)



class WrapperModule(torch.nn.Module):
    def __init__(self, mixer, unmixer, n_dim=2):
        super().__init__()

        self.mixer = mixer
        self.unmixer = unmixer

        self.predictors = torch.nn.ModuleList(
            Predictor(n_dim) for _ in range(n_dim)
        )

        self.discriminator = Discriminator()

    def forward(self, points):
        mixed = self.mixer(points)
        unmixed = self.unmixer(mixed)
        recovery_loss = F.mse_loss(points, unmixed)

        mixing_loss = 0
        for i, p in enumerate(self.predictors):
            predicted = p(mixed[:, i].unsqueeze(-1))
            # NB: only 2D
            loss_predict_0 = F.mse_loss(points[:, 0], predicted[:, 0])
            loss_predict_1 = F.mse_loss(points[:, 1], predicted[:, 1])
            mixing_loss = mixing_loss + (loss_predict_0 - loss_predict_1).abs()

        batch_size = points.size(0)
        combined_batch = torch.cat([points, mixed], dim=0)
        labels = torch.zeros(2 * batch_size, dtype=torch.long)
        labels[:batch_size] = 1

        d_predictions = self.discriminator(grad_reverse(combined_batch))
        d_loss = F.cross_entropy(d_predictions, labels)


        loss = recovery_loss + 10 * d_loss + mixing_loss
        return loss

if __name__ == '__main__':
    from .dataset import SphereData
    from torch.utils.data import DataLoader
    sender = PositionalSender(vocab_size=100)

    for example in DataLoader(SphereData(n_points=10, n_dim=2), batch_size=1):
        print(example, sender(example)[0])


