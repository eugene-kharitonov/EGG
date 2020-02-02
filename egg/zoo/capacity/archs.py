# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import itertools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical

import egg.core as core


class LinearReceiver(nn.Module):
    def __init__(self, n_outputs, vocab_size, max_length):
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.fc = nn.Linear(vocab_size * max_length, n_outputs)

        self.diagonal_embedding = nn.Embedding(vocab_size, vocab_size)
        nn.init.eye_(self.diagonal_embedding.weight)

    def forward(self, x, *rest):
        with torch.no_grad():
            x = self.diagonal_embedding(x).view(x.size(0), -1)

        result = self.fc(x)

        zeros = torch.zeros(x.size(0), device=x.device)
        return result, zeros, zeros


class NonLinearReceiver(nn.Module):
    def __init__(self, n_outputs, vocab_size, n_hidden, max_length):
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.fc_1 = nn.Linear(vocab_size * max_length, n_hidden)
        self.fc_2 = nn.Linear(n_hidden, n_outputs)

        self.diagonal_embedding = nn.Embedding(vocab_size, vocab_size)
        nn.init.eye_(self.diagonal_embedding.weight)

    def forward(self, x, *rest):
        with torch.no_grad():
            x = self.diagonal_embedding(x).view(x.size(0), -1)

        x = self.fc_1(x)
        x = F.leaky_relu(x)
        x = self.fc_2(x)

        zeros = torch.zeros(x.size(0), device=x.device)
        return x, zeros, zeros


class PositionalScrambler(nn.Module):
    def __init__(self):
        super(PositionalScrambler, self).__init__()
        self.scrambler = None

    def forward(self, x):
        assert self.scrambler is None or self.scrambler.size(0) == x[0].size(1)

        if self.scrambler is None:
            assert self.scrambler is None

            n = x[0].size(1)
            self.scrambler = torch.randperm(n)

        return x[0][:, self.scrambler], x[1], x[2]




class VocabScrambler(nn.Module):
    def __init__(self, base):
        super(VocabScrambler, self).__init__()

        self.scrambler = None 
        self.base = base

    def forward(self, x):
        positions = x[0].size(1)
        if self.scrambler is None:
            self.scrambler = [
                1 + torch.randperm(self.base + 1).to(x[0].device) for _ in range(positions)
            ]

        result = []
        for p in range(positions - 1):
            result.append(
                #torch.index_select(input=self.scrambler[p], dim=0, index=x[0][:, p])
                self.scrambler[p][x[0][:, p]].unsqueeze(1)
            )

        # eos
        result.append(x[0][:, -1].unsqueeze(-1))

        result = torch.cat(result, dim=1)
        #print(result.size())
        #exit(0)
        return result, x[1], x[2]


class Receiver(nn.Module):
    def __init__(self, n_hidden, n_dim, inner_layers=-1):
        super(Receiver, self).__init__()
        if inner_layers == -1:
            self.net = nn.Linear(n_hidden, n_dim)
        else:
            l = [nn.Linear(n_hidden, n_hidden), nn.LeakyReLU()]

            for _ in range(inner_layers):
                l += [nn.Linear(n_hidden, n_hidden), nn.LeakyReLU()]
            l.append(nn.Linear(n_hidden, n_dim))

            self.net = nn.Sequential(*l)

    def forward(self, x, _):
        x = self.net(x)
        return x


class ReceiverRandomized(nn.Module):
    def __init__(self, n_hidden, n_a, n_v, inner_layers=-1):
        super().__init__()
        n_dim = n_a * n_v
        if inner_layers == -1:
            self.net = nn.Linear(n_hidden, n_dim)
        else:
            l = [nn.Linear(n_hidden, n_hidden), nn.LeakyReLU()]

            for _ in range(inner_layers):
                l += [nn.Linear(n_hidden, n_hidden), nn.LeakyReLU()]
            l.append(nn.Linear(n_hidden, n_a * n_v))

            self.net = nn.Sequential(*l)

        self.n_a, self.n_v = n_a, n_v

    def forward(self, x, _):
        x = self.net(x)
        b = x.size(0)
        x = x.view(b, self.n_a, self.n_v)

        d = Categorical(logits=x)

        if self.training:
            sample = d.sample()
        else:
            sample = x.argmax(dim=-1)
        log_probs = d.log_prob(sample).sum(dim=-1)
        entropy = d.entropy().sum(dim=-1)

        return sample, log_probs, entropy


class IdentitySender(nn.Module):
    def __init__(self, n_attributes, n_values):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values

    def forward(self, x):
        batch_size = x.size(0)

        message = x
        #assert message.size(1) == 2

        #tail = torch.zeros(batch_size, 1).long().to(x.device)
        #message = torch.cat([message + 1, tail], dim=1)

        zeros = torch.zeros(message.size(0), message.size(1), device=x.device)
        return message + 1, zeros, zeros


class ArithmeticSender(nn.Module):
    def __init__(self, n_attributes, n_values, base):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.base = base

        log = 0
        k = 1

        while k < n_values:
            k *= base
            log += 1

        self.mapping = nn.Embedding(n_values, log)
        torch.nn.init.zeros_(self.mapping.weight)

        for i in range(n_values):
            value = i
            for k in range(log):
                self.mapping.weight[i, k] = value % base
                value = value // base

        assert (self.mapping.weight < base).all()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size * self.n_attributes)
        with torch.no_grad():
            x = self.mapping(x)
        x = x.view(batch_size, -1).long()

        zeros = torch.zeros(x.size(0), x.size(1), device=x.device)
        return x + 1, zeros, zeros


class ArithmeticSender2(nn.Module):
    def __init__(self, n_attributes, n_values, base):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.base = base

        log = 0
        k = 1

        while k < n_values:
            k *= base
            log += 1

        self.mapping = nn.Embedding(n_values, log)
        torch.nn.init.zeros_(self.mapping.weight)

        for i in range(n_values):
            value = i
            for k in range(log):
                self.mapping.weight[i, k] = value % base
                value = value // base

        assert (self.mapping.weight < base).all()

    def forward(self, x):
        batch_size = x.size(0)
        #x = x.view(batch_size * self.n_attributes)
        with torch.no_grad():
            x = self.mapping(x)

        assert x.size(1) == 2

        for a in range(self.n_attributes):
            left = (x[:, a, 0] + x[:, a, 1]).fmod(self.base)
            right = (x[:, a, 0] - x[:, a, 1] + self.base).fmod(self.base)
            x[:, a, 0] = left
            x[:, a, 1] = right

        x = x.view(batch_size, -1).long()

        zeros = torch.zeros(x.size(0), x.size(1), device=x.device)
        return x + 1, zeros, zeros


class RandomizedIdentitySender(nn.Module):
    def __init__(self, n_values):
        super().__init__()
        self.n_values = n_values
        self.multiplier = 1

    def forward(self, x):
        assert x.size(1) == 2, f'{x.size()}'
        batch_size = x.size(0)
        with torch.no_grad():
            random_shift = torch.randint(self.multiplier, size=(batch_size, 1), device=x.device)
            assert (random_shift < self.multiplier).all()
            result = x * self.multiplier + random_shift
                
        zeros = torch.zeros_like(result, dtype=torch.float)
        return result + 1, zeros, zeros

class RandomizedHashSender(nn.Module):
    def __init__(self, n_values, base):
        super().__init__()
        self.n_values = n_values
        self.base = base

        log = 0
        k = 1

        #print('NO RANDOMIZATION')
        while k < n_values * n_values:
            k *= base
            log += 1

        all_messages = list(itertools.product(*(range(base) for _ in range(log))))
        n = n_values * n_values
        selected = random.sample(all_messages, n)
        selected = torch.tensor(selected).long()
        selected.requires_grad_ = False

        self.mapping = selected.view(n, log)

    def forward(self, x):
        assert x.size(1) == 2, f'{x.size()}'
        batch_size = x.size(0)
        result = []
        with torch.no_grad():
            for i in range(2):
                random_shift = torch.randint(self.n_values, size=(batch_size,), device=x.device)
                look_up_index = x[:, i] * self.n_values + random_shift
                result.append(self.mapping[look_up_index, :])
                
        result = torch.cat(result, dim=1).to(x.device)

        zeros = torch.zeros_like(result, dtype=torch.float)#(x.size(0), x.size(1), device=x.device)
        return result + 1, zeros, zeros

class HashSender(nn.Module):
    def __init__(self, n_values, base):
        super().__init__()
        self.n_values = n_values
        self.base = base

        log = 0
        k = 1

        while k < n_values:
            k *= base
            log += 1

        all_messages = list(itertools.product(*(range(base) for _ in range(log))))
        selected = random.sample(all_messages, n_values)
        selected = torch.tensor(selected).long()
        selected.requires_grad_ = False

        self.mapping = nn.Embedding(n_values, log)
        self.mapping.weight[:, :] = selected

        assert (self.mapping.weight < base).all()

    def forward(self, x):
        batch_size = x.size(0)
        with torch.no_grad():
            x = self.mapping(x).long()
        x = x.view(batch_size, -1).long()

        zeros = torch.zeros(x.size(0), x.size(1), device=x.device)
        return x + 1, zeros, zeros

class MultiHashSender(nn.Module):
    def __init__(self, n_attributes, n_values, base):
        super().__init__()
        self.n_values = n_values
        self.n_attributes = n_attributes
        self.base = base

        log = 0
        k = 1

        while k < n_values:
            k *= base
            log += 1

        self.mappings = nn.ModuleList()

        for a in range(self.n_attributes):
            all_messages = list(itertools.product(*(range(base) for _ in range(log))))
            selected = random.sample(all_messages, n_values)
            selected = torch.tensor(selected).long()
            selected.requires_grad_ = False

            self.mappings.append(nn.Embedding(n_values, log))
            self.mappings[-1].weight[:, :] = selected

            assert (self.mappings[-1].weight < base).all()

    def forward(self, x):
        batch_size = x.size(0)
        assert self.n_attributes == x.size(1)

        result = []
        with torch.no_grad():
            for i in range(self.n_attributes):
                result.append(
                    self.mappings[i](x[:, i]).long()
                )
        x = torch.cat(result, dim=1)
        #x = x.view(batch_size, -1).long()

        zeros = torch.zeros(x.size(0), x.size(1), device=x.device)
        return x + 1, zeros, zeros


class UnfactorizedIdentitySender(nn.Module):
    def __init__(self, n_attributes, n_values):
        super().__init__()
        self.max_values = n_values ** n_attributes
        self.n_values = n_values

        all_messages = list(itertools.product(*(range(n_values) for _ in range(n_attributes))))
        random.shuffle(all_messages)

        selected = torch.tensor(all_messages).long()
        selected.requires_grad_ = False

        self.mapping = nn.Embedding(self.max_values, n_attributes)
        self.mapping.weight[:, :] = selected

    def forward(self, x):
        batch_size = x.size(0)

        k = x[:, 0]
        for i in range(1, x.size(1)):
            k = k * self.n_values + x[:, i]

        with torch.no_grad():
            x = self.mapping(k).long()

        zeros = torch.zeros(x.size(0), x.size(1), device=x.device)
        return x + 1, zeros, zeros


class UnfactorizedHashSender(nn.Module):
    def __init__(self, n_attributes, n_values, base):
        super().__init__()
        self.max_values = n_values ** n_attributes
        self.base = base
        self.n_values = n_values

        log = 0
        k = 1

        while k < self.max_values:
            k *= base
            log += 1

        self.mappings = nn.ModuleList()

        all_messages = list(itertools.product(*(range(base) for _ in range(log))))
        selected = random.sample(all_messages, self.max_values)
        selected = torch.tensor(selected).long()
        selected.requires_grad_ = False

        self.mapping = nn.Embedding(self.max_values, log)
        self.mapping.weight[:, :] = selected

        assert (self.mapping.weight < base).all()

    def forward(self, x):
        batch_size = x.size(0)

        k = x[:, 0]
        for i in range(1, x.size(1)):
            k = k * self.n_values + x[:, i]
        with torch.no_grad():
            x = self.mapping(k).long()

        zeros = torch.zeros(x.size(0), x.size(1), device=x.device)
        return x + 1, zeros, zeros


class MixerDiscrete(nn.Module):
    def __init__(self, n_attributes, n_values):
        super().__init__()

        self.n_attributes = n_attributes
        self.n_values = n_values
        
    def forward(self, inp):
        batch_size = inp.size(0)
        added = torch.zeros_like(inp).long()

        if inp.size(1) == 2:
            added[:, 0] = (inp[:, 0] + inp[:, 1]).fmod(self.n_values)
            added[:, 1] = (self.n_values + inp[:, 0] - inp[:, 1]).fmod(self.n_values)
        elif inp.size(1) == 3:
            added[:, 0] = (inp[:, 0] + inp[:, 1]).fmod(self.n_values)
            added[:, 1] = (self.n_values + inp[:, 0] - inp[:, 1]).fmod(self.n_values)
            added[:, 2] = (inp[:, 0] + inp[:, 2]).fmod(self.n_values)
        else:
            assert False

        return added

def sum_one_hots(a, b):
    assert a.size() == b.size()
    n_values = a.size(-1)

    added = torch.zeros_like(a)

    for i in range(n_values):
        for j in range(n_values):
            target_value = (i + j) % n_values
            added[:, target_value] += a[:, i] * b[:, j]
    return added


def mul_one_hots(a, b):
    assert a.size() == b.size()
    n_values = a.size(-1)

    mul = torch.zeros_like(a)

    for i in range(n_values):
        for j in range(n_values):
            target_value = (i * j) % n_values
            mul[:, target_value] += a[:, i] * b[:, j]
    return mul



class DiscreteWrapperModule(torch.nn.Module):
    def __init__(self, mixer, unmixer, n_dim=2):
        super().__init__()

        self.mixer = mixer
        self.unmixer = unmixer

    def forward(self, x):
        mixed = self.mixer(x)
        unmixed = self.unmixer(mixed)

        recovery_loss = F.binary_cross_entropy(unmixed, x)
        loss = recovery_loss
        return loss