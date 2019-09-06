# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.distributions import Categorical

import egg.core as core


class Receiver(nn.Module):
    def __init__(self, n_bits, n_hidden):
        super(Receiver, self).__init__()
        self.emb_column = nn.Linear(n_bits, n_hidden)

        self.fc1 = nn.Linear(2 * n_hidden, 2 * n_hidden)
        self.fc2 = nn.Linear(2 * n_hidden, n_bits)

    def forward(self, embedded_message, bits):
        embedded_bits = self.emb_column(bits.float())

        x = torch.cat([embedded_bits, embedded_message], dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        return x.sigmoid()


class ReinforcedReceiver(nn.Module):
    def __init__(self, n_bits, n_hidden):
        super(ReinforcedReceiver, self).__init__()
        self.emb_column = nn.Linear(n_bits, n_hidden)

        self.fc1 = nn.Linear(2 * n_hidden, 2 * n_hidden)
        self.fc2 = nn.Linear(2 * n_hidden, n_bits)

    def forward(self, embedded_message, bits):
        embedded_bits = self.emb_column(bits.float())

        x = torch.cat([embedded_bits, embedded_message], dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        probs = x.sigmoid()

        distr = Bernoulli(probs=probs)
        entropy = distr.entropy()

        if self.training:
            sample = distr.sample()
        else:
            sample = (probs > 0.5).float()
        log_prob = distr.log_prob(sample).sum(dim=1)
        return sample, log_prob, entropy


class Sender(nn.Module):
    def __init__(self, vocab_size, n_bits, n_hidden):
        super(Sender, self).__init__()
        self.emb = nn.Linear(n_bits, n_hidden)
        self.fc = nn.Linear(n_hidden, vocab_size)

    def forward(self, bits):
        x = self.emb(bits.float())
        x = F.leaky_relu(x)
        message = self.fc(x)

        return message


class FactorizedSender(nn.Module):
    def __init__(self, vocab_size, max_len, n_bits, n_hidden):
        super().__init__()


        # reserve one position for EOS
        self.symbol_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_bits, n_hidden),
                nn.LeakyReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.LeakyReLU(),
                nn.Linear(n_hidden, vocab_size)
            ) for _ in range(max_len - 1)
        ])

    def forward(self, bits):
        sequence = []
        log_probs = []
        entropy = []

        for generator in self.symbol_generators:
            logits = generator(bits.float())

            step_logits = F.log_softmax(logits, dim=-1)
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                sample = distr.sample()
            else:
                sample = step_logits.argmax(dim=-1)

            log_probs.append(distr.log_prob(sample))
            sequence.append(sample)

        zeros = torch.zeros(bits.size(0)).to(bits.device)

        sequence.append(zeros.long())
        sequence = torch.stack(sequence).permute(1, 0)

        log_probs.append(zeros)
        log_probs = torch.stack(log_probs).permute(1, 0)

        entropy.append(zeros)
        entropy = torch.stack(entropy).permute(1, 0)

        return sequence, log_probs, entropy



class Discriminator(nn.Module):
    def __init__(self, vocab_size, n_hidden, embed_dim):
        super().__init__()
        self.encoder = core.RnnEncoder(vocab_size, embed_dim=embed_dim, n_hidden=n_hidden)
        self.fc = nn.Linear(n_hidden, 2, bias=False)
        #self.emb = nn.Embedding(embed_dim, n_hidden)

    def forward(self, message):
        x = self.encoder(message)
        #x = self.fc(message[:, 0])
        #x = self.emb(message[:, 0])
        #x = F.leaky_relu(x)
        x = self.fc(x)
        return x
