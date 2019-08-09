# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

import egg.core as core

class Masker(nn.Module):
    def __init__(self, vocab_size, max_len, prior=0.0):
        super().__init__()
        self.prob_mask_logits = torch.nn.Parameter(torch.zeros(max_len) + prior)
        self.replace_id = torch.tensor([vocab_size + 1]).long()

    def forward(self, sequence):
        batch_size = sequence.size(0)
        extended_logits = self.prob_mask_logits.unsqueeze(0).expand((batch_size, sequence.size(1)))

        hard_attention_distr = Bernoulli(logits=extended_logits)
        if True:# self.training:
            hard_mask = hard_attention_distr.sample()
        else:
            hard_mask = (extended_logits > 0.0).float()

        logits = hard_attention_distr.log_prob(hard_mask).sum(dim=-1)
        hard_mask_byte = hard_mask > 0.5
        sequence[hard_mask_byte] = self.replace_id

        return sequence, logits, hard_mask

# TODO: don't hide beyond EOS
# TODO: also do lstms
class Explainer(nn.Module):
    def __init__(self, vocab_size, max_len, n_bits, prior=-2.0):
        super().__init__()

        self.encoder = core.TransformerBaseEncoder(vocab_size=vocab_size + 2, max_len=max_len + 1, embed_dim=10, num_heads=2,
                hidden_size=20, num_layers=3)

        self.predictor = nn.Linear(10, n_bits)

        self.sos_id = torch.tensor([vocab_size]).long()
        self.epochs = 0


    def forward(self, sequence):
        batch_size = sequence.size(0)
        lengths = core.find_lengths(sequence)

        prefix = self.sos_id.to(sequence.device).unsqueeze(0).expand((batch_size, 1))
        sequence = torch.cat([prefix, sequence], dim=1)
        lengths = lengths + 1

        max_len = sequence.size(1)
        len_indicators = torch.arange(max_len).expand((batch_size, max_len)).to(lengths.device)
        lengths_expanded = lengths.unsqueeze(1)

        padding_mask = len_indicators >= lengths_expanded

        transformed = self.encoder(sequence, padding_mask)
        transformed = transformed[:, 0, :]

        predicted = self.predictor(transformed)
        predicted = predicted.sigmoid()

        return predicted


class Game(nn.Module):
    def __init__(self, masker, explainer, l):
        super().__init__()

        self.masker = masker
        self.explainer = explainer
        self.l = l

    def forward(self, sequence, labels, _other=None):
        sequence, logits, hard_mask = self.masker(sequence)
        predicted = self.explainer(sequence)

        loss = F.binary_cross_entropy(predicted, labels.float(), reduction='none').mean(dim=-1)
        loss = loss + ((loss.detach() - self.l * hard_mask.mean(dim=-1)) * logits).mean()

        acc = ((predicted > 0.5).long() == labels).detach().all(dim=1).float().mean()
        nnz_att = hard_mask.float().mean()
        return loss.mean(), {'acc': acc, 'nnz_att': nnz_att}