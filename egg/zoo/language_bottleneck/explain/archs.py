# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

import egg.core as core

"""
class GradReverse(nn.Function):
    @staticmethod
    def forward(ctx, x,lambd):
        ctx.save_for_backward(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambd=ctx.saved_tensors[0]
        return grad_output.neg() * lambd, None

def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)

"""

class Masker(nn.Module):
    def __init__(self, vocab_size, max_len, prior=0.0):
        super().__init__()
        self.prob_mask_logits = torch.nn.Parameter(torch.zeros(max_len) + prior)
        self.replace_id = torch.tensor([vocab_size + 1]).long()

    def forward(self, sequence):
        batch_size = sequence.size(0)
        extended_logits = self.prob_mask_logits.unsqueeze(0).expand((batch_size, sequence.size(1)))

        hard_attention_distr = Bernoulli(logits=extended_logits)
        if True: #self.training:
            hard_mask = hard_attention_distr.sample()
        else:
            hard_mask = (extended_logits > 0.0).float()

        logits = hard_attention_distr.log_prob(hard_mask).sum(dim=-1)
        hard_mask_byte = hard_mask > 0.5
        sequence[hard_mask_byte] = self.replace_id

        return sequence, logits, hard_mask


class Explainer(nn.Module):
    def __init__(self, vocab_size, max_len, n_bits):
        super().__init__()

        self.encoder = core.TransformerBaseEncoder(vocab_size=vocab_size + 2, 
                max_len=max_len + 1, 
                embed_dim=10, num_heads=2,
                hidden_size=20, num_layers=3)

        self.predictor = nn.Linear(10, n_bits)

        self.sos_id = torch.tensor([vocab_size]).long()

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
    def __init__(self, masker, n_bits, target_bit):
        super().__init__()

        self.masker = masker
        self.explainer_X = Explainer()
        self.explainer_Y = explainer_Y

        self.l = l

    def forward(self, sequence, labels, _other=None):
        bit_x = 0
        bit_y = 2

        masked_sequence, logits, hard_mask = self.masker(sequence)
        predicted_Y = self.explainer_Y(masked_sequence)
        predicted_X = self.explainer_X(masked_sequence)

        loss_X = F.binary_cross_entropy(predicted_X[:, 0], labels.float()[:, bit_x], reduction='none')#.mean(dim=-1)
        loss_Y = F.binary_cross_entropy(predicted_Y[:, 0], labels.float()[:, bit_y], reduction='none')#.mean(dim=-1)

        loss_XY = loss_X - loss_Y
        loss_supp = (loss_XY.detach() * logits).mean()

        loss = loss_X + loss_Y + loss_supp

        acc_X = ((predicted_X > 0.5).view(-1).long() == labels[:, bit_x]).float()
        acc_Y = ((predicted_Y > 0.5).view(-1).long() == labels[:, bit_y]).float()

        nnz_att = hard_mask.float().sum(dim=-1)
        return loss.mean(), {'acc_X': acc_X, 'acc_Y': acc_Y, 'nnz': nnz_att}