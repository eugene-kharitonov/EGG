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
    def __init__(self, replace_id, max_len, prior=0.0, mask=None):
        super().__init__()
        self.prob_mask_logits = torch.zeros(max_len).fill_(prior)
        if mask is not None:
            assert len(mask) == max_len, f'{mask} {max_len}'
            for i, v in enumerate(mask):
                if v == 'x': self.prob_mask_logits[i] = +100

        self.prob_mask_logits = torch.nn.Parameter(self.prob_mask_logits)
        self.replace_id = torch.tensor([replace_id]).long()
        self.pre_mask = mask

    def forward(self, sequence):
        batch_size = sequence.size(0)
        extended_logits = self.prob_mask_logits.unsqueeze(0).expand((batch_size, sequence.size(1)))

        hard_attention_distr = Bernoulli(logits=extended_logits)
        if self.training:
            hard_mask = hard_attention_distr.sample()
        else:
            hard_mask = (extended_logits > 0.0).float()

        logits = hard_attention_distr.log_prob(hard_mask).sum(dim=-1)
        hard_mask_byte = hard_mask > 0.5
        sequence[hard_mask_byte] = self.replace_id

        return sequence, logits, hard_mask

class DummyExplainer(nn.Module):
    def __init__(self, vocab_size, max_len, n_bits):
        super().__init__()
        self.predicted = torch.zeros(n_bits)

    def forward(self, sequence):
        batch_size = sequence.size(0)
        output = self.predicted.unsqueeze(0).expand(batch_size, self.predicted.size(0))
        return output


class Explainer(nn.Module):
    def __init__(self, vocab_size, max_len, n_bits):
        super().__init__()

        self.encoder = core.TransformerBaseEncoder(vocab_size=vocab_size + 2, 
                max_len=max_len + 1, 
                embed_dim=32, num_heads=4,
                hidden_size=64, num_layers=3)

        self.predictor = nn.Linear(32, n_bits)

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

class ReverseExplainer(nn.Module):
    def __init__(self, vocab_size, n_bits):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(n_bits, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, vocab_size),
        )

    def forward(self, bits):
        x = self.predictor(bits.float())
        return x

class Game(nn.Module):
    def __init__(self, masker, explainer_X, explainer_Y, bit_x, l, preference_x):
        super().__init__()

        self.masker = masker
        self.explainer_X = explainer_X
        self.explainer_Y = explainer_Y
        self.preference_x = preference_x

        self.l = l
        self.target_position = bit_x

    def forward(self, sequence, labels, _other=None):
        bit_x = self.target_position

        masked_sequence, logits, hard_mask = self.masker(sequence)
        predicted_Y = self.explainer_Y(masked_sequence)
        predicted_X = self.explainer_X(masked_sequence)

        #rows_y = [i for i in range(8) if i != bit_x]
        #labels_y = labels[:, rows_y]

        loss_X = F.binary_cross_entropy(predicted_X[:, 0], labels.float()[:, bit_x], reduction='none')#.mean(dim=-1)
        loss_Y = F.binary_cross_entropy(predicted_Y, labels.float(), reduction='none').mean(dim=-1)

        loss_XY = self.preference_x * loss_X - loss_Y
        loss_supp = (loss_XY.detach() * logits).mean()

        regularisation_loss = (1.0 - self.masker.prob_mask_logits.sigmoid()).pow(2.0).sum()

        loss = loss_X + loss_Y + loss_supp + self.l * regularisation_loss

        acc_X = ((predicted_X > 0.5).view(-1).long() == labels[:, bit_x]).float()
        acc_Y_mean = ((predicted_Y > 0.5).long() == labels).float().mean(dim=-1)

        nnz_att = hard_mask.float().sum(dim=-1)

        info = {'acc_X': acc_X, 'acc_Y_mean': acc_Y_mean, 'zeroes': nnz_att}
        
        for i in range(8):
            key = f'adv_acc_{i}' 
            if key not in info: info[key] = 0.0

            acc_Y = ((predicted_Y[:, i] > 0.5).view(-1).long() == labels[:, i]).float().mean(dim=0)
            info[key] += acc_Y


        return loss.mean(), info


class ReverseGame(nn.Module):
    def __init__(self, masker, explainer_X, target_position, l):
        super().__init__()

        self.masker = masker
        self.explainer_X = explainer_X

        self.l = l
        self.target_position = target_position

    def forward(self, utterance, bits, _other=None):
        x = self.target_position

        masked_bits, logits, hard_mask = self.masker(bits)
        predicted_X = self.explainer_X(masked_bits)

        loss_X = F.cross_entropy(predicted_X, utterance[:, x], reduction='none')
        loss_supp = (loss_X.detach() * logits).mean()

        regularisation_loss = (1.0 - self.masker.prob_mask_logits.sigmoid()).pow(2.0).sum()

        loss = loss_X + loss_supp + self.l * regularisation_loss

        acc_X = (predicted_X.argmax(dim=-1) == utterance[:, x]).float()
        nnz_att = hard_mask.float().sum(dim=-1)

        info = {'acc_X': acc_X, 'zeroes': nnz_att}
        
        return loss.mean(), info



class MinimalCoverGame(nn.Module):
    def __init__(self, masker, explainer_X, l):
        super().__init__()

        self.masker = masker
        self.explainer_X = explainer_X
        self.l = l

    def forward(self, sequence, labels, _other=None):
        masked_sequence, logits, hard_mask = self.masker(sequence)
        predicted_X = self.explainer_X(masked_sequence)

        loss_X = F.binary_cross_entropy(predicted_X, labels.float(), reduction='none').mean(dim=-1)
        loss_supp = (loss_X.detach() * logits).mean()
        regularisation_loss = (1.0 - self.masker.prob_mask_logits.sigmoid()).pow(2.0).sum()

        loss = loss_X + loss_supp + self.l * regularisation_loss
        acc_X = ((predicted_X > 0.5).long() == labels).float().mean(dim=-1)

        nnz_att = hard_mask.float().sum(dim=-1)

        info = {'acc_X_mean': acc_X, 'zeroes': nnz_att}
        
        for i in range(8):
            key = f'adv_acc_{i}' 
            if key not in info: info[key] = 0.0

            acc_Y = ((predicted_X[:, i] > 0.5).view(-1).long() == labels[:, i]).float().mean(dim=0)
            info[key] += acc_X


        return loss.mean(), info
