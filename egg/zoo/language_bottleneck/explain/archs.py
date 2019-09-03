# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli

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

class GsMasker(nn.Module):
    def __init__(self, max_len, prior=0.0, mask=None, temperature=1.0):
        super().__init__()

        self.prob_mask_logits = torch.zeros(max_len).fill_(prior)
        if mask is not None:
            assert len(mask) == max_len, f'{mask} {max_len}'
            for i, v in enumerate(mask):
                if v == 'x': self.prob_mask_logits[i] = +100
        self.prob_mask_logits = torch.nn.Parameter(self.prob_mask_logits)
        self.temperature = temperature
        self.pre_mask = mask

    def forward(self, sequence):
        batch_size = sequence.size(0)
        extended_logits = self.prob_mask_logits.unsqueeze(0).expand((batch_size, sequence.size(1)))

        mask = LogitRelaxedBernoulli(logits=extended_logits, temperature=self.temperature).rsample()
        if self.training:
            mask = mask.softmax(dim=-1)
        else:
            mask = torch.zeros_like(mask).scatter_(-1, mask.argmax(dim=-1, keepdim=True), 1.0)

        return mask


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


class GsExplainer(nn.Module):
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
    def __init__(self, target_position, n_bits, vocab_size, mask=None, prior=0.0, l=1e-2):
        super().__init__()

        self.masker = GsMasker(n_bits, prior=prior, mask=mask)
        self.explainer = nn.Sequential(
            nn.Linear(n_bits, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, vocab_size),
        )

        self.l = l
        self.target_position = target_position

    def forward(self, utterance, bits, _other=None):
        bits = bits.float()
        x = self.target_position
        mask = self.masker(bits)

        bits = (1.0 - mask) * bits
        predicted_X = self.explainer(bits)

        loss_X = F.cross_entropy(predicted_X, utterance[:, x], reduction='none')
        regularisation_loss = (1.0 - self.masker.prob_mask_logits.sigmoid()).pow(2.0).sum()

        loss = loss_X + self.l * regularisation_loss

        acc_X = (predicted_X.argmax(dim=-1) == utterance[:, x]).float()
        nnz_att = mask.float().sum(dim=-1)

        info = {'acc_X': acc_X, 'zeroes': nnz_att}
        
        return loss.mean(), info


class GsMinimalCoverGame(nn.Module):
    def __init__(self, vocab_size, max_len, n_bits, l):
        super().__init__()

        self.masker = GsMasker(max_len)
        self.cell = nn.LSTM(input_size=32, batch_first=True,
                               hidden_size=128, num_layers=1)
        self.embedding = nn.Embedding(vocab_size, 32, padding_idx=0)

        self.mask_embedding = nn.Parameter(torch.zeros(32))
        nn.init.normal_(self.mask_embedding)

        self.fc = nn.Linear(128, n_bits)
        self.l = l

    def forward(self, sequence, labels, _other=None):
        embedded = self.embedding(sequence)
        batch_size, seq_len, emb_dim = embedded.size()

        mask = self.masker(sequence)
        mask = mask.unsqueeze(2).expand(batch_size, seq_len, emb_dim)

        shaped_mask_embedding = self.mask_embedding.unsqueeze(0).unsqueeze(0).expand_as(mask)

        embedded = (1.0 - mask) * embedded + mask * shaped_mask_embedding
        _, (hidden, _) = self.cell(embedded)

        predicted_X = self.fc(hidden[-1])

        loss_X = F.binary_cross_entropy(predicted_X.sigmoid(), labels.float())
        regularisation_loss = (1.0 - self.masker.prob_mask_logits.sigmoid()).pow(2.0).sum()

        loss = loss_X + self.l * regularisation_loss
        acc_X = ((predicted_X > 0.5).long() == labels).float().mean(dim=-1)

        zeros = mask.float().sum(dim=-1).sum(dim=-1)

        info = {'acc_X_mean': acc_X, 'zeroes': zeros}
        
        for i in range(8):
            key = f'adv_acc_{i}' 
            if key not in info: info[key] = 0.0

            acc_Y = ((predicted_X[:, i] > 0.5).view(-1).long() == labels[:, i]).float().mean(dim=0)
            info[key] += acc_Y


        return loss.mean(), info
