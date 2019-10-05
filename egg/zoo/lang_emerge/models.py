# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical

from egg import core


class RelaxedEmbeddingWithOffset(nn.Embedding):
    """
    >>> emb = RelaxedEmbeddingWithOffset(10, 20)
    >>> q = torch.LongTensor([1, 2, 3])
    >>> ground_truth = emb(q)
    >>> lookup = emb(q - 1, offset=1)
    >>> ground_truth.allclose(lookup)
    True
    >>> q_as_one_hot = torch.zeros(3, 10).scatter_(1, q.view(-1, 1), 1)
    >>> lookup = emb(q_as_one_hot)
    >>> ground_truth.allclose(lookup)
    True
    >>> q_offset_as_one_hot = torch.zeros(3, 7).scatter_(1, (q - 1).view(-1, 1), 1)
    >>> lookup = emb(q_offset_as_one_hot, offset=1)
    >>> ground_truth.allclose(lookup)
    True
    """

    def forward(self, x, offset=0):
        if isinstance(x, torch.LongTensor) or (torch.cuda.is_available() and isinstance(x, torch.cuda.LongTensor)):
            return F.embedding(x + offset, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            return torch.matmul(x, self.weight[offset:offset + x.size(1), :])


class Agent(nn.Module):
    def __init__(self, hidden_size, embed_size, in_vocab_size, out_vocab_size,
                 temperature=1.0, straight_thru=False):
        super().__init__()

        self.in_net = RelaxedEmbeddingWithOffset(in_vocab_size, embed_size)
        self.out_net = nn.Linear(hidden_size, out_vocab_size)

        self.h_state = None
        self.c_state = None

        self.hidden_size = hidden_size

        self.gs = core.GumbelSoftmaxLayer(
            temperature, straight_through=straight_thru)

    def reset(self):
        self.h_state = None
        self.c_state = None

    def listen(self, input_token, img_embed=None, offset=0):
        raise NotImplementedError("Use a sublass")

        embeds = self.in_net(input_token, offset)
        if img_embed is not None:
            embeds = torch.cat((embeds, img_embed), dim=-1)

        # now pass it through rnn
        if self.h_state is None:
            batch_size = embeds.size(0)
            self.h_state = torch.zeros(
                batch_size, self.hidden_size, device=embeds.device)
            self.c_state = torch.zeros_like(self.h_state)

        self.h_state, self.c_state = self.rnn(
            embeds, (self.h_state, self.c_state))

    def speak(self):
        logits = self.out_net(self.h_state)
        sample = self.gs(logits)

        return sample, Categorical(logits=logits).entropy()


class AnswerAgent(Agent):
    def __init__(self, hidden_size, embed_size, in_vocab_size, out_vocab_size,
                 n_attributes,
                 n_uniq_attributes,
                 img_feat_size,
                 q_out_vocab, temperature=1.0, straight_thru=False):
        super().__init__(hidden_size, embed_size, in_vocab_size, out_vocab_size,
                         temperature=temperature, straight_thru=straight_thru)

        rnn_input_size = n_uniq_attributes * img_feat_size + embed_size

        self.img_net = nn.Embedding(n_attributes, img_feat_size)
        self.rnn = nn.LSTMCell(rnn_input_size, hidden_size)

        self.listen_offset = q_out_vocab
        self.init_params()

    def init_params(self):
        torch.nn.init.xavier_normal_(self.img_net.weight)
        torch.nn.init.xavier_normal_(self.in_net.weight)
        torch.nn.init.xavier_normal_(self.out_net.weight)
        torch.nn.init.zeros_(self.out_net.bias)

    def embed_image(self, x):
        # NB: different in the original code; appends ones
        embeds = self.img_net(x)
        embeds = embeds.view(embeds.size(0), -1)
        return embeds

    def listen(self, input_token, img_embed, offset=0):
        embeds = self.in_net(input_token, offset)
        embeds = torch.cat((embeds, img_embed), dim=-1)

        if self.h_state is None:
            batch_size = embeds.size(0)
            self.h_state = torch.zeros(
                batch_size, self.hidden_size, device=embeds.device)
            self.c_state = torch.zeros_like(self.h_state)

        self.h_state, self.c_state = self.rnn(
            embeds, (self.h_state, self.c_state))


class QuestionAgent(Agent):
    def __init__(self, hidden_size, embed_size, in_vocab_size, out_vocab_size,
                 n_preds,
                 task_offset,
                 listen_offset, temperature, straight_thru=False):
        super().__init__(hidden_size, embed_size, in_vocab_size, out_vocab_size,
                         temperature=temperature, straight_thru=straight_thru)

        self.rnn = nn.LSTMCell(embed_size, hidden_size)

        self.predict_net_0 = nn.Linear(hidden_size, n_preds, bias=False)
        self.predict_net_1 = nn.Linear(hidden_size, n_preds, bias=False)

        self.task_offset = task_offset
        self.listen_offset = listen_offset
        self.init_params()

    def init_params(self):
        torch.nn.init.xavier_normal_(self.in_net.weight)
        torch.nn.init.xavier_normal_(self.out_net.weight)
        torch.nn.init.zeros_(self.out_net.bias)

    def predict(self, tasks):
        predictions = [
            self.predict_net_0(self.h_state),
            self.predict_net_1(self.h_state)
        ]
        return predictions

    def listen(self, input_token, offset=0):
        embeds = self.in_net(input_token, offset)

        if self.h_state is None:
            batch_size = embeds.size(0)
            self.h_state = torch.zeros(
                batch_size, self.hidden_size, device=embeds.device)
            self.c_state = torch.zeros_like(self.h_state)

        self.h_state, self.c_state = self.rnn(
            embeds, (self.h_state, self.c_state))


class Game(nn.Module):
    def __init__(self, a_bot, q_bot, entropy_coeff, memoryless_a=True,
                 steps=2):
        super().__init__()

        self.steps = steps
        self.a_bot = a_bot
        self.q_bot = q_bot
        self.memoryless_a = memoryless_a
        self.entropy_coeff = entropy_coeff

        entropies = []

    def do_rounds(self, batch, tasks):
        dialog = []
        entropies = []
        img_embed = self.a_bot.embed_image(batch)

        self.q_bot.listen(tasks.squeeze(1), offset=self.q_bot.task_offset)

        for round_id in range(self.steps):
            ques, q_entropy = self.q_bot.speak()
            self.q_bot.listen(ques, offset=self.q_bot.listen_offset)

            if self.memoryless_a:
                self.a_bot.reset()

            self.a_bot.listen(ques, img_embed)
            a_bot_reply, a_entropy = self.a_bot.speak()
            self.a_bot.listen(
                a_bot_reply, offset=self.a_bot.listen_offset, img_embed=img_embed)
            self.q_bot.listen(a_bot_reply)

            dialog.extend([ques.detach(), a_bot_reply.detach()])
            entropies.extend([q_entropy, a_entropy])

        return dialog, entropies

    def forward(self, batch, tasks, labels):
        self.q_bot.reset()
        self.a_bot.reset()

        _, entropies = self.do_rounds(batch, tasks)

        predictions = self.q_bot.predict(tasks)

        first_acc = (predictions[0].argmax(dim=-1) == labels[:, 0]).float()
        second_acc = (predictions[1].argmax(dim=-1) == labels[:, 1]).float()

        first_match = F.cross_entropy(
            predictions[0], labels[:, 0], reduction='none')
        second_match = F.cross_entropy(
            predictions[1], labels[:, 1], reduction='none')
        entropy = sum(entropies).mean()

        loss = first_match + second_match - self.entropy_coeff * entropy

        acc = first_acc * second_acc

        return loss.mean(), {'first_acc': first_acc.mean(),
                             'second_acc': second_acc.mean(),
                             'acc': acc.mean(),
                             'entropy': entropy}
