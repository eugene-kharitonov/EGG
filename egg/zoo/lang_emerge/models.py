# class defintions for chatbots - questioner and answerer

import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


def init_lstm(lstm_cell):
    for name, param in lstm_cell.named_parameters():
        if 'bias' in name:
            torch.nn.init.zeros_(param)
        else:
            torch.nn.init.xavier_normal_(param)

class Bot(nn.Module):
    def __init__(self, batch_size, hidden_size, embed_size, in_vocab_size, out_vocab_size):
        super().__init__()

        self.in_net = nn.Embedding(in_vocab_size, embed_size)
        self.out_net = nn.Linear(hidden_size, out_vocab_size)

        self.h_state = None
        self.c_state = None

        self.hidden_size = hidden_size

    def reset(self):
        self.h_state = None
        self.c_state = None

    def listen(self, input_token, img_embed=None):
        embeds = self.in_net(input_token)
        if img_embed is not None:
            embeds = torch.cat((embeds, img_embed), dim=-1)

        # now pass it through rnn
        if self.h_state is None:
            batch_size = embeds.size(0)
            self.h_state = torch.zeros(batch_size, self.hidden_size, device=embeds.device)
            self.c_state = torch.zeros_like(self.h_state)

        self.h_state, self.c_state = self.rnn(embeds, (self.h_state, self.c_state))

    def speak(self):
        logits = self.out_net(self.h_state)

        distr = Categorical(logits=logits)
        entropy = distr.entropy()

        if self.training:
            sample = distr.sample()
        else:
            sample = logits.argmax(dim=1)
        log_prob = distr.log_prob(sample)

        return sample, log_prob, entropy


class Answerer(Bot):
    def __init__(self, batch_size, hidden_size, embed_size, in_vocab_size, out_vocab_size,
                n_attributes,
                n_uniq_attributes,
                img_feat_size,
                q_out_vocab):
        super().__init__(batch_size, hidden_size, embed_size, in_vocab_size, out_vocab_size)

        rnn_input_size = n_uniq_attributes * img_feat_size + embed_size

        self.img_net = nn.Embedding(n_attributes, img_feat_size)
        self.rnn = nn.LSTMCell(rnn_input_size, hidden_size)

        self.listen_offset = len(q_out_vocab)
        self.init_params()

    def init_params(self):
        torch.nn.init.xavier_normal_(self.img_net.weight)
        torch.nn.init.xavier_normal_(self.in_net.weight)
        torch.nn.init.xavier_normal_(self.out_net.weight)
        torch.nn.init.zeros_(self.out_net.bias)

        init_lstm(self.rnn)

    def embed_image(self, x):
        # NB: different in the original code; appends ones
        embeds = self.img_net(x)
        embeds = embeds.view(embeds.size(0), -1)
        return embeds


class Questioner(Bot):
    def __init__(self, batch_size, hidden_size, embed_size, in_vocab_size, out_vocab_size,
            n_preds,
            task_offset,
            listen_offset):
        super().__init__(batch_size, hidden_size, embed_size, in_vocab_size, out_vocab_size)

        self.rnn = nn.LSTMCell(embed_size, hidden_size)

        self.predict_rnn = nn.LSTMCell(embed_size, hidden_size)
        self.predict_net = nn.Linear(hidden_size, n_preds)

        self.task_offset = task_offset
        self.listen_offset = listen_offset
        self.init_params()

    def init_params(self):
        torch.nn.init.xavier_normal_(self.predict_net.weight)
        torch.nn.init.zeros_(self.predict_net.bias)
        torch.nn.init.xavier_normal_(self.in_net.weight)
        torch.nn.init.xavier_normal_(self.out_net.weight)
        torch.nn.init.zeros_(self.out_net.bias)
        init_lstm(self.rnn)
        init_lstm(self.predict_rnn)

    def predict(self, tasks, n_tokens):
        predictions = []

        for _ in range(n_tokens):
            task_embeds = self.in_net(tasks).squeeze(1)
            self.h_state, self.c_state = \
                self.predict_rnn(task_embeds, (self.h_state, self.c_state))
            logits = self.predict_net(self.h_state)
            predictions.append(logits)

        return predictions

    def predict_sample(self, tasks, n_tokens):
        samples = []
        logprobs = []
        entropies = []

        for _ in range(n_tokens):
            task_embeds = self.in_net(tasks).squeeze(1)
            self.h_state, self.c_state = \
                self.predict_rnn(task_embeds, (self.h_state, self.c_state))
            logits = self.predict_net(self.h_state)

            distr = Categorical(logits=logits)
            entropy = distr.entropy()

            if self.training:
                sample = distr.sample()
            else:
                sample = logits.argmax(dim=1)
            logprob = distr.log_prob(sample)

            samples.append(sample)
            logprobs.append(logprob)
            entropies.append(entropy)

        return samples, logprobs, entropies

class Game(nn.Module):
    def __init__(self, a_bot, q_bot, entropy_coeff, memoryless_a=True, 
            steps=2, loss='diff'):
        super().__init__()

        self.steps = steps
        self.a_bot = a_bot
        self.q_bot = q_bot
        self.memoryless_a = memoryless_a
        self.entropy_coeff = entropy_coeff

        self.mean_baseline = 0.0
        self.n_points = 0.0
        self.loss_type = loss

    def get_dialog(self, batch, tasks):
        batch_size = batch.size(0)
        self.q_bot.reset()
        self.a_bot.reset()

        img_embed = self.a_bot.embed_image(batch)

        a_bot_reply = tasks + self.q_bot.task_offset
        a_bot_reply = a_bot_reply.squeeze(1)

        symbols = []

        for round_id in range(self.steps):
            self.q_bot.listen(a_bot_reply)
            q_bot_ques, *_ = self.q_bot.speak()

            self.q_bot.listen(self.q_bot.listen_offset + q_bot_ques)

            if self.memoryless_a:
                self.a_bot.reset()

            self.a_bot.listen(q_bot_ques, img_embed)
            a_bot_reply, *_ = self.a_bot.speak()
            self.a_bot.listen(a_bot_reply + self.a_bot.listen_offset, img_embed)

            symbols.extend([q_bot_ques, a_bot_reply])

        self.q_bot.listen(a_bot_reply)
        predictions = self.q_bot.predict(tasks, 2)
        return symbols, predictions

    def do_rounds(self, batch, tasks):
        img_embed = self.a_bot.embed_image(batch)

        a_bot_reply = tasks + self.q_bot.task_offset
        a_bot_reply = a_bot_reply.squeeze(1)

        sum_log_probs = 0.0
        sum_entropies = 0.0

        for round_id in range(self.steps):
            self.q_bot.listen(a_bot_reply)
            q_bot_ques, q_logprobs, q_entropy = self.q_bot.speak()

            self.q_bot.listen(self.q_bot.listen_offset + q_bot_ques)

            if self.memoryless_a:
                self.a_bot.reset()

            self.a_bot.listen(q_bot_ques, img_embed)
            a_bot_reply, a_logprobs, a_entropy = self.a_bot.speak()
            self.a_bot.listen(a_bot_reply + self.a_bot.listen_offset, img_embed)

            sum_log_probs += q_logprobs + a_logprobs
            sum_entropies += q_entropy + a_entropy

        self.q_bot.listen(a_bot_reply)
        return sum_log_probs, sum_entropies

    def forward(self, batch, tasks, labels):
        self.q_bot.reset()
        self.a_bot.reset()

        logprobs, entropies = self.do_rounds(batch, tasks)

        if self.loss_type == 'diff':
            predictions = self.q_bot.predict(tasks, 2)

            first_acc = (predictions[0].argmax(dim=-1) == labels[:, 0]).float()
            second_acc = (predictions[1].argmax(dim=-1) == labels[:, 1]).float()

            first_match = F.cross_entropy(predictions[0], labels[:, 0], reduction='none')
            second_match = F.cross_entropy(predictions[1], labels[:, 1], reduction='none')
            loss = first_match + second_match
        else:
            predictions, predict_logprobs, predict_entropies = self.q_bot.predict_sample(tasks, 2)
            logprobs += sum(predict_logprobs)
            entropies += sum(predict_entropies)
            first_acc = (predictions[0] == labels[:, 0]).float()
            second_acc = (predictions[1] == labels[:, 1]).float()
            if self.loss_type == 'sum':
                loss = first_acc + second_acc
            elif self.loss_type == 'both':
                both = first_acc * second_acc
                loss = -10 * (both < 0.5).float() + (both > 0.5).float()


        policy_loss = ((loss.detach() - self.mean_baseline) * logprobs).mean()
        optimized_loss = loss.mean() + policy_loss - entropies.mean() * self.entropy_coeff

        if self.training:
            self.n_points += 1.0
            self.mean_baseline += (loss.detach().mean().item() -
                                   self.mean_baseline) / self.n_points

        acc = first_acc * second_acc

        return optimized_loss, {'first_acc': first_acc.mean(), 'second_acc': second_acc.mean(), 
                                'acc': acc.mean(), 'baseline': self.mean_baseline}
