# class defintions for chatbots - questioner and answerer

import torch
import torch.nn as nn
from torch.distributions import Categorical


class Bot(nn.Module):
    def __init__(self, batch_size, hidden_size, embed_size, in_vocab_size, out_vocab_size):
        super().__init__()

        self.in_net = nn.Embedding(in_vocab_size, embed_size)
        self.out_net = nn.Linear(hidden_size, out_vocab_size)

        self.h_state = torch.zeros(batch_size, hidden_size)
        self.c_state = torch.zeros(batch_size, hidden_size)

    def reset(self):
        self.h_state = torch.zeros_like(self.h_state)
        self.c_state = torch.zeros_like(self.c_state)

    def listen(self, input_token, img_embed=None):
        # embed and pass through LSTM
        embeds = self.in_net(input_token)
        # concat with image representation
        if img_embed is not None:
            embeds = torch.cat((embeds, img_embed), dim=-1)

        # now pass it through rnn
        self.hState, self.cState = self.rnn(embeds, (self.hState, self.cState))

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

        # rnn inputSize
        rnn_input_size = n_uniq_attributes * img_feat_size + embed_size

        self.img_net = nn.Embedding(n_attributes, img_feat_size)
        self.rnn = nn.LSTMCell(rnn_input_size, hidden_size)

        # set offset
        self.listen_offset = q_out_vocab

    def embed_image(self, x):
        embeds = self.img_net(x)
        features = torch.cat(embeds.transpose(0, 1), 1);
        return features


class Questioner(Bot):
    def __init__(self, batch_size, hidden_size, embed_size, in_vocab_size, out_vocab_size,
            n_preds,
            task_offset,
            listen_offset):
        super().__init__(batch_size, hidden_size, embed_size, in_vocab_size, out_vocab_size)

        self.rnn = nn.LSTMCell(embed_size, hidden_size)

        # network for predicting
        self.predict_rnn = nn.LSTMCell(embed_size, hidden_size)
        self.predict_net = nn.Linear(hidden_size, n_preds)

        # setting offset
        self.task_offset = task_offset
        self.listen_offset = listen_offset

    # make a guess the given image
    def guess_attribute(self, input_embeds):
        # compute softmax and choose a token
        self.h_state, self.c_state = \
                self.predict_rnn(input_embeds, (self.h_state, self.c_state))
        logits = self.predict_net(self.h_state)

        distr = Categorical(logits=logits)
        entropy = distr.entropy()

        if self.training:
            sample = distr.sample()
        else:
            sample = logits.argmax(dim=1)
        log_prob = distr.log_prob(sample)

        return sample, log_prob, entropy

    # returning the answer, from the task
    def predict(self, tasks, n_tokens):
        samples = []
        log_probs = []
        entropies = []

        for _ in range(n_tokens):
            # explicit task dependence
            task_embeds = self.in_net(tasks)
            sample, log_prob, entropy = self.guess_attribute(task_embeds)

            samples.append(sample)
            log_probs.append(log_prob)
            entropies.append(entropy)

        return samples, log_probs, entropies

    def embed_task(self, tasks): 
        return self.in_net(tasks + self.task_offset)


class Game(nn.Module):
    # initialize
    def __init__(self, a_bot, q_bot, memoryless_a=True):
        super().__init__()
        # memorize params
        self.a_bot = a_bot
        self.q_bot = q_bot
        self.memoryless_a = memoryless_a

    def do_rounds(self, batch, tasks):
        batch_size = batch.size(0)
        self.q_bot.reset()
        self.a_bot.reset()

        img_embed = self.a_bot.embed_image(batch)

        a_bot_reply = tasks + self.q_bot.task_offset
        n_rounds = 2

        # if the conversation is to be recorded
        for round_id in range(n_rounds):
            self.q_bot.listen(a_bot_reply)
            q_bot_ques = self.q_bot.speak()

            self.q_bot.listen(self.q_bot.listen_offset + q_bot_ques)

            if self.memoryless_a:
                self.a_bot.reset()

            self.a_bot.listen(q_bot_ques, img_embed)
            a_bot_reply = self.a_bot.speak()
            self.a_bot.listen(a_bot_reply + self.a_bot.listen_offset, img_embed)

        self.q_bot.listen(a_bot_reply)

        # predict the image attributes, compute reward
        sample, logprobs, entropy = self.q_bot.predict(tasks, 2)

        return self.guessToken, self.guessDistr

    def forward(self, batch, tasks, labels):
        samples, logprobs, entropies = self.do_rounds(batch, tasks)

        first_match = (samples[0] == labels[:, 0:1]).float()
        second_match = (samples[1] == labels[:, 1:2]).float()

        reward = first_match * second_match

        loss = -reward * logprobs

        return loss
