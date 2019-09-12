# class defintions for chatbots - questioner and answerer

import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

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
        # embed and pass through LSTM
        embeds = self.in_net(input_token)
        # concat with image representation
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

        # rnn inputSize
        rnn_input_size = n_uniq_attributes * img_feat_size + embed_size

        self.img_net = nn.Embedding(n_attributes, img_feat_size)
        self.rnn = nn.LSTMCell(rnn_input_size, hidden_size)

        # set offset
        self.listen_offset = len(q_out_vocab)

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
        return logits

    # returning the answer, from the task
    def predict(self, tasks, n_tokens):
        predictions = []

        for _ in range(n_tokens):
            # explicit task dependence
            task_embeds = self.in_net(tasks).squeeze(1)
            logits = self.guess_attribute(task_embeds)
            predictions.append(logits)

        return predictions

    def embed_task(self, tasks): 
        return self.in_net(tasks + self.task_offset)


class Game(nn.Module):
    # initialize
    def __init__(self, a_bot, q_bot, entropy_coeff, memoryless_a=True, steps=2):
        super().__init__()

        self.steps = steps
        self.a_bot = a_bot
        self.q_bot = q_bot
        self.memoryless_a = memoryless_a
        self.entropy_coeff = entropy_coeff

        self.mean_baseline = 0.0
        self.n_points = 0.0

    def do_rounds(self, batch, tasks):
        batch_size = batch.size(0)
        self.q_bot.reset()
        self.a_bot.reset()

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

        # predict the image attributes, compute reward
        predictions = self.q_bot.predict(tasks, 2)

        return predictions, sum_log_probs, sum_entropies

    def forward(self, batch, tasks, labels):
        predictions, logprobs, entropies = self.do_rounds(batch, tasks)

        first_match = F.cross_entropy(predictions[0], labels[:, 0], reduce='none')
        second_match = F.cross_entropy(predictions[1], labels[:, 1], reduce='none')
        
        first_acc = (predictions[0].argmax(dim=-1) == labels[:, 0]).float().mean()
        second_acc = (predictions[1].argmax(dim=-1) == labels[:, 1]).float().mean()

        loss = first_match + second_match

        if self.training:
            self.n_points += 1.0
            self.mean_baseline += (loss.detach().mean().item() -
                                   self.mean_baseline) / self.n_points

        policy_loss = ((loss.detach() - self.mean_baseline) * logprobs).mean()
        optimized_loss = loss + policy_loss - entropies.mean() * self.entropy_coeff

        return optimized_loss, {'first_acc': first_acc, 'second_acc': second_acc, 'baseline': self.mean_baseline}
