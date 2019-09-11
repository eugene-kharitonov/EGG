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


class Answerer(ChatBot):
    def __init__(self, batch_size, hidden_size, embed_size, in_vocab_size, out_vocab_size,
                n_attributes,
                n_uniq_attributes,
                img_feat_size,
                q_out_vocab):
        super().__init__(batch_size, hidden_size, embed_size, in_vocab_size, out_vocab_size)

        # rnn inputSize
        rnn_input_size = n_uniq_attributes * img_feat_size + embed_size;

        self.img_net = nn.Embedding(n_attributes, img_feat_size)
        self.rnn = nn.LSTMCell(rnn_input_size, hidden_size)

        # set offset
        self.listen_offset = q_out_vocab

    def embed_image(self, x):
        embeds = self.img_net(x)
        features = torch.cat(embeds.transpose(0, 1), 1);
        return features


class Questioner(ChatBot):
    def __init__(self, batch_size, hidden_size, embed_size, in_vocab_size, out_vocab_size,
            n_preds):
        self.rnn = nn.LSTMCell(embed_size, hidden_size)

        # network for predicting
        self.predict_rnn = nn.LSTMCell(embed_size, hidden_size)
        self.predict_net = nn.Linear(hidden_size, n_preds)

        # setting offset
        self.task_offset = task_offset #params['aOutVocab'] + params['qOutVocab'];
        self.listen_offset = listen_offset #params['aOutVocab'];

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


class TwoRoundGame(nn.Module):
    def __init__(self, q_agent, a_agent, loss, sender_entropy_coeff=0.0):
        super().__init__()

        self.q_agent = q_agent
        self.a_agent = a_agent
        self.loss = loss

        self.receiver_entropy_coeff = receiver_entropy_coeff

    def forward(self, sender_input, labels, receiver_input=None):
        message, sender_log_prob, sender_entropy = self.sender(sender_input)
        receiver_output, receiver_log_prob, receiver_entropy = self.receiver(message, receiver_input)

        loss, rest_info = self.loss(sender_input, message, receiver_input, receiver_output, labels)
        policy_loss = ((loss.detach() - self.mean_baseline) * (sender_log_prob + receiver_log_prob)).mean()
        entropy_loss = -(sender_entropy.mean() * self.sender_entropy_coeff + receiver_entropy.mean() * self.receiver_entropy_coeff)

        if self.training:
            self.n_points += 1.0
            self.mean_baseline += (loss.detach().mean().item() -
                                   self.mean_baseline) / self.n_points

        full_loss = policy_loss + entropy_loss + loss.mean()

        for k, v in rest_info.items():
            if hasattr(v, 'mean'):
                rest_info[k] = v.mean().item()

        rest_info['baseline'] = self.mean_baseline
        rest_info['loss'] = loss.mean().item()
        rest_info['sender_entropy'] = sender_entropy.mean()
        rest_info['receiver_entropy'] = receiver_entropy.mean()

        return full_loss, rest_info

