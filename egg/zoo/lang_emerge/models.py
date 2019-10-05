# class defintions for chatbots - questioner and answerer

import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical


def init_lstm(lstm_cell):
    for name, param in lstm_cell.named_parameters():
        if 'bias' in name:
            torch.nn.init.zeros_(param)
        else:
            torch.nn.init.xavier_normal_(param)

class RelaxedEmbeddingWithOffset(nn.Embedding):
    def forward(self, x, offset=0):
        if isinstance(x, torch.LongTensor) or (torch.cuda.is_available() and isinstance(x, torch.cuda.LongTensor)):
            return F.embedding(x + offset, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            #print('padding', x.size(), self.weight.size())
            padded = torch.zeros(x.size(0), self.weight.size(0), device=x.device)
            padded[:, offset:offset+x.size(1)] += x
            #print('got tensor', x.size(), 'padded to', padded.size())
            return torch.matmul(padded, self.weight)


class Bot(nn.Module):
    def __init__(self, batch_size, hidden_size, embed_size, in_vocab_size, out_vocab_size, 
                temperature=1.0):
        super().__init__()

        self.in_net = RelaxedEmbeddingWithOffset(in_vocab_size, embed_size)
        self.out_net = nn.Linear(hidden_size, out_vocab_size)

        self.h_state = None
        self.c_state = None

        self.hidden_size = hidden_size
        self.in_vocab_size = in_vocab_size
        self.embedding_size = embed_size
        self.temperature = temperature

    def reset(self):
        self.h_state = None
        self.c_state = None

    def listen(self, input_token, img_embed=None, offset=0):
        embeds = self.in_net(input_token, offset)
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

        if not self.training:
            sample = torch.zeros_like(logits).scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        else:
            distr = RelaxedOneHotCategorical(logits=logits, temperature=self.temperature)
            sample = distr.rsample()

        return sample, Categorical(logits=logits).entropy()


class Answerer(Bot):
    def __init__(self, batch_size, hidden_size, embed_size, in_vocab_size, out_vocab_size,
                n_attributes,
                n_uniq_attributes,
                img_feat_size,
                q_out_vocab, temperature=1.0):
        super().__init__(batch_size, hidden_size, embed_size, in_vocab_size, out_vocab_size, temperature=temperature)

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
            listen_offset, temperature):
        super().__init__(batch_size, hidden_size, embed_size, in_vocab_size, out_vocab_size, temperature=temperature)

        self.rnn = nn.LSTMCell(embed_size, hidden_size)

        self.predict_net_0 = nn.Linear(hidden_size, n_preds, bias=False)
        self.predict_net_1 = nn.Linear(hidden_size, n_preds, bias=False)

        self.task_offset = task_offset
        self.listen_offset = listen_offset
        self.init_params()

    def init_params(self):
        #torch.nn.init.xavier_normal_(self.predict_net.weight)
        #torch.nn.init.zeros_(self.predict_net.bias)
        torch.nn.init.xavier_normal_(self.in_net.weight)
        torch.nn.init.xavier_normal_(self.out_net.weight)
        torch.nn.init.zeros_(self.out_net.bias)
        init_lstm(self.rnn)
        #init_lstm(self.predict_rnn)

    def predict(self, tasks):
        predictions = [
            self.predict_net_0(self.h_state),
            self.predict_net_1(self.h_state)
            ]
        return predictions


class Game(nn.Module):
    def __init__(self, a_bot, q_bot, entropy_coeff, memoryless_a=True, 
            steps=2):
        super().__init__()

        self.steps = steps
        self.a_bot = a_bot
        self.q_bot = q_bot
        self.memoryless_a = memoryless_a
        self.entropy_coeff = entropy_coeff

        self.mean_baseline = 0.0
        self.n_points = 0.0

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
            self.a_bot.listen(a_bot_reply, offset=self.a_bot.listen_offset, img_embed=img_embed)
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

        first_match = F.cross_entropy(predictions[0], labels[:, 0], reduction='none')
        second_match = F.cross_entropy(predictions[1], labels[:, 1], reduction='none')
        entropy = sum(entropies).mean()

        loss = first_match + second_match - self.entropy_coeff * entropy

        acc = first_acc * second_acc

        return loss.mean(), {'first_acc': first_acc.mean(), 'second_acc': second_acc.mean(), 
                                'acc': acc.mean(), 'entropy': entropy}
