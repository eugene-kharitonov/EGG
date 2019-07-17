# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F
import egg.core as core
import torch
from torch.distributions import Categorical


class Sender(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, emb_dim, n_hidden, max_len, num_layers=1, encoder_cell='rnn', decoder_cell='rnn'):
        super(Sender, self).__init__()
        encoder = Encoder(encoder_cell, emb_dim, n_hidden, input_vocab_size)
        self.pseudo_agent = core.RnnSenderReinforce(encoder, output_vocab_size, emb_dim, n_hidden, max_len, num_layers, decoder_cell, force_eos=False)#True)

    def forward(self, x):
        result = self.pseudo_agent(x)
        return result


class Receiver(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, emb_dim, n_hidden, output_max_len, teacher_forcing, num_layers=1, encoder_cell='rnn', decoder_cell='rnn'):
        super(Receiver, self).__init__()
        self.encoder = Encoder(encoder_cell, emb_dim, n_hidden, input_vocab_size)
        self.decoder = Decoder(output_vocab_size, emb_dim, n_hidden, output_max_len, teacher_forcing, num_layers, decoder_cell)
        self.teacher_forcing = teacher_forcing

    def forward(self, message, receiver_input, message_lengths=None):
        encoded_input = self.encoder(message, message_lengths)

        sequence, log_probs, entropies = self.decoder(encoded_input, receiver_input if self.teacher_forcing else None)
        return sequence, log_probs.sum(dim=-1), entropies.sum(dim=-1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_hidden, max_len, teacher_forcing, num_layers=1, cell='rnn'):
        super(Decoder, self).__init__()

        self.hidden_to_output = nn.Linear(n_hidden, vocab_size)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(emb_dim))
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_len = max_len
        self.teacher_forcing = teacher_forcing

        self.cell = None
        cell = cell.lower()
        cell_types = {'rnn': nn.RNNCell, 'gru': nn.GRUCell, 'lstm': nn.LSTMCell}

        if cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")
        cell_type = cell_types[cell]

        self.cells = nn.ModuleList([
            cell_type(input_size=emb_dim, hidden_size=n_hidden) if i == 0 else \
            cell_type(input_size=n_hidden, hidden_size=n_hidden) for i in range(self.num_layers)])

    def forward(self, encoder_state, ground_truth=None):
        batch_size = encoder_state.size(0)

        prev_hidden = [encoder_state] # TODO: attention mechanism
        prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)])

        prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * batch_size)

        sequence = []
        logits = []
        entropy = []
        symbol_logprobs = []

        if not self.teacher_forcing or not self.training:
            # teacher forcing is disabled, doing reinforce instead
            for step in range(self.max_len):
                for i, layer in enumerate(self.cells):
                    if isinstance(layer, nn.LSTMCell):
                        h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                        prev_c[i] = c_t
                    else:
                        h_t = layer(input, prev_hidden[i])
                    prev_hidden[i] = h_t
                    input = h_t

                step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
                distr = Categorical(logits=step_logits)
                entropy.append(distr.entropy())

                if self.training:
                    x = distr.sample()
                else:
                    x = step_logits.argmax(dim=1)
                logits.append(distr.log_prob(x))
                sequence.append(x)
                symbol_logprobs.append(step_logits)

                input = self.embedding(x)
        else:
            assert ground_truth is not None
            actual_length = ground_truth.size(1)
            loss = 0.0
            for step in range(actual_length):
                for i, layer in enumerate(self.cells):
                    if isinstance(layer, nn.LSTMCell):
                        h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                        prev_c[i] = c_t
                    else:
                        h_t = layer(input, prev_hidden[i])
                    prev_hidden[i] = h_t
                    input = h_t

                step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)

                # not used for training, hence we do argmax all the way
                sequence.append(step_logits.argmax(dim=-1))
                symbol_logprobs.append(step_logits)
                logits.append(torch.zeros_like(step_logits[:, 0]))
                entropy.append(logits[-1])

                input = self.embedding(ground_truth[:, step])


        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)
        symbol_logprobs = torch.stack(symbol_logprobs).permute(1, 0, 2)

        return (sequence, symbol_logprobs), logits, entropy


class Encoder(nn.Module):
    # TODO: encoder should be a part of Receiver wrapper
    # TODO: add Transformer as an abstraction
    # TODO: attention

    def __init__(self, cell, emb_dim, n_hidden, vocab_size):
        super(Encoder, self).__init__()

        self.encoder_cell = None
        cell = cell.lower()
        if cell == 'rnn':
            self.cell = nn.RNN(input_size=emb_dim, batch_first=True, hidden_size=n_hidden, num_layers=1)
        elif cell == 'gru':
            self.cell = nn.GRU(input_size=emb_dim, batch_first=True, hidden_size=n_hidden, num_layers=1)
        elif cell == 'lstm':
            self.cell = nn.LSTM(input_size=emb_dim, batch_first=True, hidden_size=n_hidden, num_layers=1)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, message, message_lengths=None):
        emb = self.embedding(message)
        if message_lengths is None:
            message_lengths = core.find_lengths(message)

        packed = nn.utils.rnn.pack_padded_sequence(emb, message_lengths, batch_first=True, enforce_sorted=False)
        _, rnn_hidden = self.cell(packed)
        if isinstance(rnn_hidden, tuple):  # lstm returns c_n, don't need it
            rnn_hidden = rnn_hidden[0]
        hidden = rnn_hidden.squeeze(0)
        return hidden



class CopySender(nn.Module):

    def __init__(self):
        return super().__init__()


    def forward(self, x):
        batch_size = x.size(0)

        entropy = torch.zeros_like(x).float()
        logprob = torch.zeros_like(x).float()

        return x, entropy, logprob
