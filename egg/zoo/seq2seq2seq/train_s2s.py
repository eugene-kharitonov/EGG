# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.utils.data as data
import egg.core as core
from egg.zoo.seq2seq2seq.features import SequenceData
from egg.zoo.seq2seq2seq.archs import Sender, Receiver, CopySender


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')
    parser.add_argument('--receiver_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Receiver (default: 10)')
    parser.add_argument('--receiver_cell', type=str, default='rnn',
                        help='Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)')
    parser.add_argument('--data_vocab_size', type=int, default=2,
                        help='Vocab size for the dataset')
    parser.add_argument('--data_max_len', type=int, default=3,
                        help='Max len for the dataset')
    parser.add_argument('--no_teacher_forcing', action='store_true', default=False,
                        help='Disable teacher forcing during training')
    parser.add_argument('--dataset_scaler', type=int, default=10)
    args = core.init(parser)

    return args


def accuracy_loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    output_sequence, _output_logprobs = receiver_output
    acc = (output_sequence == sender_input).float().mean(dim=1)
    return -acc, {'acc': acc}


def nll_teacher_forcing_loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    output_sequence, output_logprobs = receiver_output
    # TODO: find eos
    loss = 0.0
    assert sender_input.size(1) == output_logprobs.size(1)

    for step in range(sender_input.size(1)):
        nll = F.nll_loss(output_logprobs[:, step, :], sender_input[:, step])
        loss += nll

    acc = (output_sequence == sender_input).float().mean(dim=1)
    return loss, {'acc': acc}

if __name__ == "__main__":
    opts = get_params()

    device = torch.device("cuda" if opts.cuda else "cpu")

    train_dataset = SequenceData(max_len=opts.data_max_len, vocab_size=opts.data_vocab_size, scale_factor=opts.dataset_scaler)
    train_loader = data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)
    validation_dataset = SequenceData(max_len=opts.data_max_len, vocab_size=opts.data_vocab_size, scale_factor=1)
    validation_loader = data.DataLoader(validation_dataset, batch_size=opts.batch_size, shuffle=False)

    receiver = Receiver(input_vocab_size=opts.vocab_size, output_vocab_size=opts.data_vocab_size, emb_dim=opts.receiver_embedding, output_max_len=opts.data_max_len,
                     n_hidden=opts.receiver_hidden, encoder_cell=opts.receiver_cell, decoder_cell=opts.receiver_cell,
                     teacher_forcing=not opts.no_teacher_forcing)

    sender = CopySender()

    if opts.no_teacher_forcing:
        loss = accuracy_loss
    else:
        loss = nll_teacher_forcing_loss

    game = core.SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.05)
    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader, validation_data=validation_loader)

    trainer.train(n_epochs=opts.n_epochs)

    sender_inputs, messages, _, receiver_outputs, labels = \
        core.dump_sender_receiver(game, validation_loader, gs=False, device=device, variable_length=False)#True)

    for seq, message, output in zip(sender_inputs, messages, receiver_outputs):
        print(f'{seq} -> {message} -> {output}')

    core.close()

# TODO:
# * eos in inputs/outputs
# * joint training
# * separate encoder class from the Receiver everywhere
# * have Transformer encoder
# * dataset-based input