# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch.utils.data
import torch.nn.functional as F
import egg.core as core
from egg.zoo.lang_emerge.data import Dataset
from egg.zoo.lang_emerge.models import Answerer, Questioner, Game


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=100,
                        help='Size of the hidden layer of A/Q-Bot (default: 100)')
    parser.add_argument('--embedding', type=int, default=20,
                        help='Dimensionality of the embedding hidden layer for A/Q-Bot (default: 20)')

    parser.add_argument('--img_feat_size', default=20, type=int,\
                            help='Image feature size for each attribute')
    parser.add_argument('--q_out_vocab_size', default=3, type=int, help='Output vocab size for Q-Bot')
    parser.add_argument('--a_out_vocab_size', default=4, type=int, help='Output vocab size for A-Bot')
    parser.add_argument('--inflate', default=1, type=int)
    parser.add_argument('--memoryless_a', action='store_true')

    parser.add_argument('--entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient (default: 1e-1)')

    args = core.init(parser)

    return args


def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    assert False
    acc = (receiver_output.argmax(dim=1) == sender_input.argmax(dim=1)).detach().float()
    loss = F.cross_entropy(receiver_output, sender_input.argmax(dim=1), reduction="none")
    return loss, {'acc': acc}


if __name__ == "__main__":
    opts = get_params()
    device = torch.device("cuda" if opts.cuda else "cpu")
    
    train_dataset = Dataset('./data/toy64_split_0.8.json', mode='train', inflate=opts.inflate)
    test_dataset = Dataset('./data/toy64_split_0.8.json', mode='test')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opts.batch_size,
                                               shuffle=True
                                               )

    test_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opts.batch_size,
                                               shuffle=False
                                               )


    assert train_dataset.n_tasks == test_dataset.n_tasks

    task_vocab = ['<T%d>' % ii for ii in range(train_dataset.n_tasks)]
    q_out_vocab = [chr(ii + 97) for ii in range(opts.q_out_vocab_size)]
    q_out_vocab_size = len(q_out_vocab)
    a_out_vocab = [chr(ii + 65) for ii in range(opts.a_out_vocab_size)]
    a_out_vocab_size = len(a_out_vocab)

    a_in_vocab =  q_out_vocab + a_out_vocab
    q_in_vocab = a_out_vocab + q_out_vocab + task_vocab

    n_preds = train_dataset.attr_val_vocab

    q_task_offset = a_out_vocab_size + q_out_vocab_size
    q_listen_offset = a_out_vocab_size

    q_bot = Questioner(opts.batch_size, opts.hidden, opts.embedding, len(q_in_vocab), opts.q_out_vocab_size, n_preds, \
        q_task_offset, q_listen_offset)


    n_attrs = train_dataset.attr_val_vocab
    n_uniq_attrs = train_dataset.n_uniq_attrs

    a_bot = Answerer(opts.batch_size, opts.hidden, opts.embedding, len(a_in_vocab), a_out_vocab_size, \
             n_attrs, n_uniq_attrs, \
             opts.img_feat_size, q_out_vocab)

    game = Game(a_bot, q_bot, entropy_coeff=opts.entropy_coeff, memoryless_a=opts.memoryless_a)
    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=test_loader,
                           callbacks=[core.ConsoleLogger(as_json=True)])
    trainer.train(n_epochs=opts.n_epochs)

    core.close()

