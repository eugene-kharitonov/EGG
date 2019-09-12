# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import argparse
import torch.utils.data
import torch.nn.functional as F
import egg.core as core
from egg.zoo.lang_emerge.data import Dataset
from egg.zoo.lang_emerge.models import Answerer, Questioner, Game


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--q_hidden', type=int, default=50,
                        help='Size of the hidden layer of Q-Bot (default: 50)')
    parser.add_argument('--a_hidden', type=int, default=50,
                        help='Size of the hidden layer of A-Bot (default: 50)')

    parser.add_argument('--q_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Q-Bot (default: 10)')
    parser.add_argument('--a_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for A-Bot (default: 10)')

    parser.add_argument('--img_feat_size', default=20, type=int,\
                            help='Image feature size for each attribute');
    parser.add_argument('--q_out_vocab_size', default=3, type=int, help='Output vocab size for Q-Bot')
    parser.add_argument('--a_out_vocab_size', default=4, type=int, help='Output vocab size for A-Bot')

    #parser.add_argument('--sender_entropy_coeff', type=float, default=1e-1,
    #                    help='The entropy regularisation coefficient for Sender (default: 1e-1)')
    #parser.add_argument('--receiver_entropy_coeff', type=float, default=1e-1,
    #                    help='The entropy regularisation coefficient for Receiver (default: 1e-1)')

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
    
    train_dataset = Dataset('./data/toy64_split_0.8.json', mode='train')
    test_dataset = Dataset('./data/toy64_split_0.8.json', mode='test')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opts.batch_size,
                                               shuffle=True)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=256,
                                               shuffle=False)


    assert train_dataset.n_tasks == test_dataset.n_tasks

    task_vocab = ['<T%d>' % ii for ii in range(train_dataset.n_tasks)]
    q_out_vocab = [chr(ii + 97) for ii in range(opts.q_out_vocab_size)]
    q_out_vocab_size = len(q_out_vocab)
    a_out_vocab = [chr(ii + 65) for ii in range(opts.a_out_vocab_size)]
    a_out_vocab_size = len(a_out_vocab)

    a_in_vocab =  q_out_vocab + a_out_vocab
    q_in_vocab = a_out_vocab + q_out_vocab + task_vocab

    # pack parameters
    #self.params = {'numTasks': self.numTasks, 'taskSelect': self.taskDefn,\
    #                'props': self.props, 'attributes': self.attributes,\
    #                'qOutVocab':len(qOutVocab), 'qInVocab':len(qInVocab),\
    #                'aOutVocab':len(aOutVocab), 'aInVocab':len(aInVocab)};

    n_preds = train_dataset.attr_val_vocab

    q_task_offset = a_out_vocab_size + q_out_vocab_size
    q_listen_offset = a_out_vocab_size

    q_bot = Questioner(opts.batch_size, opts.q_hidden, opts.q_embedding, len(q_in_vocab), opts.q_out_vocab_size, n_preds, \
        q_task_offset, q_listen_offset)


    n_attrs = train_dataset.n_attrs
    n_uniq_attrs = train_dataset.n_uniq_attrs

    a_bot = Answerer(opts.batch_size, opts.a_hidden, opts.a_embedding, len(a_in_vocab), a_out_vocab_size, \
             n_attrs, n_uniq_attrs, \
             opts.img_feat_size, q_out_vocab)

    game = Game(a_bot, q_bot)

    for batch, tasks, labels in train_loader:
        game(batch, tasks, labels)


    if opts.mode.lower() == 'rf':
        sender = core.RnnSenderReinforce(sender,
                                         opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                         cell=opts.sender_cell, max_len=opts.max_len, force_eos=opts.force_eos)
        receiver = core.RnnReceiverDeterministic(receiver, opts.vocab_size, opts.receiver_embedding,
                                                 opts.receiver_hidden, cell=opts.receiver_cell)

        game = core.SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff=opts.receiver_entropy_coeff)
        callbacks = []
    elif opts.mode.lower() == 'gs':
        sender = core.RnnSenderGS(sender, opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                  cell=opts.sender_cell, max_len=opts.max_len, temperature=opts.temperature,
                                  force_eos=opts.force_eos)

        receiver = core.RnnReceiverGS(receiver, opts.vocab_size, opts.receiver_embedding,
                    opts.receiver_hidden, cell=opts.receiver_cell)

        game = core.SenderReceiverRnnGS(sender, receiver, loss)
        callbacks = [core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)]
    else:
        raise NotImplementedError(f'Unknown training mode, {opts.mode}')

    optimizer = torch.optim.Adam([
        {'params': game.sender.parameters(), 'lr': opts.sender_lr},
        {'params': game.receiver.parameters(), 'lr': opts.receiver_lr}
    ])

    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=test_loader,
                           callbacks=callbacks + [core.ConsoleLogger(as_json=True)])
    trainer.train(n_epochs=opts.n_epochs)

    core.close()

