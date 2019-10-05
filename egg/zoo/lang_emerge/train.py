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
                        help='Size of the hidden layer of the agents (default: 100)')
    parser.add_argument('--embedding', type=int, default=20,
                        help='Dimensionality of the embedding layer of the agents (default: 20)')

    parser.add_argument('--img_feat_size', default=20, type=int,\
                            help='Image feature size for each attribute')

    parser.add_argument('--q_vocab_size', default=3, type=int, help='Output vocab size for Q-Bot')
    parser.add_argument('--a_vocab_size', default=4, type=int, help='Output vocab size for A-Bot')
    parser.add_argument('--inflate', default=1, type=int)
    parser.add_argument('--turns', default=2, type=int, help='Number of communication turns')
    parser.add_argument('--temperature', default=1.0, type=float, help='Gumbel-Softmax temperature')
    parser.add_argument('--memoryless_a', action='store_true', help='If set, Answer agent becomes memoryless')

    parser.add_argument('--entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient (default: 1e-1)')

    args = core.init(parser)

    return args

def dump_dialogs(game, dataloader, device):
    game.eval()

    for batch, task, labels in dataloader:
        game.q_bot.reset()
        game.a_bot.reset()

        batch, task = batch.to(device), task.to(device)

        symbols, _ = game.do_rounds(batch, task)
        symbols = [s.argmax(dim=-1) for s in symbols]

        predictions = game.q_bot.predict(task)
        predictions = [p.argmax(dim=-1) for p in predictions]

        dataset = dataloader.dataset

        for i in range(batch.size(0)):
            t = task[i]
            s = [s[i].item() for s in symbols]
            p = [p[i].item() for p in predictions]
            inp = batch[i, :].tolist()
            label = labels[i, ...].tolist()

            l = f'input: {inp}, task: {dataset.tasks[t]}, communication: {s}, prediction: {p}, label: {label}'
            print(l)

if __name__ == "__main__":
    opts = get_params()
    device = torch.device("cuda" if opts.cuda else "cpu")
    
    train_dataset = Dataset('./data/toy64_split_0.8.json', mode='train', inflate=opts.inflate)
    #test_dataset = Dataset('./data/toy64_split_0.8.json', mode='train', inflate=1)
    test_dataset = Dataset('./data/toy64_split_0.8.json', mode='test')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opts.batch_size,
                                               shuffle=True
                                               )

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=opts.batch_size,
                                               shuffle=False
                                               )

    assert train_dataset.n_tasks == test_dataset.n_tasks

    task_vocab = ['<T%d>' % ii for ii in range(train_dataset.n_tasks)]
    q_out_vocab = [chr(ii + 97) for ii in range(opts.q_vocab_size)]
    a_out_vocab = [chr(ii + 65) for ii in range(opts.a_vocab_size)]

    a_in_vocab =  q_out_vocab + a_out_vocab
    q_in_vocab = a_out_vocab + q_out_vocab + task_vocab

    n_preds = train_dataset.attr_val_vocab

    q_task_offset = opts.a_vocab_size + opts.q_vocab_size
    q_listen_offset = opts.a_vocab_size

    q_bot = Questioner(opts.batch_size, opts.hidden, opts.embedding, len(q_in_vocab), opts.q_vocab_size, n_preds, \
        q_task_offset, q_listen_offset, temperature=opts.temperature)


    n_attrs = train_dataset.attr_val_vocab
    n_uniq_attrs = train_dataset.n_uniq_attrs

    a_bot = Answerer(opts.batch_size, opts.hidden, opts.embedding, len(a_in_vocab), opts.a_vocab_size, \
             n_attrs, n_uniq_attrs, \
             opts.img_feat_size, q_out_vocab, temperature=opts.temperature)

    game = Game(a_bot, q_bot, entropy_coeff=opts.entropy_coeff, memoryless_a=opts.memoryless_a, steps=opts.turns)
    optimizer = core.build_optimizer(game.parameters())

    stopper = core.EarlyStopperAccuracy(1.0, field_name='acc', validation=False)
    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=test_loader,
                           callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=True), stopper])
    trainer.train(n_epochs=opts.n_epochs)

    print('*** TEST ***')
    dump_dialogs(game, test_loader, device)

    train_dataset = Dataset('./data/toy64_split_0.8.json', mode='train', inflate=1)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opts.batch_size,
                                               shuffle=False
                                               )
    print('*** TRAIN ***')
    dump_dialogs(game, train_loader, device)
    core.close()

