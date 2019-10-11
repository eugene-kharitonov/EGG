# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch.utils.data
import torch.nn.functional as F
import egg.core as core
from egg.zoo.lang_emerge.data import Dataset
from egg.zoo.lang_emerge.models import AnswerAgent, QuestionAgent, Game


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=100,
                        help='Size of the hidden layer of the agents (default: 100)')
    parser.add_argument('--embedding', type=int, default=20,
                        help='Dimensionality of the embedding layer of the agents (default: 20)')

    parser.add_argument('--img_feat_size', default=20, type=int,
                        help='Image feature size for each attribute')

    parser.add_argument('--q_vocab_size', default=3, type=int,
                        help='Output vocab size for Question agent')
    parser.add_argument('--a_vocab_size', default=4, type=int,
                        help='Output vocab size for Answer agent')
    parser.add_argument('--inflate', default=10, type=int, help='Inflation rate of the training dataset: each point'
                        ' is repeated inflate times')
    parser.add_argument('--turns', default=2, type=int,
                        help='Number of turns in communication')
    parser.add_argument('--temperature', default=1.0,
                        type=float, help='Gumbel-Softmax temperature')
    parser.add_argument('--memoryless_a', action='store_true',
                        help='If set, Answer agent becomes memoryless')
    parser.add_argument('--straight_through', action='store_true',
                        help='If set, straight-through Gumbel Softmax is used')

    parser.add_argument('--entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient (default: 1e-1)')
    parser.add_argument('--data_path', default="/private/home/kharitonov/work/EGG/egg/zoo/lang_emerge/data/toy64_split_0.8.json",
                        help='Path to the data')

    args = core.init(arg_parser=parser, params=params)
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


class TemperatureUpdater(core.Callback):
    def __init__(self, agents, decay=0.9, minimum=0.1, update_frequency=1):
        self.agents = agents
        self.decay = decay
        self.minimum = minimum
        self.update_frequency = update_frequency
        self.epoch_counter = 0

    def on_epoch_end(self, loss, logs=None):
        if self.epoch_counter > 0 and self.epoch_counter % self.update_frequency == 0:
            for agent in self.agents:
                agent.gs.temperature = max(
                    self.minimum, agent.gs.temperature * self.decay)
        self.epoch_counter += 1


def main(params):
    import json
    opts = get_params(params)
    print(json.dumps(vars(opts)))

    device = torch.device("cuda" if opts.cuda else "cpu")

    train_dataset = Dataset(opts.data_path,
                            mode='train', inflate=opts.inflate)
    test_dataset = Dataset(opts.data_path, mode='test')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opts.batch_size,
                                               shuffle=True
                                               )

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=opts.batch_size,
                                              shuffle=False
                                              )

    assert train_dataset.n_tasks == test_dataset.n_tasks

    a_in_vocab = opts.q_vocab_size + opts.a_vocab_size
    q_in_vocab = opts.a_vocab_size + opts.q_vocab_size + train_dataset.n_tasks

    n_preds = train_dataset.attr_val_vocab

    q_task_offset = opts.a_vocab_size + opts.q_vocab_size
    q_listen_offset = opts.a_vocab_size

    q_bot = QuestionAgent(opts.hidden, opts.embedding, q_in_vocab, opts.q_vocab_size, n_preds,
                          q_task_offset, q_listen_offset, temperature=opts.temperature, straight_thru=opts.straight_through)

    n_attrs = train_dataset.attr_val_vocab
    n_uniq_attrs = train_dataset.n_uniq_attrs

    a_bot = AnswerAgent(opts.hidden, opts.embedding, a_in_vocab, opts.a_vocab_size,
                        n_attrs, n_uniq_attrs,
                        opts.img_feat_size, opts.q_vocab_size, temperature=opts.temperature, straight_thru=opts.straight_through)

    game = Game(a_bot, q_bot, entropy_coeff=opts.entropy_coeff,
                memoryless_a=opts.memoryless_a, steps=opts.turns)
    optimizer = core.build_optimizer(game.parameters())

    updater = TemperatureUpdater([q_bot, a_bot], decay=0.99, minimum=1.0)

    stopper = core.EarlyStopperAccuracy(
        threshold=1.0, field_name='acc', validation=False)
    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                           validation_data=test_loader,
                           callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=True), stopper])#, updater])
    trainer.train(n_epochs=opts.n_epochs)

    exit(0)
    print('*** TEST ***')
    dump_dialogs(game, test_loader, device)

    train_dataset = Dataset(
        './data/toy64_split_0.8.json', mode='train', inflate=1)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opts.batch_size,
                                               shuffle=False
                                               )
    print('*** TRAIN ***')
    dump_dialogs(game, train_loader, device)
    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
