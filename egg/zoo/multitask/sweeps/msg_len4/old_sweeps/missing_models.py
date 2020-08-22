# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# not necessary, but could be useful
def dict2string(d):
    s = []

    for k, v in d.items():
        if type(v) in (int, float):
            s.append(f"--{k}={v}")
        elif type(v) is bool and v:
            s.append(f"--{k}")
        elif type(v) is str:
            assert '"' not in v, f"Key {k} has string value {v} which contains forbidden quotes."
            s.append(f'--{k}={v}')
        else:
            raise Exception(
                f"Key {k} has value {v} of unsupported type {type(v)}.")
    return s


def grid():
    """
    Should return an iterable of the parameter strings, e.g.
    `--param1=value1 --param2`
    """

    n_values = 30
    max_len = 5
    n_attributes = 3
    early_stopping_thr = 0.9999
    n_epoch = 300
    evaluate_train_set_limit = 10000
    lr = 0.001
    receiver_embedding = 30
    sender_embedding = 30

    # 2-13
    # bs512_vocab50_rec_h512_sen_h_256
    for cell in ['gru', 'lstm']:
        for mtask in [0, 1]:
            for multi in range(1, 11):
                seed = multi * 111
                params = dict(batch_size=512, lr=lr, receiver_cell=cell , receiver_embedding=receiver_embedding, receiver_hidden=512, sender_cell=cell, sender_embedding=sender_embedding, sender_hidden=256, vocab_size=50, n_values=n_values, max_len=max_len, n_attributes=n_attributes, early_stopping=early_stopping_thr, n_epoch=n_epoch, evaluate_train_set_limit=evaluate_train_set_limit, mtask=mtask, random_seed=seed, output='/private/home/rdessi/output_stats_missing_models')
                yield dict2string(params)

    # 10-20
    # bs1024_lr0.001_vocab100_rec_h256_sen_h_512
    for cell in ['gru', 'rnn']:
        for mtask in [0, 1]:
            for multi in range(1, 11):
                seed = multi * 111
                params = dict(batch_size=1024, lr=lr, receiver_cell=cell , receiver_embedding=receiver_embedding, receiver_hidden=256, sender_cell=cell, sender_embedding=sender_embedding, sender_hidden=512, vocab_size=100, n_values=n_values, max_len=max_len, n_attributes=n_attributes, early_stopping=early_stopping_thr, n_epoch=n_epoch, evaluate_train_set_limit=evaluate_train_set_limit, mtask=mtask, random_seed=seed, output='/private/home/rdessi/output_stats_missing_models')
                yield dict2string(params)

#if __name__ == '__main__':
#    grid()
