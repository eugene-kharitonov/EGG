# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import defaultdict
import pathlib
import pickle
import random

import editdistance #package to install https://pypi.org/project/editdistance/0.3.1/
import numpy as np
from scipy import spatial
from scipy.stats import spearmanr
from timebudget import timebudget
import torch

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True,
                        help='path to where messages and meanings are stored')
    parser.add_argument('--topsim', action='store_true',
                        help='if set, topographic similarity will be computed')
    parser.add_argument('--shuffle_topsim', action='store_true', default=False,
                        help='if set, topsim baselines will be computed')
    parser.add_argument('--posdis', action='store_true',
                        help='if set, positional disentanglement will be computed')
    parser.add_argument('--compute_baseline', action='store_true',
                        help='if set, topsim baselines will be computed')
    parser.add_argument('--overlap', action='store_true')

    parser.add_argument('--seed', default=111, type=int,
                        help='seed for deterministic runs')
    return vars(parser.parse_args())

def load_tensors(path):
    path = pathlib.Path(path)

    path_mean = path / 'meanings.pt'
    meanings = torch.load(f'{path_mean}')
    path_msgs = path / 'msgs.pt'
    msgs = torch.load(f'{path_msgs}')
    return meanings, msgs

def get_meaning_specific(meanings):
    def get_meaning(meaning, idx):
        if idx == 0:
            return meaning
        if idx == 1:
            return meaning[idx, :, 30:]
        if idx == 2:
            return torch.cat([meaning[idx, :, :30], meaning[idx, :, 60:]], dim=0)
        if idx == 3:
            return meaning[idx, :, :60]
        if idx == 4:
            return meaning[idx, :, 60:]
        if idx == 5:
            return meaning[idx, :, 30:60]
        if idx == 6:
            return meaning[idx, : , :30]

    assert meanings.shape[0] == 7  # only works for 3 attribute tasks now
    meaning_topsim_spceific = [[]] * 7
    for msg_idx in meanings.shape[0]:
        meaning_for_topsim = get_meaning(meaning, msg_idx)
        meaning_topsim_specific[msg_idx].append(meaning_for_topsim.to(device))
    return meaning_topsim_spceific

def entropy_dict(freq_table):
    H = 0
    n = sum(v for v in freq_table.values())

    for m, freq in freq_table.items():
        p = freq_table[m] / n
        H += -p * np.log(p)
    return H / np.log(2)

def entropy(messages):
    freq_table = defaultdict(float)
    for m in messages:
        m = _hashable_tensor(m)
        freq_table[m] += 1.0
    return entropy_dict(freq_table)

def _hashable_tensor(t):
    if isinstance(t, tuple):
        return t
    if isinstance(t, int):
        return t

    try:
        t = t.item()
    except:
        t = tuple(t.view(-1).tolist())
    return t

def mutual_info(xs, ys):
    e_x = entropy(xs)
    e_y = entropy(ys)

    xys = []

    for x, y in zip(xs, ys):
        xy = (_hashable_tensor(x), _hashable_tensor(y))
        xys.append(xy)

    e_xy = entropy(xys)

    return e_x + e_y - e_xy

def compute_distance(_list, distance):
    distances = []
    for i, el1 in enumerate(_list[:-1]):
        for j, el2 in enumerate(_list[i+1:]):
            normalization_len = len(el1) + len(el2)
            if distance == 'edit':
                distances.append(editdistance.eval(el1, el2) / normalization_len)
            elif distance == 'cosine':
                distances.append(spatial.distance.cosine(el1, el2))
            else:
                raise Exception('I cannot compute {distance} distance')
    return distances

def compute_topsim(meanings=None,
                   strings=None,
                   list_edit_dist=None,
                   list_cos_dist=None,
                   shuffle=False,
                   save_dist=False,
                   meaning_specific=False):

    if not (list_edit_dist and list_cos_dist):
        list_strings = []
        for i, msg_batch in enumerate(strings):
            list_strings.append([msg.tolist() for msg in msg_batch])

        list_edit_dist = []
        list_cos_dist = []
        corr = []

        for string in list_strings:
            list_edit_dist.append(compute_distance(string, 'edit'))

        if meaning_specific:
            for meaning in meanings:
                list_cos_dist.append(compute_distance(meaning.cpu().numpy(), 'cosine'))
        else:
            list_cos_dist.append(compute_distance(meanings[0].cpu().numpy(), 'cosine'))

        if save_dist:
            with open('/private/home/rdessi/cos_dist', 'wb') as fp:
                pickle.dump(list_cos_dist, fp)
            with open('/private/home/rdessi/edit_dist', 'wb') as fp:
                pickle.dump(list_edit_dist, fp)

    corr = []
    if meaning_specific:
        print(len(list_cos_dist), len(list_edit_dist))
        assert len(list_cos_dist) == len(list_edit_dist)
        for edit_dist, cos_dist in zip(list_edit_dist, list_cos_dist):
            random.shuffle(edit_dist)
            print('a')
            corr.append(spearmanr(list_cos_dist, edit_dist).correlation)
    else:
        for edit_dist in list_edit_dist:
            corr.append(spearmanr(list_cos_dist[0], edit_dist).correlation)
    return corr

def overlap(strings):
    overlap12_2 = round(torch.mean(torch.sum((strings[1] == strings[-3]).float(), dim=1)).item(), 2)
    overlap12_1 = round(torch.mean(torch.sum((strings[1] == strings[-2]).float(), dim=1)).item(), 2)
    overlap12_0 = round(torch.mean(torch.sum((strings[1] == strings[-1]).float(), dim=1)).item(), 2)
    print(f'overlap12_2 {overlap12_2}, overlap12_1 {overlap12_1}, overlap12_0 {overlap12_0}')

    overlap02_2 = round(torch.mean(torch.sum((strings[2] == strings[-3]).float(), dim=1)).item(), 2)
    overlap02_1 = round(torch.mean(torch.sum((strings[2] == strings[-2]).float(), dim=1)).item(), 2)
    overlap02_0 = round(torch.mean(torch.sum((strings[2] == strings[-1]).float(), dim=1)).item(), 2)
    print(f'overlap02_2 {overlap02_2}, overlap02_1 {overlap02_1}, overlap02_0 {overlap02_0}')

    overlap01_2 = round(torch.mean(torch.sum((strings[3] == strings[-3]).float(), dim=1)).item(), 2)
    overlap01_1 = round(torch.mean(torch.sum((strings[3] == strings[-2]).float(), dim=1)).item(), 2)
    overlap01_0 = round(torch.mean(torch.sum((strings[3] == strings[-1]).float(), dim=1)).item(), 2)
    print(f'overlap01_2 {overlap01_2}, overlap01_1 {overlap01_1}, overlap01_0 {overlap01_0}')

def main(args):
    #meanings, msgs = load_tensors(args['input_path'])
    random.seed(111)
    p = pathlib.Path(args['input_path'])
    p_edit = p / 'edit_dist'
    p_cos = p / 'cos_dist'

    list_edit_dist, list_cos_dist = [], []
    with open(p_edit, 'rb') as f:
        list_edit_dist = pickle.load(f)
    with open(p_cos, 'rb') as f:
        list_cos_dist = pickle.load(f)

    print(compute_topsim(list_edit_dist=list_edit_dist, list_cos_dist=list_cos_dist, meaning_specific=True))

    '''
    if args['topsim']:
        print(compute_topsim(meanings, msgs))
        if args['meaning_specific']:
            meaning_specific = get_meaning_specific(meanings)  # python list of tensors of diff sizes, cannot cat them. They are 7 x 5000 x {30, 60, 90}
            print(compute_topsim(meaning_specific, msgs))
    if args['overlap']:
        overlap(msgs)
    '''

if __name__ == '__main__':
    args = get_params()
    main(args)


'''
def compute_posdis(meanings, representations):
    #meanings = meanings.view(-1, 3, 30).argmax(dim=-1)
    gaps = torch.zeros(representations.size(1))
    non_constant_positions = 0.0
    for j in range(representations.size(1)):
        symbol_mi = []
        h_j = None
        for i in range(meanings.size(1)):
            x, y = meanings[:, i], representations[:, j]
            info =  mutual_info(x, y)
            symbol_mi.append(info)
            if h_j is None:
                h_j = entropy(y)
        symbol_mi.sort(reverse=True)
        if h_j > 0.0:
            gaps[j] = (symbol_mi[0] - symbol_mi[1]) / h_j
            non_constant_positions += 1
    score = gaps.sum() / non_constant_positions
    return score.item()
'''
