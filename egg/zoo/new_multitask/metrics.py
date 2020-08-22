# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import json
import pathlib
import pickle
from typing import List

from egg.core import Callback#, #Evaluator
from egg.core.util import move_to
from egg.zoo.multitask.util import ask_sender, compute_binomial, get_task_embedding, get_output_file

import editdistance #package to install https://pypi.org/project/editdistance/0.3.1/
from scipy import spatial
from scipy.stats import spearmanr
import torch


class EvaluateTopSim(Callback):
    def __init__(self, n_attributes, n_values, dataset, dataset_name, sender, device, output_file_path, task_embedding, num_tasks=1, msg_path=None, store_language=False):
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.sender = sender
        self.device = device
        self.output_file_path = output_file_path
        self.task_embedding = task_embedding
        self.num_tasks = num_tasks
        self.msg_path = msg_path
        self.store_language = store_language

    def _compute_distance(self, _list, distance):
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

    def topographic_similarity(self):
        _attributes, strings, meanings, meanings_topsim_specific, lengths = ask_sender(self.n_attributes,
                                                                                       self.n_values,
                                                                                       self.dataset,
                                                                                       self.sender,
                                                                                       self.device,
                                                                                       self.task_embedding,
                                                                                       self.num_tasks)

        list_strings = [[[char_idx.item() for char_idx in msg] for msg in string] for string in strings]

        list_edit_dist = []
        corr, corr_topsim_specific = [], []

        cos_dist = self._compute_distance(meanings.cpu().numpy(), 'cosine')
        for string in list_strings:
            edit_dist = self._compute_distance(string, 'edit')
            list_edit_dist.append(edit_dist)
            corr.append(spearmanr(cos_dist, edit_dist).correlation)

        if self.store_language:
            with open(f'{self.msg_path}/cos_dist', 'wb') as fp:
                pickle.dump([cos_dist], fp)
            with open(f'{self.msg_path}/edit_dist', 'wb') as fp:
                pickle.dump(list_edit_dist, fp)
            with open(f'{self.msg_path}/msgs', 'wb') as fp:
                pickle.dump(list_strings, fp)
            torch.save(meanings, f'{self.msg_path}/meanings.pt')

        if all(map(lambda x: len(x), meanings_topsim_specific)):
            assert len(meanings_topsim_specific) == len(list_edit_dist)
            for cos, edit_dist in zip(meanings_topsim_specific, list_edit_dist):
                corr_topsim_specific.append(spearmanr(self._compute_distance(cos.cpu().numpy(), 'cosine'), edit_dist).correlation)
        return corr, corr_topsim_specific, lengths

    def compute_metrics(self):
        topsims, topsim_specific, lengths = self.topographic_similarity()
        with open(self.output_file_path, 'a') as w:
            for idx, topsim in enumerate(topsims):
                d = { f'topsim_{self.dataset_name}-task_embedding_{idx}': topsim }
                d1 = { f'topsim_{self.dataset_name}-task_embedding_{idx}_lengths': lengths[idx] }
                json.dump(d, w)
                w.write('\n')
                json.dump(d1, w)
                w.write('\n')

            for idx, topsim in enumerate(topsim_specific):
                d = { f'topsim_specific_{self.dataset_name}-task_embedding_{idx}': topsim }
                json.dump(d, w)
                w.write('\n')

    def on_train_end(self):
        self.compute_metrics()


class EvaluateAccuracy(Callback):
    def __init__(self, games, dataset, dataset_name, metric_name, device, output_file, task_embedding=False):
        self.games = games
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.device = device
        self.metric_name = metric_name
        self.output_file_path = output_file
        self.task_embedding = task_embedding

    def _add_dicts(self, a, b):
        result = dict(a)
        for k, v in b.items():
            result[k] = result.get(k, 0) + v
        return result

    def _div_dict(self, d, n):
        result = dict(d)
        for k in result:
            result[k] /= n
        return result

    def compute_accuracy(self):
        mean_rest = {}
        each_game_acc = defaultdict(float)

        n_batches = 0
        for game in self.games:
            game.eval()
        with torch.no_grad():
            for batch in self.dataset:
                batch = move_to(batch, self.device)
                for i, game in enumerate(self.games):
                    task_embedding_tensor = None
                    if self.task_embedding:
                        task_embedding_tensor = move_to(get_task_embedding(i), self.device)
                    _ , rest = game(*batch, task_embedding=task_embedding_tensor)
                    mean_rest = self._add_dicts(mean_rest, rest)
                    loss_type = '-'.join([str(attr) for attr in game.loss.skip_attributes])
                    loss_type = loss_type if loss_type else '-1'
                    each_game_acc[loss_type] = each_game_acc[loss_type] + rest['acc']

                mean_rest = self._div_dict(mean_rest, len(self.games))
                n_batches += 1
        each_game_acc = self._div_dict(each_game_acc, n_batches)
        mean_rest = self._div_dict(mean_rest, n_batches)
        with open(self.output_file_path, 'a') as w:
            mean_rest = { f'{self.metric_name}_{self.dataset_name}': mean_rest[self.metric_name] }
            json.dump(mean_rest, w)
            w.write('\n')
            json.dump(each_game_acc, w)
            w.write('\n')

    def on_train_end(self):
        self.compute_accuracy()
