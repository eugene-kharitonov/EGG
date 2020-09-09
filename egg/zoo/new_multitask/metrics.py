# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import json
from typing import List

import editdistance
from scipy.spatial import distance
from scipy.stats import spearmanr
import torch

from egg.core import Callback, Interaction, PosDisent
from egg.core.util import move_to



class Evaluator(Callback):
    def __init__(self, game, dataset, device, n_tasks = 1, is_gumbel = False):
        self.game = game
        self.dataset = dataset
        self.device = device
        self.n_tasks = n_tasks
        self.is_gumbel= is_gumbel

    def topsim(self, attributes, messages, sender_input_distance_fn = 'cosine', message_distance_fn = 'edit'):
        distances = {'edit': lambda x, y: editdistance.eval(x, y) / (len(x) + len(y)) / 2,
                     'cosine': distance.cosine,
                     'hamming': distance.hamming,
                     'jaccard': distance.jaccard,
                     'euclidean': distance.euclidean,
                     }

        sender_input_distance_fn = distances.get(sender_input_distance_fn, None) \
            if isinstance(sender_input_distance_fn, str) else sender_input_distance_fn
        message_distance_fn = distances.get(message_distance_fn, None) \
            if isinstance(message_distance_fn, str) else message_distance_fn

        assert sender_input_distance_fn and message_distance_fn, f"Cannot recognize {sender_input_distance_fn} or {message_distance_fn} distances"

        def _compute_distance(_list, distance):
            return [distance(el1, el2)
                    for i, el1 in enumerate(_list[:-1])
                    for j, el2 in enumerate(_list[i+1:])
                    ]

        messages = [msg.tolist() for msg in messages]

        input_dist = _compute_distance(
            attributes.numpy(), sender_input_distance_fn)
        message_dist = _compute_distance(messages, message_distance_fn)

        topsim = spearmanr(input_dist, message_dist,
                           nan_policy='raise').correlation
        return topsim

    def get_interactions(self):
        interactions = defaultdict(list)

        self.game.eval()
        with torch.no_grad():
            for batch in self.dataset:
                batch = move_to(batch, self.device)
                for task in range(self.n_tasks):
                    loss, interaction = self.game(*batch)
                    interactions[task].append(interaction.to('cpu'))

        full_interactions = [Interaction.from_iterable(v) for k, v in interactions.items()]
        return full_interactions

    def print_info(self):
        idx2loss = {'0': '-1', '1': '0', '2': '1', '3': '2', '4': '0-1', '5': '0-2', '6': '1-2'}

        interactions = self.get_interactions()
        for idx, interaction in enumerate(interactions):

            messages = interaction.message.argmax(dim=-1) if self.is_gumbel else interaction.message
            attributes = interaction.sender_input

            acc = interaction.aux['acc'].mean().item()
            posdis = PosDisent.posdis(attributes, messages)
            topsim = self.topsim(attributes, messages)

            output = dict(acc=acc, topsim=topsim, posdis=posdis, task=idx2loss[str(idx)])
            print(json.dumps(output), flush=True)

    def on_train_end(self):
        self.print_info()
