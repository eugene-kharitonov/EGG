# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import json
import pathlib

import torch

from egg.core.util import move_to, find_lengths


def compute_binomial(n, k):
	if 0 <= k <= n:
		ntok = 1
		ktok = 1
		for t in range(1, min(k, n - k) + 1):
			ntok *= n
			ktok *= t
			n -= 1
		return ntok // ktok
	else:
		return 0

def get_meaning(meaning, idx):
    if idx == 0:
        return meaning
    if idx == 1:
        return meaning[30:]
    if idx == 2:
        return torch.cat([meaning[:30], meaning[60:]], dim=0)
    if idx == 3:
        return meaning[:60]
    if idx == 4:
        return meaning[60:]
    if idx == 5:
        return meaning[30:60]
    if idx == 6:
        return meaning[:30]

def ask_sender(n_attributes,
               n_values,
               dataset,
               sender,
               device,
               task_embedding=False,
               num_tasks=1):

    attributes = []
    meanings = []
    strings = [[] for _ in range(num_tasks)]
    lengths = [[] for _ in range(num_tasks)]
    meanings_topsim_specific = [[] for _ in range(num_tasks)]

    sender.eval()

    for batch, _ in dataset:
        if len(batch.shape) == 1:  # handling case where batch_size=1
            batch = batch.unsqueeze(0)
        for meaning in batch:
            attribute = meaning.view(n_attributes, n_values).argmax(dim=-1)
            attributes.append(attribute)
            meanings.append(meaning.to(device))

            with torch.no_grad():
                if num_tasks > 1:
                    for idx in range(num_tasks):
                        if task_embedding:
                            task_embedding_tensor = move_to(get_task_embedding(idx), device)
                            string, *other = sender(meaning.unsqueeze(0).to(device), task_embedding_tensor)
                        else:
                            string, *other = sender(meaning.unsqueeze(0).to(device))

                        lengths[idx].append(find_lengths(string).float().mean().item())
                        strings[idx].append(string.squeeze(0))

                        meaning_specific = get_meaning(meaning, idx)
                        meanings_topsim_specific[idx].append(meaning_specific.to(device))
                else:
                    string, *other = sender(meaning.unsqueeze(0).to(device))
                    strings[0].append(string.squeeze(0))
                    lengths[0].append(find_lengths(string).float().mean().item())

    lengths = [sum(v) / len(v) for v in lengths]

    attributes = torch.stack(attributes, dim=0)
    meanings = torch.stack(meanings, dim=0)

    # meanings specific e.g. input object only with properties important for the task
    if all(map(lambda x: len(x), meanings_topsim_specific)):
        meanings_topsim_specific = [torch.stack(v, dim=0) for v in meanings_topsim_specific]

    strings = [torch.stack(v, dim=0) for v in strings]

    sender.train()
    return attributes, strings, meanings, meanings_topsim_specific, lengths

def get_task_embedding(idx, dim=7):
    embedding_tensor = torch.zeros(dim)
    embedding_tensor[idx] = 1
    return embedding_tensor

def get_output_file(opts):
        folder_path = pathlib.Path(opts.output)
        folder_path.mkdir(parents=True, exist_ok=True)
        filename = f'bs_{opts.batch_size}-lr_{opts.lr}-vocab_{opts.vocab_size}-rechidden_{opts.receiver_hidden}-senhidden_{opts.sender_hidden}-recemb_{opts.receiver_embedding}-sendemb_{opts.sender_embedding}-cell_{opts.sender_cell}-mtask_{opts.multitask}-taskembed_{opts.task_embedding}seed_{opts.random_seed}'
        final_path = pathlib.Path(opts.output) / filename
        with open(final_path, 'a') as fd:
            fd.write(str(opts))
        return final_path
