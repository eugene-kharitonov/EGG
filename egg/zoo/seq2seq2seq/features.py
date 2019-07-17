# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import pickle
import torch.utils.data as data
import torch.nn.parallel
import os
import torch
import numpy as np



class SequenceData:
    """
    Always eos (zero) terminated strings
    Returns:
        [type] -- [description]
    """
    sequences: List[torch.tensor]

    def __init__(self, vocab_size: int, max_len: int, scale_factor: int=1):
        # TODO: make uniform wrt zero-terminated stuff
        assert max_len > 1
        assert scale_factor > 0
        max_len -= 1  # account for the zero-termination
        self.scale_factor = scale_factor

        self.sequences = []
        for i in range(vocab_size ** max_len):
            sequence = []
            for j in range(max_len):
                sequence.append(i % vocab_size)
                i = i // vocab_size

                if sequence[-1] == 0:
                    to_add = max_len - len(sequence)
                    sequence.extend([0] * to_add)
                    break

            sequence.append(0)  # append eos if it didn't happen before
            self.sequences.append(sequence)

        self.sequences = list(set([tuple(s) for s in self.sequences]))  # make them uniq
        self.sequences.sort()
        self.sequences = [torch.tensor(s, dtype=torch.long) for s in self.sequences]

    def __getitem__(self, i):
        i = i % len(self.sequences)
        # sender input, labels, receiver input
        return self.sequences[i], torch.zeros(1), self.sequences[i]


    def __len__(self):
        return self.scale_factor * len(self.sequences)


if __name__ == '__main__':
    dataset = SequenceData(vocab_size=4, max_len=4)
    print(len(dataset))
    data_loader = data.DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in data_loader:
        print(batch)