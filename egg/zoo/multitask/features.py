# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import random

import numpy as np
import torch.utils.data as data
import torch

class _OneHotIterator:

    def __init__(self, n_batches_per_epoch, batches, shuffle_data=False):
        self.n_batches_per_epoch = n_batches_per_epoch
        self.batches_generated = 0
        self.batches = batches
        self.shuffle_data = shuffle_data

    def __iter__(self):
        self.batches_generated = 0
        if self.shuffle_data:
            random_idx = torch.randperm(self.batches.shape[0])
            self.batches = self.batches[random_idx, :, :]
        return self


    def __next__(self):
        if self.batches_generated >= self.n_batches_per_epoch:
            raise StopIteration()

        batch = self.batches[self.batches_generated, :, :].squeeze()
        self.batches_generated += 1

        return batch, torch.zeros(1)


class OneHotLoader:

    def __init__(self,
                 n_attributes,
                 n_values,
                 samples_per_epoch,
                 batch_size,
                 val_batch_size,
                 n_validation_samples=0,
                 validation_thr=None,
                 shuffle_train_data=False,
                 seed=None):

        self.n_attributes = n_attributes
        self.n_values = n_values

        self.n_validation_samples = n_validation_samples
        self.samples_per_epoch = samples_per_epoch

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        self.n_batches_per_epoch = int(samples_per_epoch) // self.batch_size
        self.n_val_batches = int(n_validation_samples) // self.val_batch_size

        self.validation_thr = 0.3 if validation_thr is None else validation_thr

        self.shuffle_train_data = shuffle_train_data

        if seed is None:
            self.seed = np.random.randint(0, 2 ** 32)
        else:
            self.seed = seed

        self.random_state = np.random.RandomState(seed)

        samples = list(itertools.product(*(range(n_values) for _ in range(n_attributes))))

        assert self.n_validation_samples / len(samples) < self.validation_thr

        tmp_validation_batches = self._batchify(samples, self.n_validation_samples, self.n_val_batches, self.val_batch_size, validation=True)
        training_batches = self._batchify(samples, self.samples_per_epoch, self.n_batches_per_epoch, self.batch_size)
        self.validation_batches = tmp_validation_batches.view(self.n_val_batches, self.val_batch_size, -1)

        each_value_tensor = self._get_each_attr_vector(self.n_attributes, self.n_values)

        # removing some samples to add tensors and make sure that each value is seen at least once 
        tmp_batches = training_batches[:(training_batches.shape[0] - each_value_tensor.shape[0]), :]
        training_batches = torch.cat((tmp_batches, each_value_tensor), 0)

        # randomize samples so that each_value_tensors are well distributed during training
        random_idx = torch.randperm(training_batches.shape[0])
        training_batches = training_batches[random_idx, :]
        self.training_batches = training_batches.view(self.n_batches_per_epoch, self.batch_size, -1)

        self.train_it = _OneHotIterator(self.n_batches_per_epoch, self.training_batches, shuffle_data=self.shuffle_train_data)
        self.valid_it = _OneHotIterator(self.n_val_batches, self.validation_batches)


    def _batchify(self, samples, n_samples_to_draw, n_batches, batch_size, validation=False):
        assert n_batches, 'number of batches must be non zero'
        samples_idx = self.random_state.randint(0, len(samples) - 1, size=(n_batches, batch_size))

        batches = []
        for batch_num in samples_idx:
            for sample_id in batch_num:
                if validation:
                    if sample_id >= len(samples):
                        sample_id = self.random_state.randint(0, len(samples) - 1)
                    to_add = self._one_hotify(samples.pop(sample_id))
                else:
                    to_add = self._one_hotify(samples[sample_id])
                batches.append(torch.unsqueeze(to_add, 0))
        return torch.cat(batches, 0)


    def _one_hotify(self, data):
        z = torch.zeros((self.n_attributes * self.n_values))
        for i, idx in enumerate(data):
            z[(i * self.n_values) + idx] = 1
        return z


    def _get_each_attr_vector(self, n_attributes, n_values):
        t = torch.zeros(self.n_attributes * self.n_values, self.n_attributes * self.n_values)

        for attr in range(self.n_attributes):
            start = attr * self.n_values
            for value in range(self.n_values):
                idx = start + value
                t[idx][idx] = 1  # set each attr of each value to one
                for remaining_attr in range(self.n_attributes):  # for each remaining value choose random attributes
                    if remaining_attr != attr:
                        start_rem = remaining_attr * self.n_values
                        random_idx = self.random_state.randint(low=0, high=self.n_attributes - 1)
                        t[idx][start_rem + random_idx] = 1

        random_idx = torch.randperm(t.shape[0])
        t = t[random_idx, :]
        return t


    def get_train_iterator(self):
        return self.train_it


    def get_validation_iterator(self):
        return self.valid_it
