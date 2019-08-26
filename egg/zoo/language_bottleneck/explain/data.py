# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.utils.data as data
import torch.nn.parallel
import torch
import numpy as np


def read_file(fname, prediction_mask, source_mask):
    inputs = []
    codes = []

    with open(fname, 'r') as fin:
        for line in fin:
            if line.startswith('#'): continue
            line = line.split()

            inp = [int(x) for x in line[0]]
            assert len(prediction_mask) == len(inp)
            inp = [z for (z, m) in zip(inp, prediction_mask) if m == '?']
            inp = torch.tensor(inp, dtype=torch.int64)
            inputs.append(inp)

            code = [int(x) for x in line[2:-2]]
            codes.append(code)

    max_code_len = max(len(x) for x in codes)
    average_len = sum(len(x) for x in codes) / len(codes)
    print(f'# Max length is {max_code_len}, average code length is {average_len}')
    expanded_codes = []

    for code in codes:
        positions_need = max_code_len - len(code)
        extension = [0 for _ in range(positions_need)]
        code.extend(extension)
        assert len(code) == max_code_len

        if source_mask:
            assert len(code) == len(source_mask), f'{len(code)} {len(source_mask)}'
            for i, s in enumerate(source_mask):
                if s == 'x':
                    code[i] = 1

        code = torch.tensor(code, dtype=torch.int64)
        expanded_codes.append(code)

    return inputs, expanded_codes


class LangData:
    def __init__(self, fname, prediction_mask, source_mask, scale_factor=1, transpose=False):
        self.inputs, self.codes = read_file(fname, prediction_mask, source_mask)

        if transpose:
            self.codes, self.inputs = self.inputs, self.codes

        assert len(self.codes) == len(self.inputs)
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.inputs) * self.scale_factor

    def __getitem__(self, i):
        i = i % len(self.inputs)
        return self.inputs[i], self.codes[i]


if __name__ == '__main__':
    data = LangData('vocab8_language_1.txt', prediction_mask='xxxxxxx?')
    for b in torch.utils.data.DataLoader(data, shuffle=False, batch_size=2):
        print(b)