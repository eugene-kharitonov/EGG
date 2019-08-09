# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Any
import json

import egg.core as core
import torch
import numpy as np
import torch.nn.functional as F


class CallbackEvaluator(core.Callback):
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device
        self.epoch = 0

    def on_epoch_end(self, loss: float, logs: Dict[str, Any] = None):
        game = self.trainer.game
        game.eval()
        game.epochs = self.epoch + 1

        if self.epoch % 5 == 0:
            for batch in self.dataset:
                x, y = core.move_to(batch, self.device)
                masked_seq, logits, attention = game.masker(x)
                predicted = game.explainer(x)

                x = x[0, :].tolist()
                attention = attention[0, :].tolist()
                #print('att', attention)
                y = ''.join([str(i) for i in y[0, :].tolist()])
                predicted = ''.join([str(i) for i in (predicted[0, :] > 0.5).tolist()])

                combined = []

                for symbol, att in zip(x, attention):
                    if att == 1:
                        combined.append('x')
                    elif symbol == 0:
                        combined.append('0')
                        break
                    else:
                        combined.append(str(symbol))
                print(''.join(combined), '->', predicted, '/', y)

        print('prbos', F.sigmoid(game.masker.prob_mask_logits))
        game.train()
        self.epoch += 1