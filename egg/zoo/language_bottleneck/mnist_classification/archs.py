# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
import egg.core as core


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 400)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.leaky_relu(self.fc1(x))
        return x


class LinearEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 400)

    def forward(self, x):
        bsz = x.size(0)
        x = x.view(bsz, -1)
        x = self.fc(x)
        return x



class MlpEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = nn.Linear(28*28, 400)
        self.fc_2 = nn.Linear(400, 400)

    def forward(self, x):
        bsz = x.size(0)
        x = x.view(bsz, -1)

        x = self.fc_1(x)
        x = F.leaky_relu(x)
        x = self.fc_2(x)
        return x



class Sender(nn.Module):
    def __init__(self, vocab_size):
        super(Sender, self).__init__()

        self.vision = LeNet()
        self.fc = nn.Linear(400, vocab_size)

    def forward(self, x):
        x = self.vision(x)
        x = self.fc(x)
        logits = F.log_softmax(x, dim=1)
        return logits


class Receiver(nn.Module):
    def __init__(self, vocab_size, n_classes, img_emb_type='lenet'):
        super(Receiver, self).__init__()
        self.message_inp = core.RelaxedEmbedding(vocab_size, 400)

        if img_emb_type == 'lenet':
            self.image_inp = LeNet()
        elif img_emb_type == 'linear':
            self.image_inp = LinearEmb()
        elif img_emb_type == 'mlp':
            self.image_inp = MlpEmb()
        else:
            assert False

        self.fc = nn.Linear(800, n_classes)

    def forward(self, message, image):
        x_image = self.image_inp(image)
        x_message = self.message_inp(message)

        x = torch.cat([x_message, x_image], dim=1)
        x = self.fc(x)

        return torch.log_softmax(x, dim=1)

