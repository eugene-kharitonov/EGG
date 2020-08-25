# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, transforms, utils
from torch import nn
from torch.nn import functional as F
import egg.core as core
import pathlib

from .data import get_dsprites_dataloader

class Sender(nn.Module):
    def __init__(self, message_dim, image_size=28*28):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(image_size, 400)
        self.fc21 = nn.Linear(400, message_dim)
        self.fc22 = nn.Linear(400, message_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu, logvar = self.fc21(x), self.fc22(x)

        return mu, logvar


class Receiver(nn.Module):
    def __init__(self, message_dim, image_size=28*28):
        super(Receiver, self).__init__()
        self.fc3 = nn.Linear(message_dim, 400)
        self.fc4 = nn.Linear(400, image_size)

    def forward(self, x):
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))


class VAE_Game(nn.Module):
    def __init__(self, sender, receiver, image_size):
        super().__init__()

        self.sender = sender
        self.receiver = receiver
        self.image_size = image_size

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, *batch):
        sender_input = batch[0]
        sender_input = sender_input.view(-1, self.image_size)
        mu, logvar = self.sender(sender_input)

        if self.train:
            message = self.reparameterize(mu, logvar)
        else:
            message = mu

        receiver_output = self.receiver(message)

        BCE = F.binary_cross_entropy(
            receiver_output, sender_input, reduction='sum')

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KLD

        log = core.Interaction(
            sender_input=sender_input,
            receiver_input=None,
            labels=None,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=torch.ones(message.size(0)),
            aux={}
        )

        return loss.mean(), log


class ImageDumpCallback(core.Callback):
    def __init__(self, eval_dataset, image_shape=(28, 28)):
        super().__init__()
        self.eval_dataset = eval_dataset
        self.image_shape = image_shape

    def on_epoch_end(self, loss, logs, epoch):
        dump_dir = pathlib.Path.cwd() / 'dump' / str(epoch)
        dump_dir.mkdir(exist_ok=True, parents=True)

        state = self.trainer.game.train
        self.trainer.game.eval()

        l = len(self.eval_dataset)

        for i in range(5):
            example_id = np.random.randint(0, l)
            example = self.eval_dataset[example_id]

            example = core.move_to(example, torch.device('cuda'))
            _, interaction = self.trainer.game(*example)

            image = example[0][0]

            output = interaction.receiver_output.view(*self.image_shape)
            image = image.view(*self.image_shape)
            utils.save_image(
                torch.cat([image, output], dim=1), dump_dir / (str(i) + '.png'))
        self.trainer.game.train(state)


def main(params):
    opts = core.init(params=params)
    kwargs = {'num_workers': 1, 'pin_memory': True} if opts.cuda else {}
    transform = transforms.ToTensor()

    data = 'dsprite'
    assert data in ['mnist', 'dsprite']

    if data == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transform),
            batch_size=opts.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transform),
            batch_size=opts.batch_size, shuffle=True, **kwargs)

        image_shape = (28, 28)
    elif data == 'dsprite':
        train_loader, test_loader = get_dsprites_dataloader(path_to_data='egg/zoo/mnist_vae/dsprites.npz',
                                                            batch_size=opts.batch_size)
        image_shape = (64, 64)

    image_size = np.prod(image_shape)

    sender = Sender(opts.vocab_size, image_size=image_size)
    receiver = Receiver(opts.vocab_size, image_size=image_size)
    game = VAE_Game(sender, receiver, image_size=image_size)
    optimizer = core.build_optimizer(game.parameters())

    # initialize and launch the trainer
    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader, validation_data=test_loader,
                           callbacks=[core.ConsoleLogger(as_json=True, print_train_loss=True), 
                           ImageDumpCallback(test_loader.dataset, image_shape=image_shape)])
    trainer.train(n_epochs=opts.n_epochs)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
