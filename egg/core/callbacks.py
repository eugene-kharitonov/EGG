# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import pathlib
import time
from typing import Any, Dict, List, NamedTuple, Union

import torch

from egg.core.util import get_summary_writer

from .interaction import Interaction


class Callback:
    def on_train_begin(self, trainer_instance: "Trainer"):  # noqa: F821
        self.trainer = trainer_instance

    def on_train_end(self):
        pass

    def on_early_stopping(
        self,
        train_loss: float,
        train_logs: Interaction,
        epoch: int,
        test_loss: float = None,
        test_logs: Interaction = None,
    ):
        pass

    def on_test_begin(self, epoch: int):
        pass

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        pass

    def on_epoch_begin(self, epoch: int):
        pass

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        pass


class ConsoleLogger(Callback):
    def __init__(self, print_train_loss=False, as_json=False):
        self.print_train_loss = print_train_loss
        self.as_json = as_json

    def aggregate_print(self, loss: float, logs: Interaction, mode: str, epoch: int):
        dump = dict(loss=loss)
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        dump.update(aggregated_metrics)

        if self.as_json:
            dump.update(dict(mode=mode, epoch=epoch))
            output_message = json.dumps(dump)
        else:
            output_message = ", ".join(sorted([f"{k}={v}" for k, v in dump.items()]))
            output_message = f"{mode}: epoch {epoch}, loss {loss}, " + output_message
        print(output_message, flush=True)

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        self.aggregate_print(loss, logs, "test", epoch)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.print_train_loss:
            self.aggregate_print(loss, logs, "train", epoch)


class TensorboardLogger(Callback):
    def __init__(self, writer=None):
        if writer:
            self.writer = writer
        else:
            self.writer = get_summary_writer()

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        self.writer.add_scalar(tag="test/loss", scalar_value=loss, global_step=epoch)
        for k, v in logs.aux.items():
            self.writer.add_scalar(
                tag=f"test/{k}", scalar_value=v.mean(), global_step=epoch
            )

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        self.writer.add_scalar(tag="train/loss", scalar_value=loss, global_step=epoch)
        for k, v in logs.aux.items():
            self.writer.add_scalar(
                tag=f"train/{k}", scalar_value=v.mean(), global_step=epoch
            )

    def on_train_end(self):
        self.writer.close()


class TemperatureUpdater(Callback):
    def __init__(self, agent, decay=0.9, minimum=0.1, update_frequency=1):
        self.agent = agent
        assert hasattr(
            agent, "temperature"
        ), "Agent must have a `temperature` attribute"
        assert not isinstance(
            agent.temperature, torch.nn.Parameter
        ), "When using TemperatureUpdater, `temperature` cannot be trainable"
        self.decay = decay
        self.minimum = minimum
        self.update_frequency = update_frequency

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if epoch % self.update_frequency == 0:
            self.agent.temperature = max(
                self.minimum, self.agent.temperature * self.decay
            )


class Checkpoint(NamedTuple):
    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]


class CheckpointSaver(Callback):
    def __init__(
        self,
        checkpoint_path: Union[str, pathlib.Path],
        checkpoint_freq: int = 1,
        prefix: str = "",
    ):
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.checkpoint_freq = checkpoint_freq
        self.prefix = prefix
        self.epoch_counter = 0

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        self.epoch_counter = epoch
        if self.checkpoint_freq > 0 and (
            self.epoch_counter % self.checkpoint_freq == 0
        ):
            filename = (
                f"{self.prefix}_{self.epoch_counter}"
                if self.prefix
                else str(self.epoch_counter)
            )
            self.save_checkpoint(filename=filename)

    def on_train_end(self):
        self.save_checkpoint(
            filename=f"{self.prefix}_final" if self.prefix else "final"
        )

    def save_checkpoint(self, filename: str):
        """
        Saves the game, agents, and optimizer states to the checkpointing path under `<number_of_epochs>.tar` name
        """
        self.checkpoint_path.mkdir(exist_ok=True, parents=True)
        path = self.checkpoint_path / f"{filename}.tar"
        torch.save(self.get_checkpoint(), path)

    def get_checkpoint(self):
        return Checkpoint(
            epoch=self.epoch_counter,
            model_state_dict=self.trainer.game.state_dict(),
            optimizer_state_dict=self.trainer.optimizer.state_dict(),
        )


class InteractionSaver(Callback):
    def __init__(
        self,
        train_epochs: List = None,
        test_epochs: List = None,
        folder_path: str = "./interactions",
    ):
        if isinstance(train_epochs, list):
            assert all(map(lambda x: x > 0, train_epochs))
            self.train_epochs = train_epochs
        else:
            self.train_epochs = []
        if isinstance(test_epochs, list):
            assert all(map(lambda x: x > 0, test_epochs))
            self.test_epochs = test_epochs
        else:
            self.test_epochs = []

        self.folder_path = (
            pathlib.Path(folder_path) / time.strftime("%Y_%m_%d_%H_%M_%S")
        ).expanduser()

    @staticmethod
    def dump_interactions(
        logs: Interaction, mode: str, epoch: int, dump_dir: str = "./interactions"
    ):
        dump_dir = pathlib.Path(dump_dir) / mode
        dump_dir.mkdir(exist_ok=True, parents=True)
        torch.save(logs, dump_dir / f"interactions_epoch{epoch}")

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        if epoch in self.test_epochs:
            self.dump_interactions(logs, "validation", epoch, self.folder_path)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if epoch in self.train_epochs:
            self.dump_interactions(logs, "train", epoch, self.folder_path)
