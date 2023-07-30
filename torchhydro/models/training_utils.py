"""
Author: Wenyu Ouyang
Date: 2021-08-09 10:19:13
LastEditTime: 2023-07-27 19:32:31
LastEditors: Wenyu Ouyang
Description: Some util classes and functions during hydroDL training or testing
FilePath: \HydroTL\hydrotl\models\training_utils.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from typing import Union
import warnings
import torch


def get_the_device(device_num: Union[list, int]):
    """
    Get device for torch according to its name

    Parameters
    ----------
    device_num : Union[list, int]
        number of the device -- -1 means "cpu" or 0, 1, ... means "cuda:x"
    """
    if device_num in [[-1], -1, ["-1"]]:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return (
            torch.device(f"cuda:{str(device_num)}")
            if type(device_num) is not list
            else torch.device(f"cuda:{str(device_num[0])}")
        )
    if device_num not in [[-1], -1, ["-1"]]:
        warnings.warn("You don't have GPU, so have to choose cpu for models")
    return torch.device("cpu")


class EarlyStopper(object):
    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        cumulative_delta: bool = False,
    ):
        """
        EarlyStopping handler can be used to stop the training if no improvement after a given number of events.

        Parameters
        ----------
        patience
            Number of events to wait if no improvement and then stop the training.
        min_delta
            A minimum increase in the score to qualify as an improvement,
            i.e. an increase of less than or equal to `min_delta`, will count as no improvement.
        cumulative_delta
            It True, `min_delta` defines an increase since the last `patience` reset, otherwise,
        it defines an increase after the last event. Default value is False.
        """

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")

        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.counter = 0
        self.best_score = None

    def check_loss(self, model, validation_loss) -> bool:
        score = validation_loss
        if self.best_score is None:
            self.save_model_checkpoint(model)
            self.best_score = score

        elif score + self.min_delta >= self.best_score:
            if not self.cumulative_delta and score > self.best_score:
                self.best_score = score
            self.counter += 1
            print(self.counter)
            if self.counter >= self.patience:
                return False
        else:
            self.save_model_checkpoint(model)
            self.best_score = score
            self.counter = 0
        return True

    def save_model_checkpoint(self, model):
        torch.save(model.state_dict(), "checkpoint.pth")
