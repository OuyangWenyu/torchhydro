from io import BytesIO
import logging
import PIL.Image
import matplotlib.image
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from dMC.experiments.metrics import Metrics

log = logging.getLogger(__name__)


class Writer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.metrics = Metrics()
        self._writer = SummaryWriter(log_dir=str(self.cfg.tensorboard_dir))

    def close_writer(self) -> None:
        self._writer.flush()
        self._writer.close()

    def write_metrics(
        self,
        predictions: torch.Tensor,
        observations: torch.Tensor,
        loss: torch.Tensor,
        epoch: int,
    ) -> None:
        predictions_np = predictions.cpu().detach().numpy()
        observations_np = observations.cpu().detach().numpy()
        nse = self.metrics.nse(predictions_np, observations_np)
        self._writer.add_scalar(f"loss", loss.cpu().detach().numpy(), epoch)
        self._writer.add_scalar(f"nse", nse, epoch)
        self._writer.flush()

    def write_parameters(
        self,
        names: List[str],
        params: List[torch.Tensor],
        epoch: int,
        areas: torch.Tensor,
    ) -> None:
        for i in range((len(params))):
            param = params[i]
            name = names[i]
            if param.dim() == 0:
                self._write_single_parameter(epoch, name, param)
            else:
                self._write_parameter_distribution(epoch, name, param, areas)
        self._writer.flush()

    def _write_single_parameter(
        self, epoch: int, name: str, param: torch.nn.Parameter
    ) -> None:
        self._writer.add_scalar(
            f"{name}/data", param.data.cpu().detach().numpy(), epoch
        )

    def _write_parameter_distribution(
        self, epoch: int, name: str, param: torch.Tensor, areas: torch.Tensor
    ) -> None:
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(areas, param.cpu().detach().numpy(), label=f"{name}", color="blue")
        plt.xlabel("Drainage Area (km2)")
        plt.ylabel(f"{name} (-)")
        # if name == "n":
        # plt.ylim(0, 0.15)
        # else:
        #     plt.ylim(0, 2)
        plt.title(f"{name} vs Drainage Area for epoch {epoch}")
        plt.legend()
        plot_img = self._plot_to_image(fig)
        tensor_image = transforms.ToTensor()(plot_img)
        self._writer.add_image(f"{name}_plot", tensor_image, global_step=epoch)
        self._writer.flush()

    def _plot_to_image(self, figure: matplotlib.image.PcolorImage):
        """
        Function to convert matplotlib figure to PNG image
        :param figure:
        :return:
        """
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(figure)
        buf.seek(0)
        image = PIL.Image.open(buf)
        return np.array(image)

    def integrated_grad(self, prediction, parameter, name,  epoch):
        dldn = torch.autograd.grad(
            prediction,
            parameter,
            grad_outputs=torch.ones_like(prediction),
            retain_graph=True,
        )[0].mean()
        self._writer.add_scalar(
            f"{name}/data", dldn, epoch
        )
