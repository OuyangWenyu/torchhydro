"""
Author: Antigravity
Date: 2025-12-19
Description: Fourier Neural Operator (FNO) model implementation for torchhydro.
Adapted from neuraloperator/neuralop/models/fno.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeff ci.
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width, in_channels=1, out_channels=1, n_layers=4):
        super(FNO1d, self).__init__()
        """
        The 1D FNO network. 
        input: x (batch, in_channels, seq_len)
        output: out (batch, out_channels, seq_len)
        """
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.padding = 6  # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(self.in_channels, self.width)

        self.spectral_convs = nn.ModuleList(
            [
                SpectralConv1d(self.width, self.width, self.modes)
                for _ in range(n_layers)
            ]
        )
        self.ws = nn.ModuleList(
            [nn.Conv1d(self.width, self.width, 1) for _ in range(n_layers)]
        )

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_channels)

    def forward(self, x):
        # x shape: (batch, in_channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, in_channels)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)  # (batch, width, seq_len)

        # x = F.pad(x, (0, self.padding)) # pad the domain if input is non-periodic

        for i in range(self.n_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic

        x = x.permute(0, 2, 1)  # (batch, seq_len, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)  # (batch, out_channels, seq_len)

        return x


class HydrologyFNO(nn.Module):
    def __init__(
        self,
        n_input_features: int,
        n_output_features: int,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        **kwargs,
    ):
        """
        torchhydro compatible FNO wrapper.
        Expects input x to be [seq_len, batch, features]
        """
        super(HydrologyFNO, self).__init__()
        self.fno = FNO1d(
            modes=modes,
            width=width,
            in_channels=n_input_features,
            out_channels=n_output_features,
            n_layers=n_layers,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x shape: [seq_len, batch, features]
        # FNO1d expects: [batch, features, seq_len]
        x = x.permute(1, 2, 0)
        out = self.fno(x)
        # out shape: [batch, out_channels, seq_len]
        # torchhydro expects: [seq_len, batch, out_channels]
        out = out.permute(2, 0, 1)
        return out
