"""
This script implements a conditional diffusion model for time-series data.
It leverages the `diffusers` library for the diffusion scheduling process and
uses a custom LSTM-based network to predict the noise, which is suitable for
hydrological modeling where temporal dependencies are key.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast, Dict, Any
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm


class SinusoidalPosEmb(nn.Module):
    """Helper class for sinusoidal time embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LSTMDenoiseNet(nn.Module):
    """A denoising network for time-series data based on LSTM."""

    def __init__(self, cfgs: dict) -> None:
        super().__init__()
        model_cfgs = cfgs["model"]
        diffusion_cfgs = model_cfgs["diffusion"]
        denoise_cfgs = model_cfgs["denoise_net"]

        data_dim = diffusion_cfgs["data_dim"]
        condition_dim = diffusion_cfgs["condition_dim"]
        time_emb_dim = diffusion_cfgs["time_emb_dim"]
        hidden_size = denoise_cfgs["hidden_size"]
        num_layers = denoise_cfgs["num_layers"]

        # Time embedding network
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Core LSTM network
        input_size = data_dim + condition_dim + time_emb_dim
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Output layer
        self.final_layer = nn.Linear(
            hidden_size * 2, data_dim
        )  # *2 because of bidirectional

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the denoising network.

        Args:
            sample (torch.Tensor): The noisy input data (x_t). Shape:
                [batch, data_dim, seq_len].
            timestep (torch.Tensor): The current timestep tensor. Shape:
                [batch].
            encoder_hidden_states (torch.Tensor): Conditioning data (y). Shape:
                [batch, condition_dim, seq_len].
        """
        # 1. Permute inputs to (batch, seq_len, features) for LSTM
        sample = sample.permute(0, 2, 1)
        encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1)
        seq_len = sample.shape[1]

        # 2. Process time embedding and match batch size
        t_emb = self.time_mlp(timestep)

        # Repeat t_emb per batch item if needed
        batch_size = sample.shape[0]
        if t_emb.shape[0] != batch_size:
            t_emb = t_emb.repeat(batch_size, 1)

        t_emb = t_emb.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # [batch_size, seq_len, time_emb_dim]

        # 3. Concatenate all inputs along the feature dimension
        lstm_input = torch.cat([sample, encoder_hidden_states, t_emb], dim=-1)

        # 4. Pass through LSTM
        lstm_out, _ = self.lstm(lstm_input)  # [batch, seq_len, hidden_size*2]
        # 5. Apply final linear layer to predict noise
        pred_noise = self.final_layer(lstm_out)  # [batch, seq_len, data_dim]

        # 6. Permute back to the original shape [batch, data_dim, seq_len]
        return cast(torch.Tensor, pred_noise.permute(0, 2, 1))


class ConditionalDiffusionModel(nn.Module):
    """
    A conditional diffusion model for time-series, refactored to use the diffusers library.
    """

    def __init__(self, cfgs: Dict[str, Any]) -> None:
        super().__init__()
        self.cfgs = cfgs
        self.denoise_net = LSTMDenoiseNet(cfgs)

        # Initialize the scheduler from diffusers
        self.scheduler = DDPMScheduler(
            num_train_timesteps=cfgs["model"]["diffusion"]["num_timesteps"],
            beta_schedule=cfgs["model"]["diffusion"].get("beta_schedule", "cosine"),
            clip_sample=False,  # Not predicting images; no need to clip to [0, 1]
        )

    def forward(self, x_0: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            x_0 (torch.Tensor): The clean target data (e.g., runoff) with shape
                [batch, data_dim, seq_len].
            y (torch.Tensor): The conditioning data (e.g., rainfall) with shape
                [batch, condition_dim, seq_len].
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # 1. Sample noise
        noise = torch.randn_like(x_0)

        # 2. Sample a random timestep for each sequence
        timesteps = torch.randint(
            low=0,
            high=self.scheduler.config.num_train_timesteps,
            size=(batch_size,),
            device=device,
        ).long()

        # 3. Add noise to the clean data to get the noisy data
        #    for the given timesteps
        noisy_data = self.scheduler.add_noise(x_0, noise, timesteps)

        # 4. Predict the noise residual
        predicted_noise = self.denoise_net(
            sample=noisy_data, timestep=timesteps, encoder_hidden_states=y
        )

        # 5. Calculate the loss
        loss = F.mse_loss(predicted_noise, noise)

        return {"loss": loss}

    @torch.no_grad()
    def sample(self, context: torch.Tensor) -> torch.Tensor:
        """
        Inference method to generate samples from noise.

        Args:
            context (torch.Tensor): Conditioning data, e.g. rainfall, with shape
                [batch, condition_dim, seq_len].
        """
        batch_size, _, seq_len = context.shape
        device = context.device

        # Start with random noise.
        # Shape must match the data dimension (not the condition dimension).
        data_dim = self.cfgs["model"]["diffusion"]["data_dim"]
        sample = torch.randn(batch_size, data_dim, seq_len, device=device)

        # Set the number of inference steps.
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)

        # Loop through the timesteps to denoise the data.
        for t in tqdm(self.scheduler.timesteps, desc="Sampling loop"):
            # Predict the noise residual.
            noise_pred = self.denoise_net(
                sample=sample, timestep=t, encoder_hidden_states=context
            )
            # Compute the previous noisy sample x_t -> x_{t-1}.
            sample = self.scheduler.step(noise_pred, t, sample).prev_sample

        return sample
