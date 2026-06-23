"""
This script implements a conditional diffusion model for time-series data.
It leverages the `diffusers` library for the diffusion scheduling process and
uses a custom LSTM-based network to predict the noise.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm


class SinusoidalPosEmb(nn.Module):
    """Helper class for sinusoidal time embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
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
    def __init__(self, data_dim, condition_dim, time_emb_dim, hidden_size, num_layers, **kwargs):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        input_size = data_dim + condition_dim + time_emb_dim
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.final_layer = nn.Linear(hidden_size * 2, data_dim)

    def forward(self, sample, timestep, encoder_hidden_states):
        seq_len = sample.shape[1]
        t_emb = self.time_mlp(timestep)
        batch_size = sample.shape[0]
        if t_emb.shape[0] != batch_size:
            t_emb = t_emb.repeat(batch_size, 1)
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat([sample, encoder_hidden_states, t_emb], dim=-1)
        lstm_out, _ = self.lstm(lstm_input)
        pred_noise = self.final_layer(lstm_out)
        return pred_noise


class ConditionalDiffusionModel(nn.Module):
    # Flag to indicate this is a generative model that computes loss internally
    is_generative = True

    def __init__(self, diffusion: dict, denoise_net: dict, **kwargs):
        super().__init__()
        self.data_dim = diffusion["data_dim"]
        self.training_cfgs = kwargs.get("training_cfgs", {})
        self.denoise_net = LSTMDenoiseNet(
            data_dim=diffusion["data_dim"],
            condition_dim=diffusion["condition_dim"],
            time_emb_dim=diffusion["time_emb_dim"],
            **denoise_net
        )
        self.scheduler = DDPMScheduler(
            num_train_timesteps=diffusion["num_timesteps"],
            beta_schedule=diffusion.get("beta_schedule", "linear"),
            clip_sample=False,
        )

    def forward(self, x_condition, y_target):
        x_0 = y_target
        batch_size = x_0.shape[0]
        device = x_0.device
        noise = torch.randn_like(x_0)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noisy_data = self.scheduler.add_noise(x_0, noise, timesteps)
        predicted_noise = self.denoise_net(
            sample=noisy_data,
            timestep=timesteps,
            encoder_hidden_states=x_condition
        )
        loss = F.mse_loss(predicted_noise, noise)
        return {'loss': loss}

    @torch.no_grad()
    def sample(self, context):
        batch_size = context.shape[0]
        seq_len = context.shape[1]
        device = context.device
        
        sample = torch.randn(batch_size, seq_len, self.data_dim, device=device)
        
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)

        for t in tqdm(self.scheduler.timesteps, desc="Sampling loop"):
            denoiser_to_call = getattr(self.denoise_net, "module", self.denoise_net)
            noise_pred = denoiser_to_call.forward(
                sample=sample,
                timestep=t,
                encoder_hidden_states=context
            )
            sample = self.scheduler.step(noise_pred, t, sample).prev_sample
            
        return sample