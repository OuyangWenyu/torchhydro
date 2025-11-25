"""
This script implements a conditional diffusion model for time-series data,
using an LSTM-based network to predict the noise. This is suitable for
hydrological modeling where temporal dependencies are key.
"""

import math

import torch
import torch.nn as nn
from tqdm.auto import tqdm


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LSTMDenoiseNet(nn.Module):
    def __init__(self, cfgs):
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
            bidirectional=True,  # Using a bidirectional LSTM can capture dependencies from both past and future
        )

        # Output layer
        self.final_layer = nn.Linear(
            hidden_size * 2, data_dim
        )  # *2 because of bidirectional

    def forward(self, x, time, context):
        # x: [batch, data_dim, seq_len]
        # time: [batch]
        # context: [batch, condition_dim, seq_len]

        # 1. Permute to (batch, seq_len, features) for LSTM
        x = x.permute(0, 2, 1)
        context = context.permute(0, 2, 1)
        seq_len = x.shape[1]

        # 2. Process time embedding and expand it to match sequence length
        t_emb = self.time_mlp(time)  # [batch, time_emb_dim]
        t_emb = t_emb.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # [batch, seq_len, time_emb_dim]

        # 3. Concatenate all inputs along the feature dimension
        lstm_input = torch.cat([x, context, t_emb], dim=-1)

        # 4. Pass through LSTM
        lstm_out, _ = self.lstm(lstm_input)  # [batch, seq_len, hidden_size * 2]

        # 5. Apply final linear layer to predict noise
        pred_noise = self.final_layer(lstm_out)  # [batch, seq_len, data_dim]

        # 6. Permute back to the original shape [batch, data_dim, seq_len]
        return pred_noise.permute(0, 2, 1)


class ConditionalDiffusionModel(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        model_cfgs = cfgs["model"]
        diffusion_cfgs = model_cfgs["diffusion"]

        self.data_dim = diffusion_cfgs["data_dim"]
        self.condition_dim = diffusion_cfgs["condition_dim"]

        self.denoise_net = LSTMDenoiseNet(cfgs)

        # --- Diffusion parameters ---
        timesteps = diffusion_cfgs["num_timesteps"]
        beta_schedule = diffusion_cfgs.get("beta_schedule", "cosine")

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        self.num_timesteps = timesteps

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_sample(self, x, t, context):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=batched_times, context=context
        )
        noise = torch.randn_like(x) if t > 0 else 0.0
        pred_x_start = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_x_start

    def p_mean_variance(self, x, t, context):
        pred_noise = self.denoise_net(x, t, context=context)
        x_start = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        return self.q_posterior(x_start=x_start, x_t=x, t=t)

    @torch.no_grad()
    def sample(self, context):
        batch_size, _, seq_len = context.shape
        shape = (batch_size, self.data_dim, seq_len)
        device = self.betas.device

        img = torch.randn(shape, device=device)

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop",
            total=self.num_timesteps,
        ):
            img = self.p_sample(img, t, context)

        return img

    def forward(self, x_0, y):
        batch_size = x_0.shape[0]

        t = torch.randint(
            0, self.num_timesteps, (batch_size,), device=x_0.device
        ).long()
        noise = torch.randn_like(x_0)

        x_t = self.q_sample(x_start=x_0, t=t, noise=noise)

        predicted_noise = self.denoise_net(x_t, t, context=y)

        loss = nn.functional.mse_loss(predicted_noise, noise)

        return {"loss": loss}
