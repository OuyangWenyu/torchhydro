from typing import Optional
import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import GConvGRU


class MultiFeatureStreamGCGRU(nn.Module):
    def __init__(
        self,
        gcgru_out_channels: int,
        gcgru_K: int,
        mlp_hidden_dim: int,
        mlp_dropout: float,
        forecast_length: int,
        teacher_forcing_ratio: float = 0.0,
        input_size: int = None,
        # TODO: "Add support for multiple target_feature_idx"
        target_feature_idx: int = 2,
    ) -> None:
        super(MultiFeatureStreamGCGRU, self).__init__()
        self.forecast_length = forecast_length
        self.gcgru_out_channels = gcgru_out_channels
        self.gcgru_K = gcgru_K
        self.target_feature_idx = target_feature_idx
        
        # Static initialization if input_size is provided
        if input_size is not None:
            self.gconv_gru = GConvGRU(input_size, self.gcgru_out_channels, self.gcgru_K)
        else:
            self.gconv_gru = None
            
        self.output_mlp = nn.Sequential(
            nn.Linear(gcgru_out_channels, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        lambda_max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                "Expected x with dims [total_nodes, hindcast_length, features]"
            )
        device = x.device
        total_nodes, seq_len, features = x.shape
        # ensure graph inputs are on same device
        edge_index = edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
        # ensure submodules are on same device
        if self.gconv_gru is None:
            self.gconv_gru = GConvGRU(features, self.gcgru_out_channels, self.gcgru_K)
        
        self.gconv_gru = self.gconv_gru.to(device)
        self.output_mlp = self.output_mlp.to(device)
        H = None
        
        predictions = []
        history_length = seq_len - self.forecast_length
        
        for t in range(seq_len):
            # Create a clone for the current time step to avoid in-place modification error
            # x is [total_nodes, seq_len, features] -> x[:, t, :] is [total_nodes, features]
            X_t = x[:, t, :].clone()

            # Autoregressive: Update input with previous prediction if in forecast period
            if t >= history_length:
                # We need input at t to include prediction from t-1 (which predicts value at t)
                # prediction from t-1 is stored in predictions[-1]
                if len(predictions) > 0:
                    last_pred = predictions[-1]
                    X_t[:, self.target_feature_idx] = last_pred.squeeze(-1)
        
            H = self.gconv_gru(
                X_t, edge_index, edge_weight=edge_weight, H=H, lambda_max=lambda_max
            )
        
            # Make prediction if we have processed enough history
            # X_{M-1} -> predicts Y_M (first forecast)
            if t >= history_length - 1:
                pred = self.output_mlp(H)
                if len(predictions) < self.forecast_length:
                    predictions.append(pred)
                    
        return torch.cat(predictions, dim=1)
