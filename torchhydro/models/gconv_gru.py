from typing import Optional
import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import GConvGRU, GConvLSTM


class MultiFeatureStreamGCGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        gcgru_out_channels: int,
        gcgru_K: int,
        mlp_hidden_dim: int,
        mlp_dropout: float,
        forecast_length: int,
        teacher_forcing_ratio: float = 0.0,
        # TODO: "Add support for multiple target_feature_idx"
        target_feature_idx: int = 2,
    ) -> None:
        super(MultiFeatureStreamGCGRU, self).__init__()
        self.forecast_length = forecast_length
        self.gcgru_K = gcgru_K
        self.target_feature_idx = target_feature_idx
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # Static initialization if input_size is provided
        self.gconv_gru = GConvGRU(input_size, gcgru_out_channels, self.gcgru_K)
            
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
        # # ensure submodules are on same device     
        # self.gconv_gru = self.gconv_gru.to(device)
        # self.output_mlp = self.output_mlp.to(device)
        
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
                is_teacher_forcing = True
                if not self.training:
                    is_teacher_forcing = False
                elif torch.rand(1).item() > self.teacher_forcing_ratio:
                    is_teacher_forcing = False
                if not is_teacher_forcing:
                    if len(predictions) > 0:
                        last_pred = predictions[-1]
                        X_t[:, self.target_feature_idx] = last_pred.squeeze(-1)
        
            H = self.gconv_gru(
                X_t, edge_index, edge_weight=edge_weight, H=H, lambda_max=lambda_max
            )
        
            # 4. 生成并保存预测 (仅在预报期)
            if t >= history_length:
                pred = self.output_mlp(H)
                predictions.append(pred)
                    
        return torch.cat(predictions, dim=1)




class MultiFeatureStreamGCLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        gclstm_out_channels: int,
        gclstm_K: int,
        mlp_hidden_dim: int,
        mlp_dropout: float,
        forecast_length: int,
        teacher_forcing_ratio: float = 0.0,
        # TODO: "Add support for multiple target_feature_idx"
        target_feature_idx: int = 2,
    ) -> None:
        super(MultiFeatureStreamGCLSTM, self).__init__()
        self.forecast_length = forecast_length
        self.gclstm_K = gclstm_K
        self.target_feature_idx = target_feature_idx
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # Static initialization if input_size is provided
        self.gconv_lstm = GConvLSTM(input_size, gclstm_out_channels, self.gclstm_K)
            
        self.output_mlp = nn.Sequential(
            nn.Linear(gclstm_out_channels, mlp_hidden_dim),
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
        # # ensure submodules are on same device     
        # self.gconv_lstm = self.gconv_lstm.to(device)
        # self.output_mlp = self.output_mlp.to(device)
        
        H = None
        C = None
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
                is_teacher_forcing = True
                if not self.training:
                    is_teacher_forcing = False
                elif torch.rand(1).item() > self.teacher_forcing_ratio:
                    is_teacher_forcing = False
                if not is_teacher_forcing:
                    if len(predictions) > 0:
                        last_pred = predictions[-1]
                        X_t[:, self.target_feature_idx] = last_pred.squeeze(-1)
        
            H, C = self.gconv_lstm(
                X_t, edge_index, edge_weight=edge_weight, H=H, C=C, lambda_max=lambda_max
            )
        
            # 4. 生成并保存预测 (仅在预报期)
            if t >= history_length:
                pred = self.output_mlp(H)
                predictions.append(pred)
                    
        return torch.cat(predictions, dim=1)
