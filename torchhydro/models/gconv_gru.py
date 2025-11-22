from typing import Optional
import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import GConvGRU

# 所有_calculate方法都对类似conv_h的方法进行了升维，原代码参考：
# https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/gconv_gru.py


class GConvGRULayer(nn.Module):
    """优化的GConvGRU包装层 - 支持完全批处理

    直接利用GConvGRU的批处理能力，移除内层batch循环，实现完全并行化。
    输入形状为 [batch_size, seq_len, num_nodes, in_channels]。
    """

    def __init__(self, in_channels: int, out_channels: int, K: int) -> None:
        super().__init__()
        self.cell = GConvGRU(in_channels, out_channels, K)
        self.out_channels = out_channels

    def forward(
        self,
        X_seq: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        H: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # X_seq: [batch_size, seq_len, num_nodes, in_channels]
        batch_size, seq_len, num_nodes, in_channels = X_seq.shape

        # 初始化隐藏状态
        if H is None:
            H = torch.zeros(
                batch_size,
                num_nodes,
                self.out_channels,
                device=X_seq.device,
                dtype=X_seq.dtype,
            )

        outputs = []
        for t in range(seq_len):
            # 取出当前时间步的所有batch数据: [batch_size, num_nodes, in_channels]
            X_t = X_seq[:, t, :, :]
            # GConvGRU支持批处理，直接传入整个batch
            H = self.cell(X_t, edge_index, edge_weight, H)
            outputs.append(H)

        # 返回所有时间步的输出和最后的隐藏状态
        # outputs: [seq_len, batch_size, num_nodes, out_channels]
        return torch.stack(outputs, dim=0), H


class MultiFeatureStreamGCGRU(nn.Module):
    """多特征Stream GC-GRU模型 - 使用降水、LSTM流量、实测流量和静态属性"""

    def __init__(
        self,
        gcgru_out_channels: int,
        gcgru_K: int,
        mlp_hidden_dim: int,
        mlp_dropout: float,
        forecast_length: int,
        teacher_forcing_ratio: float = 0.0,
    ) -> None:
        super(MultiFeatureStreamGCGRU, self).__init__()

        # 将edge_index注册为buffer，这样它会自动跟随模型移动到正确的设备
        self.forecast_length = forecast_length
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # 输入特征维度：3个时间序列特征 + 静态属性维度
        self.time_series_dim = 3  # 降水、LSTM流量、实测流量
        # TODO: need to be deleted
        self.static_attr_dim = 0

        self.input_feature_dim = self.time_series_dim + self.static_attr_dim

        # GConvGRU层（外部时间循环）：直接使用多通道输入特征，时间维度由外层循环处理
        self.gcgru_layer = GConvGRULayer(
            in_channels=self.input_feature_dim,
            out_channels=gcgru_out_channels,
            K=gcgru_K,
        )

        # 输出MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(gcgru_out_channels, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, forecast_length),
        )

    def _forward_single_step(self, x: torch.Tensor) -> torch.Tensor:
        """单步前向传播 - 优化的批处理版本，移除外层batch循环

        Args:
            x: 输入序列 [batch_size, hindcast_length, num_nodes, features]

        Returns:
            predictions: [batch_size, forecast_length, num_nodes] 或 [batch_size, num_nodes]
        """
        # x: [batch_size, hindcast_length, num_nodes, features]
        batch_size, seq_len, num_nodes, features = x.shape

        # 直接使用批处理的GConvGRULayer，无需逐batch循环
        # 输入: [batch_size, seq_len, num_nodes, features]
        # 输出: outputs: [seq_len, batch_size, num_nodes, gcgru_out_channels]
        #       h_last: [batch_size, num_nodes, gcgru_out_channels]
        outputs, h_last = self.gcgru_layer(x, self.edge_index, edge_weight=None, H=None)

        # 步骤2: 输出预测（基于最后一个隐藏状态）
        predictions = self.output_mlp(
            h_last
        )  # [batch_size, num_nodes, forecast_length]

        # 转换维度顺序以匹配targets的格式：[batch_size, forecast_length, num_nodes]
        if self.forecast_length > 1:
            predictions = predictions.transpose(
                1, 2
            )  # [batch_size, forecast_length, num_nodes]
        else:
            # 单步预测时去掉最后一个维度
            predictions = predictions.squeeze(-1)  # [batch_size, num_nodes]

        return predictions

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        自回归多步预测（主要forward接口）

        如果不提供 forecast_steps，则执行单步预测；否则执行自回归多步预测。

        Args:
            x: 初始输入序列 [batch_size, hindcast_length, num_nodes, features]
            x include: 
            - precipitation: 降水数据 [batch_size, hindcast_length, num_nodes]
            - lstm: LSTM预测数据 [batch_size, hindcast_length, num_nodes]
            - targets: 真实流量数据 [batch_size, hindcast_length, num_nodes]

        Returns:
            predictions: [batch_size, forecast_length, num_nodes]
        """
        # 如果没有指定forecast_steps，执行单步预测
        if self.forecast_length == 1:
            return self._forward_single_step(x)

        # 否则执行自回归多步预测
        batch_size, seq_len, num_nodes, features = x.shape
        device = x.device

        # 存储所有预测结果
        all_predictions = []

        # 当前输入序列（滑动窗口）
        current_input = x.clone()  # [batch_size, hindcast_length, num_nodes, features]

        for step in range(self.forecast_length):
            # 单步预测（保留梯度用于训练）
            logits = self._forward_single_step(
                current_input
            )  # [batch_size, forecast_length, num_nodes] 或 [batch_size, num_nodes]
            # 取第1个预测步作为单步预测
            if logits.dim() == 3:
                step_prediction = logits[:, 0, :]  # [batch_size, num_nodes]
            else:
                step_prediction = logits  # 已是单步

            # 保存当前步的预测用于损失计算
            step_pred_unsq = step_prediction.unsqueeze(1)  # [batch_size, 1, num_nodes]
            all_predictions.append(step_pred_unsq)

            # 准备下一步的输入（除最后一步外）
            if step < forecast_steps - 1:
                # 未来外生量（若未提供则回退为最后时间步）
                if precipitation_future is not None:
                    future_precip = precipitation_future[:, step : step + 1, :]
                else:
                    future_precip = current_input[:, -1:, :, 0]
                if lstm_future is not None:
                    future_lstm = lstm_future[:, step : step + 1, :]
                else:
                    future_lstm = current_input[:, -1:, :, 1]

                # teacher forcing: 用真实流量替代下一步输入的预测流量（按比率）
                input_flow = step_pred_unsq  # 默认用预测
                if (targets_future is not None) and (teacher_forcing_ratio > 0.0):
                    gt_flow = targets_future[
                        :, step : step + 1, :
                    ]  # [batch_size, 1, num_nodes]
                    tf_mask = (
                        torch.rand(batch_size, 1, 1, device=device)
                        < self.teacher_forcing_ratio
                    )
                    input_flow = torch.where(tf_mask, gt_flow, input_flow)

                # 构建新的时间步特征 [batch_size, 1, num_nodes, 3]
                new_time_step = torch.stack(
                    [
                        future_precip,  # 降水 [batch_size, 1, num_nodes]
                        future_lstm,  # LSTM流量 [batch_size, 1, num_nodes]
                        input_flow,  # 实测流量（预测或真值） [batch_size, 1, num_nodes]
                    ],
                    dim=3,
                )

                # 拼接静态属性
                if self.static_attr_dim > 0:
                    static_attrs = current_input[
                        :, -1:, :, 3:
                    ]  # [batch_size, 1, num_nodes, static_attr_dim]
                    new_time_step = torch.cat([new_time_step, static_attrs], dim=3)

                # 滑动窗口前移并追加新时间步
                current_input = torch.cat(
                    [current_input[:, 1:, :, :], new_time_step], dim=1
                )

        predictions = torch.cat(all_predictions, dim=1)
        return predictions
