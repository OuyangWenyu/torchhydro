import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv
# 所有_calculate方法都对类似conv_h的方法进行了升维，原代码参考：
# https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/gconv_gru.py

class GConvGRU(torch.nn.Module):
    r"""An implementation of the Chebyshev Graph Convolutional Gated Recurrent Unit
    Cell. For details see this paper: `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(GConvGRU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):

        self.conv_x_z = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_z = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_x_r = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_r = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_x_h = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_h = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            # X的形状是 [batch_size, num_nodes, features] 或 [num_nodes, features]
            if len(X.shape) == 3:  # 批处理情况
                batch_size, num_nodes = X.shape[0], X.shape[1]
                H = torch.zeros(batch_size, num_nodes, self.out_channels).to(X.device)
            else:  # 单个图情况
                num_nodes = X.shape[0]
                H = torch.zeros(num_nodes, self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H, lambda_max, num_nodes):
        Z = self.conv_x_z(X, edge_index, edge_weight, lambda_max=lambda_max)
        conv_h_z = self.conv_h_z(H, edge_index, edge_weight, lambda_max=lambda_max)
        conv_h_z = conv_h_z.unsqueeze(1) if len(H.shape)!=len(Z.shape) else conv_h_z
        Z = Z + conv_h_z
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H, lambda_max, num_nodes):
        R = self.conv_x_r(X, edge_index, edge_weight, lambda_max=lambda_max)
        conv_h_r = self.conv_h_r(H, edge_index, edge_weight, lambda_max=lambda_max)
        conv_h_r = conv_h_r.unsqueeze(1) if len(conv_h_r.shape)!=len(R.shape) else conv_h_r
        R = R + conv_h_r
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R, lambda_max, num_nodes):
        H_tilde = self.conv_x_h(X, edge_index, edge_weight, lambda_max=lambda_max)
        H = H.unsqueeze(1) if len(H.shape)!=len(H_tilde.shape) else H
        conv_h_h = self.conv_h_h(H * R, edge_index, edge_weight, lambda_max=lambda_max)
        conv_h_h = conv_h_h.unsqueeze(1) if len(H.shape)!=len(H_tilde.shape) else conv_h_h
        H_tilde = H_tilde + conv_h_h
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = H.unsqueeze(1) if len(H.shape)!=len(H_tilde.shape) else H
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
        num_nodes: int = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.


        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        # 如果没有提供num_nodes，从输入张量推断
        if num_nodes is None:
            num_nodes = X.size(0)
            
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H, lambda_max, num_nodes)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H, lambda_max, num_nodes)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R, lambda_max, num_nodes)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class GConvGRULayer(nn.Module):
    """优化的GConvGRU包装层 - 支持完全批处理

    直接利用GConvGRU的批处理能力，移除内层batch循环，实现完全并行化。
    输入形状为 [batch_size, seq_len, num_nodes, in_channels]。
    """
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.cell = GConvGRU(in_channels, out_channels, K)
        self.out_channels = out_channels

    def forward(self, X_seq, edge_index, edge_weight=None, H=None):
        # X_seq: [batch_size, seq_len, num_nodes, in_channels]
        batch_size, seq_len, num_nodes, in_channels = X_seq.shape
        
        # 初始化隐藏状态
        if H is None:
            H = torch.zeros(batch_size, num_nodes, self.out_channels, 
                          device=X_seq.device, dtype=X_seq.dtype)
        
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
    
    def __init__(self, edge_index, config, node_attributes=None):
        super(MultiFeatureStreamGCGRU, self).__init__()
        
        # 将edge_index注册为buffer，这样它会自动跟随模型移动到正确的设备
        self.register_buffer('edge_index', edge_index)
        self.config = config
        self.node_attributes = node_attributes
        
        # 输入特征维度：3个时间序列特征 + 静态属性维度
        self.time_series_dim = 3  # 降水、LSTM流量、实测流量
        self.static_attr_dim = config['node_attr_dim'] if config['use_node_attr'] and node_attributes is not None else 0
        self.input_feature_dim = self.time_series_dim + self.static_attr_dim
        
        # GConvGRU层（外部时间循环）：直接使用多通道输入特征，时间维度由外层循环处理
        self.gcgru_layer = GConvGRULayer(
            in_channels=self.input_feature_dim,
            out_channels=config['gcgru_out_channels'],
            K=config['gcgru_K']
        )
        
        # 输出MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(config['gcgru_out_channels'], config['mlp_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['mlp_dropout']),
            nn.Linear(config['mlp_hidden_dim'], config['forecast_length'])
        )
        
        print(f"多特征模型创建完成:")
        print(f"  - 时间序列特征维度: {self.time_series_dim}")
        print(f"  - 静态属性维度: {self.static_attr_dim}")
        print(f"  - 原始输入特征维度: {self.input_feature_dim}")
        print(f"  - GConvGRU输出维度: {config['gcgru_out_channels']}")
        print(f"  - 预测长度: {config['forecast_length']}")
        
    def forward(self, x):
        """前向传播 - 优化的批处理版本，移除外层batch循环"""
        # x: [batch_size, hindcast_length, num_nodes, features]
        batch_size, seq_len, num_nodes, features = x.shape

        # 直接使用批处理的GConvGRULayer，无需逐batch循环
        # 输入: [batch_size, seq_len, num_nodes, features]
        # 输出: outputs: [seq_len, batch_size, num_nodes, gcgru_out_channels]
        #       h_last: [batch_size, num_nodes, gcgru_out_channels]
        outputs, h_last = self.gcgru_layer(x, self.edge_index, edge_weight=None, H=None)

        # 步骤2: 输出预测（基于最后一个隐藏状态）
        predictions = self.output_mlp(h_last)  # [batch_size, num_nodes, forecast_length]

        # 转换维度顺序以匹配targets的格式：[batch_size, forecast_length, num_nodes]
        if self.config['forecast_length'] > 1:
            predictions = predictions.transpose(1, 2)  # [batch_size, forecast_length, num_nodes]
        else:
            # 单步预测时去掉最后一个维度
            predictions = predictions.squeeze(-1)  # [batch_size, num_nodes]

        return predictions
    
    def autoregressive_predict(self, x, forecast_steps, precipitation_future=None, lstm_future=None, targets_future=None, teacher_forcing_ratio=0.0):
        """
        自回归多步预测
        
        Args:
            x: 初始输入序列 [batch_size, hindcast_length, num_nodes, features]
            forecast_steps: 预测步数
            precipitation_future: 未来降水数据 [batch_size, forecast_steps, num_nodes] (可选)
            lstm_future: 未来LSTM预测数据 [batch_size, forecast_steps, num_nodes] (可选)
            
        Returns:
            predictions: [batch_size, forecast_steps, num_nodes] 多步预测结果
        """
        batch_size, seq_len, num_nodes, features = x.shape
        device = x.device
        
        # 存储所有预测结果
        all_predictions = []
        
        # 当前输入序列（滑动窗口）
        current_input = x.clone()  # [batch_size, hindcast_length, num_nodes, features]
        
        for step in range(forecast_steps):
            # 单步预测（保留梯度用于训练）
            logits = self.forward(current_input)  # [batch_size, forecast_length, num_nodes] 或 [batch_size, num_nodes]
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
                    future_precip = precipitation_future[:, step:step+1, :]
                else:
                    future_precip = current_input[:, -1:, :, 0]
                if lstm_future is not None:
                    future_lstm = lstm_future[:, step:step+1, :]
                else:
                    future_lstm = current_input[:, -1:, :, 1]
                
                # teacher forcing: 用真实流量替代下一步输入的预测流量（按比率）
                input_flow = step_pred_unsq  # 默认用预测
                if (targets_future is not None) and (teacher_forcing_ratio > 0.0):
                    gt_flow = targets_future[:, step:step+1, :]  # [batch_size, 1, num_nodes]
                    tf_mask = (torch.rand(batch_size, 1, 1, device=device) < teacher_forcing_ratio)
                    input_flow = torch.where(tf_mask, gt_flow, input_flow)
                
                # 构建新的时间步特征 [batch_size, 1, num_nodes, 3]
                new_time_step = torch.stack([
                    future_precip,              # 降水 [batch_size, 1, num_nodes]
                    future_lstm,                # LSTM流量 [batch_size, 1, num_nodes]
                    input_flow                  # 实测流量（预测或真值） [batch_size, 1, num_nodes]
                ], dim=3)
                
                # 拼接静态属性
                if self.static_attr_dim > 0:
                    static_attrs = current_input[:, -1:, :, 3:]  # [batch_size, 1, num_nodes, static_attr_dim]
                    new_time_step = torch.cat([new_time_step, static_attrs], dim=3)
                
                # 滑动窗口前移并追加新时间步
                current_input = torch.cat([current_input[:, 1:, :, :], new_time_step], dim=1)
        
        predictions = torch.cat(all_predictions, dim=1)
        return predictions
