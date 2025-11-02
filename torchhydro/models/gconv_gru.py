import torch
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


class AutoregressiveStreamGCGRU(nn.Module):
    def __init__(self, edge_index, config, upstream_mapping=None, node_areas=None):
        super().__init__()
        self.hindcast_length = config['hindcast_length']
        self.forecast_length = config['forecast_length']
        self.node_attr_dim = config.get('node_attr_dim', 0)
        self.gcgru_out_channels = config['gcgru_out_channels']
        self.upstream_mapping = upstream_mapping or {}
        self.register_buffer('edge_index', edge_index)
        
        # 存储节点面积信息用于面积加权聚合
        if node_areas is not None:
            self.register_buffer('node_areas', torch.tensor(node_areas, dtype=torch.float32))
        else:
            self.node_areas = None
        
        # 历史数据处理的GConvGRU层
        self.gcgru_past_rain = GConvGRU(
            in_channels=config['hindcast_length'], 
            out_channels=config['gcgru_out_channels'], 
            K=config['gcgru_K']
        )
        self.gcgru_past_lstm_flow = GConvGRU(
            in_channels=config['hindcast_length'], 
            out_channels=config['gcgru_out_channels'], 
            K=config['gcgru_K']
        )
        self.gcgru_past_upstream_flow = GConvGRU(
            in_channels=config['hindcast_length'], 
            out_channels=config['gcgru_out_channels'], 
            K=config['gcgru_K']
        )
        
        # 当前时刻数据处理的GConvGRU层（用于自回归）
        self.gcgru_current_rain = GConvGRU(
            in_channels=1,  # 单天数据
            out_channels=config['gcgru_out_channels'], 
            K=config['gcgru_K']
        )
        self.gcgru_current_lstm_flow = GConvGRU(
            in_channels=1,  # 单天数据
            out_channels=config['gcgru_out_channels'], 
            K=config['gcgru_K']
        )
        
        # 节点属性处理层
        if self.node_attr_dim > 0:
            self.node_attr_mlp = nn.Sequential(
                nn.Linear(self.node_attr_dim, config['node_attr_hidden_dim']),
                nn.ReLU(),
                nn.Linear(config['node_attr_hidden_dim'], config['gcgru_out_channels'])
            )
            fusion_input_dim = config['gcgru_out_channels'] * 6  # 包含节点属性
        else:
            self.node_attr_mlp = None
            fusion_input_dim = config['gcgru_out_channels'] * 5  # 不包含节点属性
        
        # 添加节点类型感知的特征权重
        self.feature_weights = nn.ModuleDict({
            'source_nodes': nn.Linear(fusion_input_dim, fusion_input_dim),
            'confluence_nodes': nn.Linear(fusion_input_dim, fusion_input_dim)
        })
        
        # 为源头节点和汇流节点设置不同的权重初始化
        self._init_feature_weights()
        
        # 融合层
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, config['mlp_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['mlp_dropout']),
            nn.Linear(config['mlp_hidden_dim'], config['gcgru_out_channels'])
        )
        
        # 输出层（单天预测）
        self.output_linear = nn.Linear(config['gcgru_out_channels'], 1)
        
    def forward(self, x, autoregressive=True):
        """
        前向传播
        
        Args:
            x: 输入数据元组
                x[0]: past_rain [batch, M, num_nodes]
                x[1]: past_lstm_flow [batch, M, num_nodes]
                x[2]: past_upstream_flow [batch, M, num_nodes]
                x[3]: future_rain [batch, n, num_nodes]
                x[4]: future_lstm_flow [batch, n, num_nodes]
                x[5]: node_attributes [num_nodes, node_attr_dim] (可选)
            autoregressive: 是否使用自回归模式
            
        Returns:
            output: 预测结果 [batch, n, num_nodes]
        """
        batch_size = x[0].shape[0]
        num_nodes = x[0].shape[2]
        
        if autoregressive:
            return self._autoregressive_forward(x, batch_size, num_nodes)
        else:
            return self._single_step_forward(x, batch_size, num_nodes, day_idx=0)
    
    def _init_feature_weights(self):
        """初始化特征权重，体现不同节点类型的特征重要性"""
        # 获取特征权重矩阵的维度
        feature_dim = self.feature_weights['source_nodes'].weight.shape[0]
        gcgru_channels = self.gcgru_out_channels
        
        # 判断是否包含节点属性
        has_node_attr = (feature_dim == gcgru_channels * 6)  # 5个动态特征 + 1个节点属性特征
        
        if has_node_attr:
            # 包含节点属性的情况：6个特征组
            source_weights = torch.zeros(6)  # [rain_past, lstm_past, upstream_past, rain_current, lstm_current, node_attr]
            source_weights[0] = 1.0  # 历史降雨权重大
            source_weights[1] = 1.5  # LSTM流量权重中等
            source_weights[2] = 0.0  # 上游流量权重为0
            source_weights[3] = 1.0  # 当前降雨权重最大
            source_weights[4] = 3.0  # 当前LSTM流量权重小
            source_weights[5] = 1.0  # 节点属性权重中等偏大
            
            confluence_weights = torch.zeros(6)
            confluence_weights[0] = 1.0  # 历史降雨权重小
            confluence_weights[1] = 1.0  # LSTM流量权重大
            confluence_weights[2] = 4.0  # 上游流量权重最大
            confluence_weights[3] = 0.5  # 当前降雨权重小
            confluence_weights[4] = 1.5  # 当前LSTM流量权重大
            confluence_weights[5] = 1.0  # 节点属性权重中等
        else:
            # 不包含节点属性的情况：5个特征组
            source_weights = torch.zeros(5)  # [rain_past, lstm_past, upstream_past, rain_current, lstm_current]
            source_weights[0] = 1.0  # 历史降雨权重大
            source_weights[1] = 1.5  # LSTM流量权重中等
            source_weights[2] = 0.0  # 上游流量权重为0
            source_weights[3] = 1.0  # 当前降雨权重最大
            source_weights[4] = 3.0  # 当前LSTM流量权重小
            
            confluence_weights = torch.zeros(5)
            confluence_weights[0] = 1.0  # 历史降雨权重小
            confluence_weights[1] = 1.0  # LSTM流量权重大
            confluence_weights[2] = 4.0  # 上游流量权重最大
            confluence_weights[3] = 0.5  # 当前降雨权重小
            confluence_weights[4] = 1.5  # 当前LSTM流量权重大
        
        # 将权重扩展到所有通道
        source_weights_expanded = source_weights.repeat_interleave(gcgru_channels)
        confluence_weights_expanded = confluence_weights.repeat_interleave(gcgru_channels)
        
        # 初始化权重矩阵为对角矩阵乘以权重
        with torch.no_grad():
            self.feature_weights['source_nodes'].weight.copy_(
                torch.diag(source_weights_expanded)
            )
            self.feature_weights['confluence_nodes'].weight.copy_(
                torch.diag(confluence_weights_expanded)
            )
    
    def _apply_node_type_weighting(self, combined_features, batch_size, num_nodes):
        """根据节点类型应用不同的特征权重"""
        weighted_features = torch.zeros_like(combined_features)
        
        for node_idx in range(num_nodes):
            # 判断节点类型
            has_upstream = (node_idx in self.upstream_mapping and 
                          len(self.upstream_mapping[node_idx]) > 0)
            
            if has_upstream:
                # 汇流节点：使用汇流节点权重
                weighted_features[:, node_idx, :] = self.feature_weights['confluence_nodes'](
                    combined_features[:, node_idx, :]
                )
            else:
                # 源头节点：使用源头节点权重
                weighted_features[:, node_idx, :] = self.feature_weights['source_nodes'](
                    combined_features[:, node_idx, :]
                )
        
        return weighted_features



    def _single_step_forward(self, x, batch_size, num_nodes, day_idx=0):
        """单步前向传播（添加节点类型感知的特征加权）"""
        # 处理历史数据 - 输入形状: [batch, time_steps, num_nodes]
        # GConvGRU期望输入形状: [batch, num_nodes, time_steps]
        
        # 处理历史数据，使用与原始模型相同的permute操作
        past_rain_features = self.gcgru_past_rain(
            x[0].permute(0, 2, 1), self.edge_index, num_nodes=num_nodes
        )  # 输出: [batch, num_nodes, out_channels]
        
        past_lstm_flow_features = self.gcgru_past_lstm_flow(
            x[1].permute(0, 2, 1), self.edge_index, num_nodes=num_nodes
        )
        
        past_upstream_flow_features = self.gcgru_past_upstream_flow(
            x[2].permute(0, 2, 1), self.edge_index, num_nodes=num_nodes
        )
        
        # 处理当前时刻数据 - 需要添加时间维度
        # x[3][:, day_idx, :] 形状: [batch, num_nodes]
        # 需要转换为 [batch, num_nodes, 1] 以匹配GConvGRU期望的输入
        current_rain = x[3][:, day_idx:day_idx+1, :].permute(0, 2, 1)  # [batch, num_nodes, 1]
        current_lstm_flow = x[4][:, day_idx:day_idx+1, :].permute(0, 2, 1)  # [batch, num_nodes, 1]
        
        current_rain_features = self.gcgru_current_rain(
            current_rain, self.edge_index, num_nodes=num_nodes
        )  # 输出: [batch, num_nodes, out_channels]
        
        current_lstm_flow_features = self.gcgru_current_lstm_flow(
            current_lstm_flow, self.edge_index, num_nodes=num_nodes
        )
        
        # 特征融合
        if len(x) > 5 and self.node_attr_mlp is not None:
            # 处理节点属性
            node_attrs = x[5]
            node_attr_features = self.node_attr_mlp(node_attrs)
            # 如果节点属性没有batch维度，添加batch维度
            if node_attr_features.dim() == 2:
                node_attr_features = node_attr_features.unsqueeze(0).expand(batch_size, -1, -1)
            
            combined_features = torch.cat([
                past_rain_features, past_lstm_flow_features, past_upstream_flow_features,
                current_rain_features, current_lstm_flow_features, node_attr_features
            ], dim=-1)
        else:
            combined_features = torch.cat([
                past_rain_features, past_lstm_flow_features, past_upstream_flow_features,
                current_rain_features, current_lstm_flow_features
            ], dim=-1)
        
        # 应用节点类型感知的特征加权
        weighted_features = self._apply_node_type_weighting(combined_features, batch_size, num_nodes)
        
        # 通过融合层和输出层
        fused = self.fusion_mlp(weighted_features)
        output = self.output_linear(fused)  # [batch, num_nodes, 1]
        
        return output.permute(0,2,1)  # [batch, 1, num_nodes]
    
    def _autoregressive_forward(self, x, batch_size, num_nodes):
        """自回归前向传播"""
        outputs = []
        
        # 初始化历史数据
        current_past_rain = x[0].clone()  # [batch, M, num_nodes]
        current_past_lstm_flow = x[1].clone()  # [batch, M, num_nodes]
        current_past_upstream_flow = x[2].clone()  # [batch, M, num_nodes]
        
        for day in range(self.forecast_length):
            # 构建当前输入
            current_input = (
                current_past_rain,
                current_past_lstm_flow,
                current_past_upstream_flow,
                x[3],  # future_rain
                x[4],  # future_lstm_flow
            )
            
            # 添加节点属性（如果存在）
            if len(x) > 5:
                current_input = current_input + (x[5],)
            
            # 单步预测
            day_output = self._single_step_forward(
                current_input, batch_size, num_nodes, day_idx=day
            )  # [batch, 1, num_nodes]
            
            outputs.append(day_output)
            
            # 更新历史数据（滑动窗口）
            if day < self.forecast_length - 1:
                # 更新历史降水和LSTM流量
                new_rain = x[3][:, day:day+1, :]  # [batch, 1, num_nodes]
                new_lstm_flow = x[4][:, day:day+1, :]  # [batch, 1, num_nodes]
                
                current_past_rain = torch.cat([
                    current_past_rain[:, 1:, :], new_rain
                ], dim=1)
                current_past_lstm_flow = torch.cat([
                    current_past_lstm_flow[:, 1:, :], new_lstm_flow
                ], dim=1)
                
                # 更新历史上游流量（使用预测结果）
                # 这里需要根据拓扑关系更新上游流量
                new_upstream_flow = self._update_upstream_flow(
                    day_output, current_past_upstream_flow
                )  # [batch, 1, num_nodes]
                
                current_past_upstream_flow = torch.cat([
                    current_past_upstream_flow[:, 1:, :], new_upstream_flow
                ], dim=1)
        
        # 拼接所有输出
        output = torch.cat(outputs, dim=1)  # [batch, n, num_nodes]
        return output
    
    def _update_upstream_flow(self, predicted_flow, past_upstream_flow):
        """
        根据预测结果更新上游流量，使用面积加权聚合
        
        Args:
            predicted_flow: 当前预测结果 [batch, 1, num_nodes]
            past_upstream_flow: 历史上游流量 [batch, M, num_nodes]
            
        Returns:
            new_upstream_flow: 更新后的上游流量 [batch, 1, num_nodes]
        """
        batch_size, _, num_nodes = predicted_flow.shape
        new_upstream_flow = torch.zeros_like(predicted_flow)
        
        # 使用与训练阶段一致的拓扑关系计算上游流量
        for node_idx in range(num_nodes):
            # 获取当前节点的上游节点索引
            upstream_indices = self.upstream_mapping.get(node_idx, [])
            
            if upstream_indices:
                # 获取上游节点的预测流量
                upstream_flows = predicted_flow[:, :, upstream_indices]  # [batch, 1, num_upstream]
                
                if self.node_areas is not None:
                    # 使用面积加权聚合
                    upstream_areas = self.node_areas[upstream_indices]  # [num_upstream]
                    # 计算面积权重（归一化）
                    area_weights = upstream_areas / upstream_areas.sum()  # [num_upstream]
                    # 扩展权重维度以匹配流量数据
                    area_weights = area_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, num_upstream]
                    # 面积加权聚合
                    new_upstream_flow[:, :, node_idx] = (upstream_flows * area_weights).sum(dim=2)
                else:
                    # 如果没有面积信息，回退到简单均值
                    new_upstream_flow[:, :, node_idx] = upstream_flows.mean(dim=2)  # [batch, 1]
            else:
                # 如果没有上游节点，使用0
                new_upstream_flow[:, :, node_idx] = 0.0
        
        return new_upstream_flow
