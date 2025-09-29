"""
Standard Spatio-Temporal Graph Convolutional Network (STGCN) Implementation
Reference: Spatio-Temporal Graph Convolutional Networks (Yu et al., 2018)
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv
from typing import Optional, Union
import math


class SimpleBatchIndexManager:
    """简化的批次索引管理器 - 解决复杂的batch模式索引问题"""
    
    @staticmethod
    def validate_and_fix_edge_index(edge_index, num_nodes, device=None):
        """验证并修复边索引，全部在CPU上做检查和过滤，最后再to(device)，防止CUDA断言"""
        if device is None:
            device = edge_index.device
        edge_index = edge_index.to(dtype=torch.long)
        ei_cpu = edge_index.cpu()
        if ei_cpu.numel() == 0:
            print("[INFO] edge_index is empty!")
            indices = torch.arange(num_nodes, device=device, dtype=torch.long)
            return torch.stack([indices, indices], dim=0)
        
        if (ei_cpu < 0).any() or (ei_cpu >= num_nodes).any():
            print("[WARNING] edge_index contains invalid indices!")
        valid_mask = (ei_cpu[0] >= 0) & (ei_cpu[0] < num_nodes) & (ei_cpu[1] >= 0) & (ei_cpu[1] < num_nodes)
        valid_edges = ei_cpu[:, valid_mask]
        if valid_edges.shape[1] == 0:
            print("[INFO] All edges are invalid, fallback to self-loops.")
            indices = torch.arange(num_nodes, device=device, dtype=torch.long)
            return torch.stack([indices, indices], dim=0)
        return valid_edges.to(device=device, dtype=torch.long)
    
    @staticmethod
    def create_self_loops(num_nodes, device):
        """为所有节点创建自环"""
        indices = torch.arange(num_nodes, device=device)
        return torch.stack([indices, indices], dim=0)


class TemporalGLU(nn.Module):
    """Temporal Gated Linear Unit (GLU) for STGCN - 符合论文原始设计
    
    实现文献中的GLU门控时序卷积：
    - 1D卷积在时间维度上
    - GLU门控机制：输出 = P ⊙ σ(Q)，其中P、Q为卷积输出的两部分
    - 支持残差连接和归一化
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.1,
                 norm_type: str = "group"):
        super(TemporalGLU, self).__init__()
        
        # GLU需要输出2倍通道数，然后分成P和Q两部分
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels * 2,  # 输出2倍通道用于GLU
            kernel_size=(1, kernel_size), 
            padding=(0, kernel_size // 2)  # 保持时间序列长度
        )
        
        # 归一化层：BatchNorm2d 在 batch=1 时易退化为恒定输出，改为可配置
        self.norm_type = norm_type
        if norm_type == "batch":
            self.bn = nn.BatchNorm2d(out_channels * 2)
        elif norm_type == "layer":
            # LayerNorm 需要在 forward 中动态 reshape，这里用 GroupNorm(1, C) 近似通道层归一化
            self.bn = nn.GroupNorm(1, out_channels * 2)
        elif norm_type == "group":
            # 组归一化：较稳健，不依赖 batch 大小
            num_groups = max(1, min(8, out_channels * 2 // 4))  # 自适应分组数
            self.bn = nn.GroupNorm(num_groups, out_channels * 2)
        else:
            self.bn = nn.Identity()
        
        # Dropout正则化（可配置）
        self.dropout = nn.Dropout2d(dropout)
        
        # 残差连接（如果输入输出维度不同）
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None
        
    def forward(self, x):
        # x: [batch_size, in_channels, num_nodes, seq_len]
        residual = x
        
        # 时序卷积 + 批量归一化
        conv_out = self.conv(x)  # [batch_size, out_channels*2, num_nodes, seq_len]
        conv_out = self.bn(conv_out)
        conv_out = self.dropout(conv_out)
        
        # GLU门控机制：分成P和Q两部分
        P, Q = conv_out.chunk(2, dim=1)  # 各自为 [batch_size, out_channels, num_nodes, seq_len]
        
        # GLU公式：P ⊙ σ(Q)
        glu_out = P * torch.sigmoid(Q)
        
        # 残差连接
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        
        # 确保维度匹配后相加
        if residual.shape != glu_out.shape:
            # 处理时间序列长度不匹配的情况
            min_seq_len = min(residual.shape[-1], glu_out.shape[-1])
            residual = residual[:, :, :, :min_seq_len]
            glu_out = glu_out[:, :, :, :min_seq_len]
        
        # 残差连接 + 最终激活
        out = glu_out + residual
        
        return out


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for STGCN - 基于GLU的三层时序卷积"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.1,
                 norm_type: str = "group"):
        super(TemporalConvNet, self).__init__()
        
        # 🔧 使用GLU门控的三层时序卷积（增强时序学习能力）
        # 设计三层渐进式通道变化：输入 -> 扩展 -> 保持 -> 输出
        
        # 计算中间层通道数，确保有足够的表达能力
        # 第一层扩展到较大维度以捕捉复杂时序模式
        expand_channels = max(in_channels * 2, out_channels * 2, 64)
        # 第二层保持扩展维度进行深度特征提取
        maintain_channels = expand_channels
        
        # 第一层：输入 -> 扩展维度（捕捉初级时序模式）
        self.glu1 = TemporalGLU(in_channels, expand_channels, kernel_size, dropout, norm_type=norm_type)
        
        # 第二层：扩展维度 -> 保持维度（深度时序特征提取）
        self.glu2 = TemporalGLU(expand_channels, maintain_channels, kernel_size, dropout, norm_type=norm_type)
        
        # 第三层：保持维度 -> 输出维度（特征整合和输出）
        self.glu3 = TemporalGLU(maintain_channels, out_channels, kernel_size, dropout, norm_type=norm_type)
        
        # 可选的残差连接（当输入输出维度相同时）
        if in_channels == out_channels:
            self.residual_connection = True
        else:
            self.residual_connection = False
            # 添加1x1卷积用于维度匹配的残差连接
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # x: [batch_size, in_channels, num_nodes, seq_len]
        residual = x
        
        # 三层GLU时序卷积
        x = self.glu1(x)  # 第一层：扩展维度
        x = self.glu2(x)  # 第二层：深度特征提取  
        x = self.glu3(x)  # 第三层：输出整合
        
        # 残差连接（有助于训练深层网络）
        if self.residual_connection:
            x = x + residual
        else:
            # 通过1x1卷积匹配维度后进行残差连接
            residual = self.residual_conv(residual)
            x = x + residual
            
        return x


class SpatialGCN(nn.Module):
    """Spatial Graph Convolutional Network for STGCN - 标准版本"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(SpatialGCN, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels, cached=False)  # 不使用缓存
        
    def forward(self, x, edge_index, batch_vector=None):
        # x: [batch_size * num_nodes, in_channels]
        # edge_index: [2, num_edges]
        x = self.gcn(x, edge_index)
        x = torch.relu(x)
        return x


class STGCNBlock(nn.Module):
    """STGCN Block with Bottleneck Strategy: Temporal Conv -> Spatial GCN -> Temporal Conv
    
    实现文献中的瓶颈策略（Bottleneck Strategy）：
    - 通道压缩：第一个时序卷积将输入通道压缩到较小的spatial_channels
    - 空间卷积：在压缩的特征空间中进行高效的图卷积
    - 通道扩展：第二个时序卷积将特征扩展回输出通道数
    - 残差连接：跨整个Block的残差连接
    """
    
    def __init__(self, in_channels: int, spatial_channels: int, out_channels: int, 
                 num_nodes: int, kernel_size: int = 3, bottleneck_ratio: float = 0.5,
                 tcn_dropout: float = 0.1, norm_type: str = "group"):
        super(STGCNBlock, self).__init__()
        
        # 🔧 瓶颈策略：计算压缩后的通道数
        # bottleneck_channels通常比spatial_channels更小，实现特征压缩
        bottleneck_channels = max(1, int(spatial_channels * bottleneck_ratio))
        
        print(f"STGCNBlock瓶颈策略: {in_channels} → {bottleneck_channels} → {spatial_channels} → {out_channels}")
        
        # 第一个时序卷积：输入通道 → 瓶颈通道（压缩）
        self.temporal1 = TemporalConvNet(in_channels, bottleneck_channels, kernel_size, dropout=tcn_dropout, norm_type=norm_type)
        
        # 空间图卷积：瓶颈通道 → 空间通道（在压缩特征上进行图卷积，然后扩展）
        self.spatial = SpatialGCN(bottleneck_channels, spatial_channels)
        
        # 第二个时序卷积：空间通道 → 输出通道（扩展）
        # 注意：这里输入应该是spatial_channels，即GCN的输出通道数
        self.temporal2 = TemporalConvNet(spatial_channels, out_channels, kernel_size, dropout=tcn_dropout, norm_type=norm_type)
        
        # Layer Normalization（更适合时空序列，符合原文）
        # 只对通道维度进行归一化，适应动态的时序长度和节点数
        self.layer_norm = nn.LayerNorm(out_channels)
        
        # Residual connection across the entire block
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None
            
    def forward(self, x, edge_index, batch_vector=None, debug: bool = False):
        # x: [batch_size, in_channels, num_nodes, seq_len]
        batch_size, in_channels, num_nodes, seq_len = x.shape

        # Store input for residual connection
        residual = x

        # Temporal convolution 1
        x = self.temporal1(x)  # [batch_size, spatial_channels, num_nodes, seq_len]
        if debug:
            try:
                with torch.no_grad():
                    s = tuple(x.shape)
                    print(f"[STGCNBlock Debug] After temporal1: shape={s}, mean={x.mean().item():.5f}, std={x.std().item():.5f}")
            except Exception:
                pass

        # 🎯 高效无for循环空间图卷积处理（拼大图前只做一次合法性检查，offset严格基于base_num_nodes）
        batch_size, spatial_channels, num_nodes, seq_len = x.shape
        # base_num_nodes 用于时间展开偏移，必须>=实际num_nodes，避免无边节点导致的负pad
        edge_index_cpu = edge_index.cpu()
        if edge_index_cpu.numel() > 0:
            inferred_nodes = edge_index_cpu.max().item() + 1
        else:
            inferred_nodes = num_nodes
        base_num_nodes = max(num_nodes, inferred_nodes)
        edge_index_safe = SimpleBatchIndexManager.validate_and_fix_edge_index(edge_index_cpu, base_num_nodes, device='cpu')
        edge_index_safe = edge_index_safe.to(dtype=torch.long)
        # x_reshape补pad到base_num_nodes，确保每个子图节点数一致
        x_reshape = x.permute(0, 3, 2, 1).contiguous()  # [B, T, N, C]
        pad_size = base_num_nodes - num_nodes
        if pad_size != 0:
            # F.pad 支持pad_size为0时自动无影响
            x_reshape = F.pad(x_reshape, (0, 0, 0, pad_size))  # [B, T, base_num_nodes, C]
        x_reshape = x_reshape.view(batch_size * seq_len * base_num_nodes, spatial_channels)  # [B*T*base_num_nodes, C]

        # offset严格用base_num_nodes
        edge_index_list = []
        total_graphs = batch_size * seq_len
        for i in range(total_graphs):
            offset = i * base_num_nodes
            edge_index_list.append(edge_index_safe + offset)
        big_edge_index = torch.cat(edge_index_list, dim=1).to(x.device)  # [2, E * (B*T)]

        # 断言：big_edge_index所有索引都在[0, x_reshape.shape[0])
        num_total_nodes = x_reshape.shape[0]
        assert big_edge_index.max().item() < num_total_nodes, f"big_edge_index max {big_edge_index.max().item()} >= num_total_nodes {num_total_nodes}"
        assert big_edge_index.min().item() >= 0, f"big_edge_index min {big_edge_index.min().item()} < 0"
        # 打印调试信息
    # 调试输出已移除

        try:
            x_gcn = self.spatial(x_reshape, big_edge_index)
        except Exception as e:
            x_gcn = x_reshape
        if debug:
            try:
                with torch.no_grad():
                    s = tuple(x_gcn.shape)
                    print(f"[STGCNBlock Debug] After spatial GCN: shape={s}, mean={x_gcn.mean().item():.5f}, std={x_gcn.std().item():.5f}")
            except Exception:
                pass

        # 自动推断实际节点数，防止shape不匹配
        total_gcn = x_gcn.shape[0]
        actual_nodes = total_gcn // (batch_size * seq_len)
        assert total_gcn % (batch_size * seq_len) == 0, f"GCN输出长度不能整除batch*seq_len: {total_gcn} vs {batch_size*seq_len}"
        # 🔧 修正：使用实际的GCN输出通道数
        actual_gcn_channels = x_gcn.shape[1]  # GCN的实际输出通道数
        x_gcn = x_gcn.view(batch_size, seq_len, actual_nodes, actual_gcn_channels)
        if actual_nodes > num_nodes:
            x_gcn = x_gcn[:, :, :num_nodes, :]
        x_gcn = x_gcn.permute(0, 3, 2, 1).contiguous()  # [B, actual_gcn_channels, N, T]
        x = x_gcn
        
        # Temporal convolution 2
        x = self.temporal2(x)  # [batch_size, out_channels, num_nodes, seq_len]
        if debug:
            try:
                with torch.no_grad():
                    s = tuple(x.shape)
                    print(f"[STGCNBlock Debug] After temporal2: shape={s}, mean={x.mean().item():.5f}, std={x.std().item():.5f}")
            except Exception:
                pass
        
        # Apply Layer Normalization（符合原文设计）
        # x: [batch_size, out_channels, num_nodes, seq_len]
        # 对通道维度进行归一化，保持其他维度不变
        x_for_norm = x.permute(0, 2, 3, 1)  # [batch_size, num_nodes, seq_len, out_channels]
        x_normed = self.layer_norm(x_for_norm)  # 对最后一个维度(out_channels)归一化
        x = x_normed.permute(0, 3, 1, 2)  # 恢复到 [batch_size, out_channels, num_nodes, seq_len]
        
        # Residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        # 修正节点数shape不一致问题
        if residual.shape[2] < x.shape[2]:
            pad_size = x.shape[2] - residual.shape[2]
            residual = F.pad(residual, (0, 0, 0, pad_size))
        elif residual.shape[2] > x.shape[2]:
            residual = residual[:, :, :x.shape[2], :]
        x = x + residual
        if debug:
            try:
                with torch.no_grad():
                    s = tuple(x.shape)
                    print(f"[STGCNBlock Debug] After residual+LN: shape={s}, mean={x.mean().item():.5f}, std={x.std().item():.5f}")
            except Exception:
                pass
        return torch.relu(x)


class STGCN(nn.Module):
    """
    Standard Spatio-Temporal Graph Convolutional Network
    
    Args:
        in_channels: Number of input features per node
        hidden_channels: Number of hidden channels in STGCN blocks
        out_channels: Number of output features
        num_nodes: Number of nodes in the graph
        seq_len: Input sequence length
        output_time: Output sequence length
        num_blocks: Number of STGCN blocks
        kernel_size: Kernel size for temporal convolution
        dropout: Dropout rate
    """
    
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int = 64,
                 out_channels: int = 1,
                 num_nodes: int = 6,
                 seq_len: int = 102,
                 output_time: int = 12,
                 num_layers: int = 2,  # 🔧 改为num_layers，与训练配置一致
                 kernel_size: int = 3,
                 dropout: float = 0.3,
                 **kwargs):
        super(STGCN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.output_time = output_time
        self.num_layers = num_layers  # 🔧 改为num_layers
        
        # 新增：可配置的时序卷积与输出头的dropout
        self.tcn_dropout = kwargs.get('tcn_dropout', 0.1)
        # 输出头dropout默认沿用dropout参数，可单独覆盖
        self.output_head_dropout = kwargs.get('output_head_dropout', dropout)
        # 归一化类型（batch|layer|group|none）
        self.norm_type = kwargs.get('norm_type', 'group')
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        
        # STGCN blocks - 增强版本：瓶颈策略 + GLU门控
        self.stgcn_blocks = nn.ModuleList()
        for i in range(num_layers):  # 🔧 使用num_layers
            block_in_channels = hidden_channels if i == 0 else hidden_channels
            # 🔧 瓶颈策略：空间卷积通道数设置为hidden_channels的一半，实现特征压缩
            spatial_channels = hidden_channels // 2  # 瓶颈压缩
            self.stgcn_blocks.append(
                STGCNBlock(
                    in_channels=block_in_channels, 
                    spatial_channels=spatial_channels,  # 瓶颈策略：压缩空间特征
                    out_channels=hidden_channels, 
                    num_nodes=num_nodes, 
                    kernel_size=kernel_size,
                    bottleneck_ratio=0.5,  # 瓶颈比例：进一步压缩特征
                    tcn_dropout=self.tcn_dropout,
                    norm_type=self.norm_type
                )
            )
        
        # Output projection - 温和版本，更适合流量预测的平滑输出
        norm_head: nn.Module
        if self.norm_type == 'batch':
            norm_head = nn.BatchNorm2d(hidden_channels // 2)
        elif self.norm_type == 'layer':
            norm_head = nn.GroupNorm(1, hidden_channels // 2)
        elif self.norm_type == 'group':
            num_groups = max(1, min(8, (hidden_channels // 2) // 4))
            norm_head = nn.GroupNorm(num_groups, hidden_channels // 2)
        else:
            norm_head = nn.Identity()
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1),
            norm_head,
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(self.output_head_dropout),
            nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=1),
        )
        
        # 时间映射方法：'pool'（自适应平均）或 'last'（直接截取最近T_out步），默认'pool'
        self.time_mapping_method = kwargs.get('time_mapping_method', 'pool')
        if self.time_mapping_method not in ('pool', 'last'):
            self.time_mapping_method = 'pool'
        
    def forward(self, x, edge_index, **kwargs):
        """
        单图推理模式：
        x: [num_nodes, seq_len, in_channels]
        edge_index: [2, num_edges]，全局节点编号
        返回: [num_nodes, output_time, out_channels]
        """
        if x.dim() != 3:
            raise ValueError(f"STGCN单图推理要求输入为[num_nodes, seq_len, in_channels]，实际{x.shape}")
        num_nodes, seq_len, in_channels = x.shape
        # [1, num_nodes, seq_len, in_channels]
        x = x.unsqueeze(0)
        # [1, in_channels, num_nodes, seq_len]
        x = x.permute(0, 3, 1, 2)
        x = self.input_proj(x)
        debug = kwargs.get('debug_shapes', False) or kwargs.get('debug', False)
        if debug:
            try:
                with torch.no_grad():
                    s = tuple(x.shape)
                    print(f"[STGCN Debug] After input_proj: shape={s}, mean={x.mean().item():.5f}, std={x.std().item():.5f}")
            except Exception:
                pass
        for i, block in enumerate(self.stgcn_blocks):
            x = block(x, edge_index, debug=debug)
            if debug:
                try:
                    with torch.no_grad():
                        s = tuple(x.shape)
                        print(f"[STGCN Debug] After block[{i}]: shape={s}, mean={x.mean().item():.5f}, std={x.std().item():.5f}")
                except Exception:
                    pass
        b, c, n, t = x.shape
        if self.output_time != t:
            if self.time_mapping_method == 'last' and self.output_time <= t:
                # 直接截取最近的T_out步，减少平均带来的抹平
                x = x[:, :, :, -self.output_time:]
            else:
                # 默认：自适应平均到T_out
                x_temp = x.reshape(b * c * n, t)
                x_temp = F.adaptive_avg_pool1d(x_temp.unsqueeze(1), self.output_time).squeeze(1)
                x = x_temp.reshape(b, c, n, self.output_time)
        if debug:
            try:
                with torch.no_grad():
                    s = tuple(x.shape)
                    print(f"[STGCN Debug] After time-mapping ('{self.time_mapping_method}'): shape={s}, mean={x.mean().item():.5f}, std={x.std().item():.5f}")
            except Exception:
                pass
        x = self.output_proj(x)
        # 如果输出几乎恒定（极低方差），打印一次警告（仅在训练阶段）
        if self.training:
            with torch.no_grad():
                var = x.var().item()
                if not hasattr(self, '_warned_low_variance') and var < 1e-6:
                    print(f"[STGCN Warning] 输出方差极低 ({var:.2e})，可能出现恒定预测。建议尝试：norm_type='group' 或降低聚合平滑。")
                    self._warned_low_variance = True
        if debug:
            try:
                with torch.no_grad():
                    s = tuple(x.shape)
                    print(f"[STGCN Debug] After output_head: shape={s}, mean={x.mean().item():.5f}, std={x.std().item():.5f}")
            except Exception:
                pass
        x = x.permute(0, 2, 3, 1)  # [1, num_nodes, output_time, out_channels]
        # 保持batch维度，不要squeeze，这样输出就是[1, num_nodes, output_time, out_channels]
        return x


class STGCNModel(STGCN):
    """Wrapper class for compatibility with existing codebase with graph-level aggregation support"""
    
    def __init__(self, 
                 num_features: Optional[int] = None,
                 output_size: int = 1,
                 data_feature: Optional[dict] = None,
                 **kwargs):
        # Handle parameter aliasing for compatibility
        in_channels = kwargs.pop('in_channels', num_features or 1)
        out_channels = kwargs.pop('out_channels', output_size)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        )
        
        # 图级聚合功能配置
        if data_feature is None:
            data_feature = {}
            
        self.graph_aggregation_enabled = data_feature.get("enable_graph_aggregation", False)
        
        if self.graph_aggregation_enabled:
            # 步骤5：图级聚合相关参数
            self.aggregation_method = data_feature.get("aggregation_method", "mean")  # mean, max, sum
            # 注意力聚合（可选）：为每个节点输出生成权重，沿节点维softmax
            if self.aggregation_method == "attention":
                # 基于每个时间步的节点表征 h_{n,t} ∈ R^C 生成标量注意力
                self.node_attn = nn.Linear(out_channels, 1)
            
            # 步骤6：输出投影层（从STGCN输出维度到最终输出维度）
            # 计算输出投影层的输入维度
            projection_input_dim = out_channels
            projection_output_dim = data_feature.get("final_output_dim", out_channels)
            self.output_projection = torch.nn.Linear(projection_input_dim, projection_output_dim)
            
            # 步骤7：时间截断参数
            self.truncate_time = data_feature.get("truncate_time", None)  # 截断到指定时间步数
            
            print(f"[STGCNModel] Graph aggregation enabled with method: {self.aggregation_method}")
            print(f"[STGCNModel] Output projection: {projection_input_dim} -> {projection_output_dim}")
            if self.truncate_time is not None:
                print(f"[STGCNModel] Time truncation enabled, keeping last {self.truncate_time} time steps")

        # 是否将所有样本一次性拼成一张大图处理（更高效，与示意图一致）
        # 默认开启；如需回退到逐样本处理，可在data_feature中设置 process_all_in_one=False
        self.process_all_in_one = data_feature.get("process_all_in_one", True)
        # 是否在大图路径中过滤跨图边（强烈建议开启）
        self.filter_cross_graph_edges = data_feature.get("filter_cross_graph_edges", True)
        # 新增：是否返回按batch_vector复原的节点级输出（保持训练兼容，默认False）
        self.return_node_outputs = data_feature.get("return_node_outputs", False)
        # 新增：轻量调试形状日志（默认关闭）
        self.debug_shapes = data_feature.get("debug_shapes", False)
        # 新增：是否禁用图级线性投影（避免进一步平滑；默认False保持兼容）
        self.disable_output_projection = data_feature.get("disable_output_projection", False)
        # 新增：是否禁用输出头BN/Dropout（减少抹平；默认False保持兼容）
        self.disable_output_head_bn_dropout = data_feature.get("disable_output_head_bn_dropout", False)

        # 新增：可选仿射标定（全局通道尺度与偏置），帮助快速校准幅值
        self.enable_affine_calibration = data_feature.get("enable_affine_calibration", True)
        # 标定维度：若存在且启用线性投影，则以其输出维度为准，否则为模型out_channels
        calib_dim = self.out_channels
        if self.graph_aggregation_enabled and hasattr(self, 'output_projection') and (not self.disable_output_projection):
            try:
                calib_dim = self.output_projection.out_features
            except Exception:
                calib_dim = self.out_channels
        self.affine_scale = nn.Parameter(torch.ones(calib_dim))
        self.affine_bias = nn.Parameter(torch.zeros(calib_dim))

        # 可选：根据开关调整输出头（STGCN 输出节点级特征的 1x1 卷积头）
        if self.disable_output_head_bn_dropout:
            # 用最简洁的1x1卷积替换，避免BN/Dropout带来的平滑
            self.output_proj = nn.Sequential(
                nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=1)
            )

    @staticmethod
    def _debug_tensor_stats(name: str, t: torch.Tensor):
        try:
            with torch.no_grad():
                s = t.shape
                mean = t.mean().item() if t.numel() > 0 else float('nan')
                std = t.std().item() if t.numel() > 1 else float('nan')
                t_min = t.min().item() if t.numel() > 0 else float('nan')
                t_max = t.max().item() if t.numel() > 0 else float('nan')
                print(f"[STGCNModel Debug] {name}: shape={s}, mean={mean:.5f}, std={std:.5f}, min={t_min:.5f}, max={t_max:.5f}")
        except Exception as e:
            print(f"[STGCNModel Debug] {name}: stats error -> {e}")

    def _aggregate_nodes_to_graph_single_batch(self, batch_output):
        """
        对单个batch的STGCN输出进行图级聚合
        
        Args:
            batch_output: [1, batch_nodes, output_time, out_channels]
        
        Returns:
            aggregated: [1, output_time, out_channels]
        """
        if self.aggregation_method == "simple_mean":
            # 简单平均聚合：对节点维度取平均
            aggregated = torch.mean(batch_output, dim=1)
            # print(f"[STGCNModel] Single batch simple mean aggregation: {batch_output.shape} -> {aggregated.shape}")
            return aggregated
        elif self.aggregation_method == "mean":
            aggregated = torch.mean(batch_output, dim=1)
            # print(f"[STGCNModel] Single batch mean aggregation: {batch_output.shape} -> {aggregated.shape}")
            return aggregated
        elif self.aggregation_method == "max":
            aggregated, _ = torch.max(batch_output, dim=1)
            # print(f"[STGCNModel] Single batch max aggregation: {batch_output.shape} -> {aggregated.shape}")
            return aggregated
        elif self.aggregation_method == "sum":
            aggregated = torch.sum(batch_output, dim=1)
            # print(f"[STGCNModel] Single batch sum aggregation: {batch_output.shape} -> {aggregated.shape}")
            return aggregated
        elif self.aggregation_method == "attention":
            # batch_output: [1, N, T, C]
            x = batch_output.squeeze(0)  # [N, T, C]
            N, T, C = x.shape
            # 计算每个时间步的注意力：对节点维进行softmax
            attn_logits = self.node_attn(x)  # [N, T, 1]
            attn = torch.softmax(attn_logits, dim=0)  # 节点维归一化
            aggregated = (attn * x).sum(dim=0, keepdim=True)  # [1, T, C]
            return aggregated
        else:
            # 默认使用简单平均聚合
            aggregated = torch.mean(batch_output, dim=1)
            # print(f"[STGCNModel] Single batch default mean aggregation: {batch_output.shape} -> {aggregated.shape}")
            return aggregated
    
    def forward(self, x, edge_index, edge_weight=None, batch_vector=None, **kwargs):
        """
        支持batch处理的前向传播：输入x为[total_nodes, seq_len, in_channels]，
        edge_index为[2, total_edges]，batch_vector为[total_nodes]指示节点的batch归属。
        支持图级聚合功能的输出。
        """
        if batch_vector is None:
            raise ValueError("batch_vector is required for batch processing in STGCNModel.")
        
        device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch_vector = batch_vector.to(device)
        total_nodes = x.shape[0]
        
        if self.debug_shapes:
            self._debug_tensor_stats("Input x", x)
            print(f"[STGCNModel Debug] edge_index shape={tuple(edge_index.shape)} total_nodes={total_nodes}")
        
        # 从batch_vector中获取真实的batch_size
        batch_size = int(batch_vector.max().item()) + 1
        # print(f"[STGCNModel Forward] Detected batch_size: {batch_size}")
        
        edge_index_safe = SimpleBatchIndexManager.validate_and_fix_edge_index(edge_index, total_nodes, device)
        
        # 两种路径：
        # 1) 统一大图路径（推荐）：一次把所有样本拼成大图做GCN，再按batch_vector聚合到图级输出
        # 2) 逐样本路径（兼容旧逻辑）

        node_outputs = None  # 可选的节点级输出（[total_nodes, T_out, C_out]）

        if self.process_all_in_one:
            # 过滤跨图边，确保大图由若干互不相连的子图组成
            if batch_vector is not None and self.filter_cross_graph_edges and edge_index.numel() > 0:
                mask_same_graph = (batch_vector[edge_index[0]] == batch_vector[edge_index[1]])
                edge_index = edge_index[:, mask_same_graph]
                if self.debug_shapes:
                    print(f"[STGCNModel Debug] After cross-graph filter, edge_index shape={tuple(edge_index.shape)}")
            # 合法化边索引
            edge_index_safe = SimpleBatchIndexManager.validate_and_fix_edge_index(edge_index, total_nodes, device)

            # 一次性运行STGCN（把所有样本视为一个大图）
            all_nodes_output = super().forward(x, edge_index_safe, debug_shapes=self.debug_shapes)  # [1, total_nodes, output_time, out_channels]
            if self.debug_shapes:
                self._debug_tensor_stats("All-nodes output (pre-agg)", all_nodes_output)

            if self.return_node_outputs:
                node_outputs = all_nodes_output.squeeze(0)  # [total_nodes, T_out, C_out]

            # 图级聚合：按batch_vector把节点聚合为图
            out_list = []
            if batch_vector is None:
                # 无batch_vector，退化为单图聚合
                if self.graph_aggregation_enabled:
                    # 直接对 [1, total_nodes, T, C] 执行聚合方法
                    y = self._aggregate_nodes_to_graph_single_batch(all_nodes_output)
                else:
                    y = all_nodes_output.mean(dim=1)
                out_list.append(y)
            else:
                for b in range(batch_size):
                    node_mask = (batch_vector == b)
                    if node_mask.any():
                        node_outputs_b = all_nodes_output[0, node_mask, :, :]  # [N_b, T_out, C_out]
                        if self.graph_aggregation_enabled:
                            # 使用配置的聚合方法（sum/mean/max/attention）
                            y_b = self._aggregate_nodes_to_graph_single_batch(node_outputs_b.unsqueeze(0))  # [1, T_out, C_out]
                        else:
                            y_b = node_outputs_b.mean(dim=0, keepdim=True)  # [1, T_out, C_out]
                    else:
                        y_b = torch.zeros(1, self.output_time, self.out_channels, device=device)
                    out_list.append(y_b)
            final_output = torch.cat(out_list, dim=0)  # [B or 1, T_out, C_out]
            if self.debug_shapes:
                self._debug_tensor_stats("Graph-level output (post-agg, pre-proj)", final_output)
        else:
            # 原逐样本路径
            # STGCN期待单图输入，需要将batch处理的输入转换
            # 这里我们逐个batch处理，然后聚合结果
            batch_outputs = []
            # 可选：为逐样本路径准备节点级输出容器
            if self.return_node_outputs:
                node_outputs = torch.zeros(total_nodes, self.output_time, self.out_channels, device=device)

            for b in range(batch_size):
                # 获取当前batch的节点
                mask = (batch_vector == b)
                if mask.sum() == 0:
                    # 如果某个batch没有节点，创建零输出
                    zero_output = torch.zeros(1, self.output_time, self.out_channels, device=device)
                    batch_outputs.append(zero_output)
                    continue

                batch_nodes = torch.where(mask)[0]
                x_batch = x[mask]  # [batch_nodes, seq_len, in_channels]

                # 重新映射边索引（只保留当前batch内的边）
                # 创建节点映射：从全局索引到batch内索引
                node_mapping = torch.full((total_nodes,), -1, dtype=torch.long, device=device)
                node_mapping[batch_nodes] = torch.arange(len(batch_nodes), device=device)

                # 筛选边索引
                valid_edges = (batch_vector[edge_index_safe[0]] == b) & (batch_vector[edge_index_safe[1]] == b)
                if valid_edges.sum() > 0:
                    edge_index_batch = edge_index_safe[:, valid_edges]
                    # 重新映射边索引到batch内索引
                    edge_index_batch = node_mapping[edge_index_batch]
                else:
                    # 如果没有边，创建空的边索引
                    edge_index_batch = torch.empty((2, 0), dtype=torch.long, device=device)

                # 对当前batch运行STGCN
                batch_output = super().forward(x_batch, edge_index_batch, debug_shapes=self.debug_shapes)  # [1, batch_nodes, output_time, out_channels]
                if self.debug_shapes:
                    self._debug_tensor_stats(f"Per-graph {b} node output (pre-agg)", batch_output)

                if self.return_node_outputs:
                    # 写回节点级输出容器
                    node_outputs[mask] = batch_output.squeeze(0)  # [N_b, T_out, C]

                # 聚合当前batch的节点到图级别
                if self.graph_aggregation_enabled:
                    # 使用图聚合方法
                    if batch_output.dim() == 4 and batch_output.shape[0] == 1:
                        # [1, batch_nodes, output_time, out_channels] -> [1, output_time, out_channels]
                        batch_aggregated = self._aggregate_nodes_to_graph_single_batch(batch_output)
                    else:
                        batch_aggregated = batch_output
                else:
                    # 简单平均聚合
                    if batch_output.dim() == 4 and batch_output.shape[0] == 1:
                        batch_aggregated = batch_output.mean(dim=1)  # [1, output_time, out_channels]
                    else:
                        batch_aggregated = batch_output

                batch_outputs.append(batch_aggregated)

            # 将所有batch的输出堆叠
            final_output = torch.cat(batch_outputs, dim=0)  # [batch_size, output_time, out_channels]
        # print(f"[STGCNModel Forward] Final output shape: {final_output.shape}")
        
        # 步骤6：输出投影
        if (not getattr(self, 'disable_output_projection', False)) and hasattr(self, 'output_projection') and self.output_projection is not None:
            projected = self.output_projection(final_output)
            if self.debug_shapes:
                self._debug_tensor_stats("Graph-level output (post-proj)", projected)
            final_output = projected
            if self.return_node_outputs and node_outputs is not None:
                node_outputs = self.output_projection(node_outputs)
                if self.debug_shapes:
                    self._debug_tensor_stats("Node-level output (post-proj)", node_outputs)
        
        # 步骤7：时间维度截断
        if hasattr(self, 'truncate_time') and self.truncate_time is not None and final_output.dim() >= 2:
            time_dim = -2
            if final_output.shape[time_dim] > self.truncate_time:
                truncated = final_output[:, -self.truncate_time:, :]
                if self.debug_shapes:
                    self._debug_tensor_stats("Graph-level output (post-trunc)", truncated)
                final_output = truncated
            if self.return_node_outputs and node_outputs is not None and node_outputs.dim() >= 2:
                if node_outputs.shape[time_dim] > self.truncate_time:
                    node_outputs = node_outputs[:, -self.truncate_time:, :]
                    if self.debug_shapes:
                        self._debug_tensor_stats("Node-level output (post-trunc)", node_outputs)

        # 步骤8：仿射标定（a*x + b），默认开启但初始化为恒等映射（a=1, b=0）
        if getattr(self, 'enable_affine_calibration', False):
            # final_output: [B, T, C]
            scale = self.affine_scale.view(1, 1, -1)
            bias = self.affine_bias.view(1, 1, -1)
            final_output = final_output * scale + bias
            if self.return_node_outputs and node_outputs is not None:
                node_outputs = node_outputs * scale + bias
        
        return final_output, node_outputs if self.return_node_outputs else None, None, None


# Export for model registry
__all__ = ['STGCN', 'STGCNModel']
