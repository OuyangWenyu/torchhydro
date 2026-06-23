import torch
import torch.nn as nn
import torch.nn.functional as F
import torchhydro.models.dpl4xaj as dpl4xaj


class CNNModule(nn.Module):
    """
    定义CNN模块
    CNN模块用于从输入的多源数据中提取空间特征。这可以通过定义一个简单的CNN层来实现。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1
    ):
        """
        Parameters
        ----------
        in_channels
            输入通道数，在多源数据处理中，每个数据源可以视为一个通道
        out_channels
            输出通道数，即CNN层的卷积核数，指定了卷积层将会产生的特征图（feature maps）的数量。
        kernel_size
            卷积核大小，指定了卷积核的高度和宽度。
        padding
            卷积核填充大小，在输入数据的每一边都填充1个像素的0值，使得卷积层的输出尺寸与输入尺寸相同，前提是步长（stride）为1。
        """

        super(CNNModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class TransformerModule(nn.Module):
    """
    定义transformer模块
    transformer模块用于处理时间和跨源依赖性。
    """

    def __init__(
        self,
        input_dim,
        num_heads,
        num_layers
    ):
        """
        Parameters
        ----------
        input_dim
            输入和输出向量的维度，即每个时间步的特征维度。
        num_heads
            多头注意力机制的头数，指定了注意力机制的并行数。
            每个头会独立地学习输入数据的不同方面，头数通常是一个可以被input_dim整除的数，确保每个头输出的维度相等。
        num_layers
            transformer模块的层数，指定了模块中编码器和解码器的层数。
        """

        super(TransformerModule, self).__init__()
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )

    def forward(self, src, tgt):
        return self.transformer(src, tgt)


class CNNTransformerModule(nn.Module):
    """
    整合CNN和transformer模块
    将特定的CNN层用于空间特征提取，接着使用transformer层来捕获这些特征之间的长距离依赖性，确保在每个特征提取级别都应用注意力机制
    """

    def __init__(
        self,
        num_sources,
        cnn_output_dim,
        transformer_heads,
        transformer_layers,
        kernel_size,
        padding):
        """
        Parameters
        ----------
        num_sources
            输入数据的源数，即输入数据的通道数。
        cnn_output_dim
            CNN模块的输出维度，指定了CNN层的输出通道数。
            每个CNN模块输出的特征维度，影响了后续变压器模块的输入维度，因为变压器的输入需要与CNN输出的维度相匹配。
        transformer_heads
            多头注意力机制的头数，指定了注意力机制的并行数。
            决定了模型在处理输入特征时能并行处理的子空间数量。
        transformer_layers
            transformer模块的层数，指定了模块中编码器和解码器的层数。
        """

        super(CNNTransformerModule, self).__init__()
        self.cnn_modules = nn.ModuleList(
            [CNNModule(1, cnn_output_dim, kernel_size, padding) for _ in range(num_sources)])
        self.transformer = TransformerModule(cnn_output_dim, transformer_heads, transformer_layers)
        self.fc = nn.Linear(cnn_output_dim, cnn_output_dim)  # 保持维度与dpl4xaj_model输入维度相同

    def forward(self, x):
        cnn_outputs = [cnn(x[:, i:i + 1]) for i, cnn in
                       enumerate(self.cnn_modules)]  # 假设x的形状是[batch_size, num_sources, height, width]
        combined = torch.cat(cnn_outputs, dim=1)  # 沿特征维合并
        combined = combined.permute(2, 3, 0, 1).contiguous()  # 调整为transformer所需的维度[seq_len, batch, features]
        transformer_output = self.transformer(combined, combined)
        transformer_output = transformer_output.permute(2, 0, 1, 3).contiguous()
        out = self.fc(transformer_output[:, :, :, 0])  # 取序列的第一个元素作为输出
        return out.squeeze(-1)  # 确保输出是一个二维张量 [batch_size, seq_len]


class CNN_LSTM_Model(nn.Module):
    """
    创建降雨融合模型
    该模型将CNNTransformer和dpl4xaj模块整合在一起，确保数据流从CNNTransformer到dpl4xaj。
    整合CNN和变压器模块，确保在每个特征提取级别都应用注意力机制
    """

    def __init__(self,
                 num_sources,
                 out_channels,
                 transformer_heads,
                 transformer_layers,
                 n_input_features,
                 n_output_features,
                 n_hidden_states,
                 kernel_size_hydrograph,
                 warmup_length,
                 param_limit_func="sigmoid",
                 param_test_way="final",
                 source_book="HF",
                 source_type="sources",
                 kernel_size=3,
                 padding=1):
        super(CNN_LSTM_Model, self).__init__()
        self.cnn_transformer_model = CNNTransformerModule(num_sources, out_channels, transformer_heads,
                                                          transformer_layers, kernel_size, padding)
        self.dpl4xaj_model = dpl4xaj.DplLstmXaj(n_input_features, n_output_features, n_hidden_states,
                                                kernel_size_hydrograph, warmup_length, param_limit_func, param_test_way,
                                                source_book, source_type)

    def forward(self, x, z, t):
        cnn_transformer_output = self.CNNTransformer_model(x)
        cnn_transformer_output = cnn_transformer_output.unsqueeze(1)  # 增加一个序列长度的维度
        dpl4xaj_output = self.dpl4xaj_model(cnn_transformer_output)
        return dpl4xaj_output
