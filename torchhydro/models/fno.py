import torch
import itertools
from copy import deepcopy
from typing import List, Union, Optional, Tuple
from torch import nn
import torch.fft
from torch.nn import functional as F
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


class CGELU(nn.Module):
    """GELU applied separately to the real and imaginary parts of a complex tensor."""

    def forward(self, x):
        return torch.complex(F.gelu(x.real), F.gelu(x.imag))
Number = Union[int, float]


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """
    由于去掉了 FactorizedTensor，统一返回原生 Dense 收缩函数
    """
    return _contract_dense


def validate_scaling_factor(
    scaling_factor: Union[None, Number, List[Number], List[List[Number]]],
    n_dim: int,
    n_layers: Optional[int] = None,
) -> Union[None, List[float], List[List[float]]]:
    """
    Parameters
    ----------
    scaling_factor : None OR float OR list[float] Or list[list[float]]
    n_dim : int
    n_layers : int or None; defaults to None
        If None, return a single list (rather than a list of lists)
        with `factor` repeated `dim` times.
    """
    if scaling_factor is None:
        return None
    if isinstance(scaling_factor, (float, int)):
        if n_layers is None:
            return [float(scaling_factor)] * n_dim

        return [[float(scaling_factor)] * n_dim] * n_layers

    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (float, int)) for s in scaling_factor])
    ):
        if n_layers is None and len(scaling_factor) == n_dim:
            # this is a dim-wise scaling
            return [float(s) for s in scaling_factor]
        return [[float(s)] * n_dim for s in scaling_factor]

    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (list)) for s in scaling_factor])
    ):
        s_sub_pass = True
        for s in scaling_factor:
            if all([isinstance(s_sub, (int, float)) for s_sub in s]):
                pass
            else:
                s_sub_pass = False
            if s_sub_pass:
                return scaling_factor

    return None


def einsum_complexhalf(eq, *args):
    """
    Compute einsum for complexhalf tensors.
    Because torch.einsum currently does not support complex32 (complexhalf) types.
    The inputs and outputs are the same as in torch.einsum
    """
    import opt_einsum
    import tensorly as tl

    # find the optimal path
    _, path_info = opt_einsum.contract_path(eq, *args)
    partial_eqns = [
        contraction_info[2] for contraction_info in path_info.contraction_list
    ]

    # create a dict of the input tensors by their label in the einsum equation
    tensors = {}
    input_labels = eq.split("->")[0].split(",")
    output_label = eq.split("->")[1]
    tensors = dict(zip(input_labels, args))

    # convert all tensors to half precision and "view as real" form
    for key, tensor in tensors.items():
        tensor = torch.view_as_real(tensor)
        tensor = tensor.half()
        tensors[key] = tensor

    for partial_eq in partial_eqns:
        # get the input tensors to partial_eq
        in_labels, out_label = partial_eq.split("->")
        in_labels = in_labels.split(",")
        in_tensors = [tensors[label] for label in in_labels]

        # create new einsum equation that takes into account "view as real" form
        input_output = partial_eq.split("->")
        new_output = "xy" + input_output[1]
        input_terms = input_output[0].split(",")
        new_inputs = [input_terms[0] + "x", input_terms[1] + "y"]
        new_eqn = new_inputs[0] + "," + new_inputs[1] + "->" + new_output

        # perform the einsum, and convert to "view as real" form
        tmp = tl.einsum(new_eqn, *in_tensors)
        result = torch.stack(
            [tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1
        )
        tensors[out_label] = result

    return torch.view_as_complex(tensors[output_label])


def _contract_dense(x, weight, separable=False):
    # x: (s, b, f) -> sequence_modes, batch, in_features
    # weight: (f, o, s) -> in_features, out_features, sequence_modes

    if separable:
        # separable 模式下 weight 形状通常为 (f, s)
        eq = "sbf,fs->sbf"
    else:
        # 对应关系：
        # s: sequence_modes (x的第0维, weight的第2维)
        # b: batch (x的第1维)
        # f: in_features (x的第2维, weight的第0维)
        # o: out_features (weight的第1维)
        eq = "sbf,fos->sbo"

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, weight)
    else:
        return torch.einsum(eq, x, weight)


class DomainPadding(nn.Module):
    """
    专为水文流量预测设计的 Domain Padding。
    输入形状: (sequence, batch, features)
    处理逻辑: 针对 sequence 维度进行 Padding。
    """

    def __init__(
        self,
        domain_padding: Union[float, list],
        resolution_scaling_factor: Union[int, List[int]] = 1,
    ):
        super().__init__()
        self.domain_padding = domain_padding
        if resolution_scaling_factor is None:
            resolution_scaling_factor = 1
        self.resolution_scaling_factor: Union[int, List[int]] = (
            resolution_scaling_factor
        )

        # 记录 padding 方案：dict(f'{seq_len}' = [pad_left, pad_right])
        self._padding = dict()

        # 记录去 padding 索引：dict(f'{padded_seq_len}' = slice_indices)
        self._unpad_indices = dict()

    def forward(self, x):
        """
        前向传播：对输入进行 Padding
        x: (seq_len, batch, features)
        """
        return self.pad(x)

    def pad(self, x, verbose=False):
        """
        自动根据时间步长度 (resolution) 进行缩放 Padding
        """
        # 将 (seq, batch, feat) 转换为 (batch, feat, seq) 以适配 F.pad
        x = x.permute(1, 2, 0)

        # 在水文任务中，resolution 就是时间序列长度
        resolution = x.shape[2:]  # 结果为 (seq_len,)

        if isinstance(self.domain_padding, (float, int)):
            self.domain_padding = [float(self.domain_padding)] * len(resolution)

        assert len(self.domain_padding) == len(
            resolution
        ), "domain_padding 长度必须匹配时间维度 (1D)"

        resolution_scaling_factor = self.resolution_scaling_factor
        if not isinstance(self.resolution_scaling_factor, list):
            resolution_scaling_factor: List[float] = validate_scaling_factor(
                self.resolution_scaling_factor, len(resolution), n_layers=None
            )

        try:
            padding = self._padding[f"{resolution}"]
            padded = F.pad(x, padding, mode="constant")
            # 还原回 (seq, batch, feat)
            return padded.permute(2, 0, 1)

        except KeyError:
            # 计算 padding 长度：例如 seq=100, padding=0.1 -> pad=10
            padding = [round(p * r) for (p, r) in zip(self.domain_padding, resolution)]

            if verbose:
                print(
                    f"Padding inputs of seq_len={resolution} with padding={padding}, symmetric"
                )

            output_pad = padding
            output_pad = [
                round(i * j) for (i, j) in zip(resolution_scaling_factor, output_pad)
            ]

            # 翻转 padding 列表以适配 F.pad 的从后往前逻辑
            padding = padding[::-1]

            # 对称 Padding 索引计算
            unpad_list = list()
            for p in output_pad:
                if p == 0:
                    padding_end = None
                    padding_start = None
                else:
                    padding_end = p
                    padding_start = -p
                unpad_list.append(slice(padding_end, padding_start, None))

            # 对应 (batch, feat, seq) 维度，前两个维度不进行 unpad
            unpad_indices = (slice(None), slice(None)) + tuple(unpad_list)

            # 构造 F.pad 格式：[pad_left, pad_right]
            padding = [i for p in padding for i in (p, p)]

            self._padding[f"{resolution}"] = padding

            padded = F.pad(x, padding, mode="constant")

            output_shape = padded.shape[2:]
            output_shape = [
                round(i * j) for (i, j) in zip(resolution_scaling_factor, output_shape)
            ]

            self._unpad_indices[f"{[i for i in output_shape]}"] = unpad_indices

            # 还原回 (seq, batch, feat)
            return padded.permute(2, 0, 1)

    def unpad(self, x):
        """
        移除 Padding，输入形状: (seq_len_padded, batch, features)
        """
        # 先转为 (batch, feat, seq) 进行索引切片
        x = x.permute(1, 2, 0)
        unpad_indices = self._unpad_indices[f"{list(x.shape[2:])}"]

        # 执行切片并还原维度
        x = x[unpad_indices]
        return x.permute(2, 0, 1)


class BaseSpectralConv(nn.Module):
    def __init__(self, device=None, dtype=None):
        """Base Class for Spectral Convolutions

        Use it when you want to build your own FNO-type Neural Operators
        """
        super().__init__()

        self.dtype = dtype
        self.device = device

    def transform(self, x):
        """Transforms an input x for a skip connection, by default just an identity map

        If your function transforms the input then you should also implement this transform method
        so the skip connection can also work.

        Typical usecases are:

        * Your upsample or downsample the input in the Spectral conv: the skip connection has to be similarly scaled.
          This allows you to deal with it however you want (e.g. avoid aliasing)
        * You perform a change of basis in your Spectral Conv, again, this needs to be applied to the skip connection too.
        """
        return x


def resample(x, res_scale, axis=None, output_shape=None):
    """
    针对水文数据布局 (Sequence, Batch, Features) 优化的重采样函数。

    参数
    ----------
    x : torch.Tensor
            输入张量，形状为 (S, B, F)
    res_scale: float 或 tuple
            沿 axis 指定维度的缩放比例
    axis: int 或 tuple
            需要重采样的维度。默认为 0 (Sequence 轴)
    output_shape : None 或 tuple[int]
            手动指定输出的序列长度
    """
    # 1. 初始化 axis 和 res_scale
    if axis is None:
        axis = [0]  # 默认对 Sequence 轴进行重采样
    if isinstance(axis, int):
        axis = [axis]

    if isinstance(res_scale, (float, int)):
        res_scale = [res_scale] * len(axis)

    assert len(res_scale) == len(axis), "res_scale 和 axis 的长度必须一致"

    # 2. 计算目标形状 (new_size)
    # 这里的 old_size 取的是 axis 指定的那些维度的当前大小
    old_size = [x.shape[i] for i in axis]
    if output_shape is None:
        new_size = tuple([int(round(s * r)) for (s, r) in zip(old_size, res_scale)])
    else:
        # 如果提供了 output_shape (int 或 tuple)，确保它是 tuple
        new_size = (
            output_shape if isinstance(output_shape, (list, tuple)) else (output_shape,)
        )

    # 3. 针对 1D 空间（水文时间序列常用）的快速插值
    # F.interpolate 期望 (N, C, L)，我们的 x 是 (S, B, F)
    if len(axis) == 1 and axis[0] == 0:
        # (S, B, F) -> (B, F, S)
        x = x.permute(1, 2, 0)
        # 执行插值
        x = F.interpolate(x, size=new_size[0], mode="linear", align_corners=True)
        # (B, F, S) -> (S, B, F)
        return x.permute(2, 0, 1)

    # 4. 针对 2D 的插值 (如果 axis 包含两个维度)
    if len(axis) == 2:
        # 这里假设 axis=[0, 1] 类似情况，需要根据实际需求调整 permute
        # 水文任务通常只在 axis=0 上 resample，此处保留通用性
        x = x.permute(1, 2, 0)  # 临时转换
        x = F.interpolate(
            x.unsqueeze(0), size=new_size, mode="bicubic", align_corners=True
        ).squeeze(0)
        return x.permute(2, 0, 1)

    # 5. 频谱重采样 (Spectral Resampling) 逻辑
    # 适用于更高维或需要更高精度的重采样
    X = torch.fft.rfftn(x.float(), norm="forward", dim=axis)

    # 构造输出频谱的形状
    out_fft_shape = list(x.shape)
    for i, ax in enumerate(axis):
        out_fft_shape[ax] = new_size[i]

    # rfft 的最后一个维度大小是 n//2 + 1
    last_ax = axis[-1]
    out_fft_shape[last_ax] = out_fft_shape[last_ax] // 2 + 1

    out_fft = torch.zeros(out_fft_shape, device=x.device, dtype=torch.cfloat)

    # 频谱切片与填充逻辑
    new_fft_size = [out_fft_shape[i] for i in axis]
    old_fft_size = [X.shape[i] for i in axis]
    new_fft_size_c = [min(i, j) for (i, j) in zip(new_fft_size, old_fft_size)]

    # 构造索引元组进行频谱拷贝
    mode_indexing = [((None, m // 2), (-m // 2, None)) for m in new_fft_size_c[:-1]] + [
        ((None, new_fft_size_c[-1]),)
    ]

    # 这里的 itertools 逻辑保持不变，但作用于我们自定义的维度
    for boundaries in itertools.product(*mode_indexing):
        idx_tuple = [slice(None)] * x.ndim
        for i, ax in enumerate(axis):
            idx_tuple[ax] = slice(*boundaries[i])

        idx_tuple = tuple(idx_tuple)
        out_fft[idx_tuple] = X[idx_tuple]

    y = torch.fft.irfftn(out_fft, s=new_size, norm="forward", dim=axis)

    return y


class SpectralConv(BaseSpectralConv):
    """
    针对水文预测任务优化的 SpectralConv。
    去掉了 FactorizedTensor，使用原生 nn.Parameter。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        complex_data=False,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor=None,
        fno_block_precision="full",
        init_std="auto",
        fft_norm="forward",
        **kwargs,  # 接收并忽略 rank, factorization 等参数以保持兼容性
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.complex_data = complex_data

        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.order = len(self.n_modes)

        if max_n_modes is None:
            max_n_modes = self.n_modes
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self.max_n_modes = max_n_modes

        self.fno_block_precision = fno_block_precision
        self.fft_norm = fft_norm
        self.separable = separable

        # 验证缩放因子 (需外部定义 validate_scaling_factor)
        # self.resolution_scaling_factor = validate_scaling_factor(resolution_scaling_factor, self.order)
        self.resolution_scaling_factor = resolution_scaling_factor

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels)) ** 0.5

        # --- 核心修改：使用 nn.Parameter 替代 FactorizedTensor ---
        if separable:
            if in_channels != out_channels:
                raise ValueError(f"separable 要求输入输出通道一致")
            weight_shape = (in_channels, *self.max_n_modes)
        else:
            weight_shape = (in_channels, out_channels, *self.max_n_modes)

        self.weight = nn.Parameter(torch.empty(weight_shape, dtype=torch.cfloat))
        nn.init.normal_(self.weight, 0, init_std)
        # -------------------------------------------------------

        self._contract = get_contract_fun(self.weight, separable=separable)

        if bias:
            self.bias = nn.Parameter(
                init_std * torch.randn(*((1,) * self.order + (1, self.out_channels)))
            )
        else:
            self.bias = None

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        # 如果是实数 FFT，最后一个维度模式数减半
        if not self.complex_data:
            n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def transform(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        """
        对输入 x 进行重采样，以匹配频谱卷积分支的输出分辨率。
        适配布局: (Sequence, Batch, Features)
        """
        # 在 (S, B, F) 布局中，in_shape 指的是 Sequence 维度
        in_shape = [x.shape[0]]

        # 1. 确定输出形状
        if self.resolution_scaling_factor is not None and output_shape is None:
            # 这里的 resolution_scaling_factor 通常是一个数值或列表
            # 对于水文 1D，我们取第一个元素或直接相乘
            scale = self.resolution_scaling_factor
            if isinstance(scale, list):
                scale = scale[0]

            out_shape = tuple([round(in_shape[0] * scale)])
        elif output_shape is not None:
            # 如果 output_shape 是 tuple，确保它只包含 Sequence 长度
            out_shape = (
                output_shape if isinstance(output_shape, tuple) else (output_shape,)
            )
        else:
            out_shape = tuple(in_shape)

        # 2. 如果形状未变，直接返回
        if tuple(in_shape) == out_shape:
            return x
        else:
            return resample(x, res_scale=1.0, axis=[0], output_shape=output_shape)

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        """
        x 形状: (sequence, batch_size, channels)
        """
        mode_sizes, batchsize, channels = x.shape
        mode_sizes = [mode_sizes]

        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1

        fft_dims = list(range(0, self.order))

        if self.fno_block_precision == "half":
            x = x.half()

        # 正向 FFT
        if self.complex_data:
            x = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims
        else:
            x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims[:-1] if len(fft_dims) > 1 else []

        if self.order > 1 and dims_to_fft_shift:
            x = torch.fft.fftshift(x, dim=dims_to_fft_shift)

        out_dtype = torch.cfloat  # 简化处理，默认使用 cfloat

        # 初始化输出频谱 (S_freq, B, F_out)
        out_fft = torch.zeros(
            [*fft_size, batchsize, self.out_channels], device=x.device, dtype=out_dtype
        )

        # 计算切片索引
        starts = [
            (max_modes - min(size, n_mode))
            for (size, n_mode, max_modes) in zip(
                fft_size, self.n_modes, self.max_n_modes
            )
        ]

        # 构造权重切片 (F_in, F_out, S_modes)
        if self.separable:
            slices_w = [slice(None)]
        else:
            slices_w = [slice(None), slice(None)]

        if self.complex_data:
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(None)
                for start in starts
            ]
        else:
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(None)
                for start in starts[:-1]
            ]
            slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]

        weight = self.weight[tuple(slices_w)]

        # 构造输入 x 的切片 (S, B, F)
        slices_x = []
        weight_start_idx = 1 if self.separable else 2
        for all_modes, kept_modes in zip(
            fft_size, list(weight.shape[weight_start_idx:])
        ):
            center = all_modes // 2
            negative_freqs = kept_modes // 2
            positive_freqs = kept_modes // 2 + kept_modes % 2
            slices_x += [slice(center - negative_freqs, center + positive_freqs)]

        if weight.shape[-1] < fft_size[-1]:
            slices_x[-1] = slice(None, weight.shape[-1])
        else:
            slices_x[-1] = slice(None)

        slices_x = tuple(slices_x) + (slice(None), slice(None))

        # --- 收缩计算 ---
        out_fft[slices_x] = self._contract(
            x[slices_x], weight, separable=self.separable
        )

        # 逆变换
        if self.resolution_scaling_factor is not None and output_shape is None:
            # 简化处理，假设为单维缩放
            mode_sizes = tuple(
                [round(s * self.resolution_scaling_factor) for s in mode_sizes]
            )
        if output_shape is not None:
            mode_sizes = output_shape

        if self.order > 1 and dims_to_fft_shift:
            out_fft = torch.fft.ifftshift(out_fft, dim=dims_to_fft_shift)

        if self.complex_data:
            x = torch.fft.ifftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
        else:
            x = torch.fft.irfftn(
                out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm
            )

        if self.bias is not None:
            x = x + self.bias

        return x


class SoftGating(nn.Module):
    """
    针对水文任务优化的 soft-gating。
    输入 x 尺寸为 `(sequence, batch, features)`，
    返回 `x * w`，其中 w 形状为 `(1, 1, features)`。
    """

    def __init__(self, in_features, out_features=None, n_dim=1, bias=False):
        super().__init__()
        if out_features is not None and in_features != out_features:
            raise ValueError(
                f"Got in_features={in_features} and out_features={out_features}, "
                "but these two must be the same for soft-gating"
            )
        self.in_features = in_features
        self.out_features = out_features

        # 水文适配：将 features 维放在最后。
        # 原代码中使用 *(1,) * n_dim 补充维度，这里我们保持逻辑一致，
        # 结果形状为 (1, 1, in_features)
        self.weight = nn.Parameter(torch.ones(*(1,) * (n_dim + 1), self.in_features))

        if bias:
            self.bias = nn.Parameter(torch.ones(*(1,) * (n_dim + 1), self.in_features))
        else:
            self.bias = None

    def forward(self, x):
        """Applies soft-gating to a batch of activations"""
        if self.bias is not None:
            return self.weight * x + self.bias
        else:
            return self.weight * x


class Flattened1dConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, bias=False):
        """
        水文任务适配版：处理输入形状为 (sequence, batch, features) 的 Skip Layer。
        逻辑：将 sequence 视为卷积长度，features 视为通道。
        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,  # 这里的 in_channels 对应 features
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,  # 保持 sequence 长度不变
            bias=bias,
        )

    def forward(self, x):
        # x.shape: (s, b, f)
        size = list(x.shape)

        # 1. 转换维度以适配 nn.Conv1d: (batch, features, sequence)
        x = x.permute(1, 2, 0)

        # 2. 如果存在额外的维度（针对高维水文张量），逻辑 1:1 复制展平
        # 注意：这里展平的是最后一个维度（原本的 sequence 方向）
        current_shape = list(x.shape)
        x = x.view(*current_shape[:2], -1)

        # 3. 卷积操作
        x = self.conv(x)

        # 4. 还原形状并转回 (sequence, batch, features)
        # 重新映射回原本的序列维度
        x = x.view(size[1], self.conv.out_channels, *size[0:1])
        return x.permute(2, 0, 1)


def skip_connection(
    in_features, out_features, n_dim=1, bias=False, skip_type="soft-gating"
):
    """
    针对水文任务优化的 skip connection 包装器。
    适配输入维度: (sequence, batch, features)
    返回 nn.Module，可选 {'identity', 'linear', 'soft-gating'}

    Parameters
    ----------
    in_features : int
        输入特征维度 (features)
    out_features : int
        输出特征维度
    n_dim : int, optional
        时间维度数量，水文任务通常为 1，默认 1
    bias : bool, optional
        是否使用偏置，默认 False
    skip_type : {'identity', 'linear', 'soft-gating'}, optional
        跳跃连接类型，默认 "soft-gating"

    Returns
    -------
    nn.Module
        执行 skip(x) 的模块
    """
    if skip_type.lower() == "soft-gating":
        # 调用适配 (S, B, F) 的 SoftGating
        return SoftGating(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            n_dim=n_dim,
        )
    elif skip_type.lower() == "linear":
        # 调用适配 (S, B, F) 的 Flattened1dConv
        # 这里的 kernel_size=1 相当于特征维度的线性变换 (Linear Layer)
        return Flattened1dConv(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            bias=bias,
        )
    elif skip_type.lower() == "identity":
        # Identity 无需修改，它直接返回输入
        return nn.Identity()
    else:
        raise ValueError(
            f"Got skip-connection type={skip_type}, expected one of"
            f" {'soft-gating', 'linear', 'identity'}."
        )


def apply_complex(real_func, imag_func, x, dtype=torch.cfloat):
    """
    fr: a function (e.g., conv) to be applied on real part of x
    fi: a function (e.g., conv) to be applied on imag part of x
    x: complex input.
    """
    return (real_func(x.real) - imag_func(x.imag)).type(dtype) + 1j * (
        real_func(x.imag) + imag_func(x.real)
    ).type(dtype)


class ComplexValued(nn.Module):
    """
    Wrapper class that converts a standard nn.Module that operates on real data
    into a module that operates on complex-valued spatial data.
    """

    def __init__(self, module):
        super(ComplexValued, self).__init__()
        self.fr = deepcopy(module)
        self.fi = deepcopy(module)

    def forward(self, x):
        return apply_complex(self.fr, self.fi, x)


class ChannelMLP(nn.Module):
    """
    针对水文任务适配的 Multi-layer perceptron。
    输入 x 形状: (sequence, batch, in_channels)
    输出 x 形状: (sequence, batch, out_channels)
    逻辑：在特征维度（channels）上应用 1D 卷积，对时间步和批次具有不变性。
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        hidden_channels=None,
        n_layers=2,
        n_dim=1,  # 水文任务通常为 1D 序列
        non_linearity=F.gelu,
        dropout=0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
            if dropout > 0.0
            else None
        )

        # 构建 MLP 层，使用 kernel_size=1 的 1D 卷积
        # 这在特征维度上执行线性变换，同时保持序列结构不变
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.in_channels, self.out_channels, 1))
            elif i == 0:
                self.fcs.append(nn.Conv1d(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(
                    nn.Conv1d(self.hidden_channels, self.hidden_channels, 1)
                )

    def forward(self, x):
        """
        x: (sequence, batch, in_channels)
        """
        # 1. 记录原始形状
        size = list(x.shape)  # [S, B, C]

        # 2. 维度重排以适配 nn.Conv1d: (batch, channels, sequence)
        # Conv1d 期望 (N, C, L)，我们将 B 视为 N，features 视为 C，sequence 视为 L
        x = x.permute(1, 2, 0)  # [B, C, S]

        # 3. 处理高维输入（保持原代码逻辑）
        reshaped = False
        if x.ndim > 3:
            # 如果除了 S 还有其他维度，全部展平到最后一个维度
            current_shape = list(x.shape)
            x = x.reshape((*current_shape[:2], -1))
            reshaped = True

        # 4. 应用 MLP 层
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        # 5. 还原形状
        if reshaped:
            # 先还原展平的维度，再还原 permute
            x = x.reshape((size[1], self.out_channels, *size[0:1]))

        # 还原为 (sequence, batch, out_channels)
        x = x.permute(2, 0, 1)  # [S, B, C_out]

        return x


class AdaIN(nn.Module):
    """
    针对水文任务适配的 AdaIN 层。
    输入 x 形状: (sequence, batch, in_channels)
    """

    def __init__(self, embed_dim, in_channels, mlp=None, eps=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.eps = eps

        if mlp is None:
            mlp = nn.Sequential(
                nn.Linear(embed_dim, 512), nn.GELU(), nn.Linear(512, 2 * in_channels)
            )
        self.mlp = mlp
        self.embedding = None

    def set_embedding(self, x):
        """设置流域特征 Embedding (例如形状为 [batch, embed_dim] 或 [embed_dim])"""
        self.embedding = x

    def forward(self, x):
        assert (
            self.embedding is not None
        ), "AdaIN: update embedding before running forward"

        # 1. 计算权重和偏置 [batch, 2*in_channels] -> split -> [batch, in_channels]
        mlp_out = self.mlp(self.embedding)
        weight, bias = torch.split(mlp_out, self.in_channels, dim=-1)

        # 2. 转换维度以使用 group_norm: (S, B, F) -> (B, F, S)
        # group_norm 会在每个样本的 F 和 S 维度上进行归一化
        x = x.permute(1, 2, 0)

        # 3. 应用归一化
        # 注意：weight 和 bias 已经是 per-batch 的，符合 AdaIN 逻辑
        x = nn.functional.group_norm(
            x, self.in_channels, weight.flatten(), bias.flatten(), eps=self.eps
        )

        # 4. 还原维度: (B, F, S) -> (S, B, F)
        return x.permute(2, 0, 1)


class InstanceNorm(nn.Module):
    """针对水文任务的实例归一化，在时间序列(S)上计算统计量"""

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        size = x.shape  # (S, B, F)

        # 转换为 (B, F, S)
        x = x.permute(1, 2, 0)

        # InstanceNorm1d 在 S 维度计算均值方差
        x = torch.nn.functional.instance_norm(x, **self.kwargs)

        # 还原为 (S, B, F)
        x = x.permute(2, 0, 1)
        assert x.shape == size
        return x


class BatchNorm(nn.Module):
    """
    针对水文任务适配的 BatchNorm。
    输入 x: (sequence, batch, num_features)
    """

    def __init__(self, n_dim: int, num_features: int, **kwargs):
        super().__init__()
        self.n_dim = n_dim  # 通常为 1
        self.num_features = num_features
        self.kwargs = kwargs

        # 水文任务是 1D 时间序列，使用 BatchNorm1d
        self.norm = torch.nn.BatchNorm1d(num_features=num_features, **kwargs)

    def forward(self, x):
        size = x.shape  # (S, B, F)

        # BatchNorm1d 期望 (N, C, L)
        # 我们将 B 视为 N，F 视为 C，S 视为 L
        x = x.permute(1, 2, 0)

        # 处理 1D 以上的序列结构（保持原逻辑）
        if self.n_dim >= 2:
            current_shape = x.shape
            x = x.reshape(current_shape[0], current_shape[1], -1)

        x = self.norm(x)

        # 还原维度
        if self.n_dim >= 2:
            x = x.reshape(size[1], size[2], *size[0:1])

        # 还原为 (S, B, F)
        x = x.permute(2, 0, 1)

        assert x.shape == size
        return x


class SubModule(nn.Module):
    """Class representing one of the sub_module from the mother joint module

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules,
    they all point to the same data, which is shared.
    """

    def __init__(self, main_module, indices):
        super().__init__()
        self.main_module = main_module
        self.indices = indices

    def forward(self, x):
        return self.main_module.forward(x, self.indices)


class FNOBlocks(nn.Module):
    """
    针对水文任务适配的 FNOBlocks。
    输入维度: (sequence, batch, features)
    逻辑与变量名与原版 FNOBlocks 保持 1:1 对应。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        resolution_scaling_factor=None,
        n_layers=1,
        max_n_modes=None,
        fno_block_precision="full",
        use_channel_mlp=True,
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        ada_in_features=None,
        preactivation=False,
        fno_skip="linear",
        channel_mlp_skip="soft-gating",
        complex_data=False,
        separable=False,
        factorization=None,
        rank=1.0,
        conv_module=SpectralConv,  # 使用水文适配版
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)  # 水文通常为 1

        # 验证缩放因子
        self.resolution_scaling_factor = validate_scaling_factor(
            resolution_scaling_factor, self.n_dim, n_layers
        )

        self.max_n_modes = max_n_modes
        self.fno_block_precision = fno_block_precision
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.stabilizer = stabilizer
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = fno_skip
        self.channel_mlp_skip = channel_mlp_skip
        self.complex_data = complex_data

        self.use_channel_mlp = use_channel_mlp
        self.channel_mlp_expansion = channel_mlp_expansion
        self.channel_mlp_dropout = channel_mlp_dropout
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features

        # 逻辑：实数用原激活函数，复数用 CGELU
        if self.complex_data:
            self.non_linearity = CGELU()
        else:
            self.non_linearity = non_linearity

        # 1. 初始化卷积层 (HydroSpectralConv)
        self.convs = nn.ModuleList(
            [
                conv_module(
                    self.in_channels,
                    self.out_channels,
                    self.n_modes,
                    resolution_scaling_factor=(
                        None
                        if resolution_scaling_factor is None
                        else self.resolution_scaling_factor[i]
                    ),
                    max_n_modes=max_n_modes,
                    rank=rank,
                    fixed_rank_modes=fixed_rank_modes,
                    implementation=implementation,
                    separable=separable,
                    factorization=factorization,
                    fno_block_precision=fno_block_precision,
                    decomposition_kwargs=decomposition_kwargs,
                    complex_data=complex_data,
                )
                for i in range(n_layers)
            ]
        )

        # 2. 初始化 FNO Skip Connections (Hydro版)
        if fno_skip is not None:
            self.fno_skips = nn.ModuleList(
                [
                    skip_connection(
                        self.in_channels,
                        self.out_channels,
                        skip_type=fno_skip,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.fno_skips = None

        # 3. 初始化 ChannelMLP (ChannelMLP)
        if self.use_channel_mlp:
            self.channel_mlp = nn.ModuleList(
                [
                    ChannelMLP(
                        in_channels=self.out_channels,
                        hidden_channels=round(
                            self.out_channels * channel_mlp_expansion
                        ),
                        dropout=channel_mlp_dropout,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )

            # ChannelMLP Skip
            if channel_mlp_skip is not None:
                self.channel_mlp_skips = nn.ModuleList(
                    [
                        skip_connection(
                            self.in_channels,
                            self.out_channels,
                            skip_type=channel_mlp_skip,
                            n_dim=self.n_dim,
                        )
                        for _ in range(n_layers)
                    ]
                )
            else:
                self.channel_mlp_skips = None

        # 4. 初始化归一化层 (Hydro版)
        self.n_norms = 2
        if norm is None:
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.ModuleList(
                [InstanceNorm() for _ in range(n_layers * self.n_norms)]
            )
        elif norm == "group_norm":
            # GroupNorm 逻辑：在 (S, B, F) 维度上需先转为 (B, F, S)
            # 这里简单起见，使用包装了 Permute 的类
            self.norm = nn.ModuleList(
                [
                    BatchNorm(
                        n_dim=self.n_dim, num_features=self.out_channels, num_groups=1
                    )
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        elif norm == "batch_norm":
            self.norm = nn.ModuleList(
                [
                    BatchNorm(n_dim=self.n_dim, num_features=self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        elif norm == "ada_in":
            self.norm = nn.ModuleList(
                [
                    AdaIN(ada_in_features, self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        else:
            raise ValueError(f"Got norm={norm}")

    # set_ada_in_embeddings 逻辑不变
    def set_ada_in_embeddings(self, *embeddings):
        if self.norm is not None:
            if len(embeddings) == 1:
                for norm in self.norm:
                    if hasattr(norm, "set_embedding"):
                        norm.set_embedding(embeddings[0])
            else:
                for norm, embedding in zip(self.norm, embeddings):
                    if hasattr(norm, "set_embedding"):
                        norm.set_embedding(embedding)

    # forward 逻辑与原版 1:1 保持一致
    def forward(self, x, index=0, output_shape=None):
        if self.preactivation:
            return self.forward_with_preactivation(x, index, output_shape)
        else:
            return self.forward_with_postactivation(x, index, output_shape)

    def forward_with_postactivation(self, x, index=0, output_shape=None):
        # 1. Skip Connection 预处理
        if self.fno_skips is not None:
            x_skip_fno = self.fno_skips[index](x)
            x_skip_fno = self.convs[index].transform(
                x_skip_fno, output_shape=output_shape
            )

        if self.use_channel_mlp and self.channel_mlp_skips is not None:
            x_skip_channel_mlp = self.channel_mlp_skips[index](x)
            x_skip_channel_mlp = self.convs[index].transform(
                x_skip_channel_mlp, output_shape=output_shape
            )

        # 2. 稳定器
        if self.stabilizer == "tanh":
            x = torch.tanh(x)

        # 3. 核心频谱卷积
        x_fno = self.convs[index](x, output_shape=output_shape)

        # 4. 归一化 1
        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](x_fno)

        # 5. 合并 FNO 残差
        x = x_fno + x_skip_fno if self.fno_skips is not None else x_fno

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        # 6. Channel MLP 层
        if self.use_channel_mlp:
            if self.channel_mlp_skips is not None:
                x = self.channel_mlp[index](x) + x_skip_channel_mlp
            else:
                x = self.channel_mlp[index](x)

        # 7. 归一化 2
        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        return x

    # forward_with_preactivation 逻辑同样 1:1 保持不变
    def forward_with_preactivation(self, x, index=0, output_shape=None):
        x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index](x)

        if self.fno_skips is not None:
            x_skip_fno = self.fno_skips[index](x)
            x_skip_fno = self.convs[index].transform(
                x_skip_fno, output_shape=output_shape
            )

        if self.use_channel_mlp and self.channel_mlp_skips is not None:
            x_skip_channel_mlp = self.channel_mlp_skips[index](x)
            x_skip_channel_mlp = self.convs[index].transform(
                x_skip_channel_mlp, output_shape=output_shape
            )

        if self.stabilizer == "tanh":
            x = torch.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)
        x = x_fno + x_skip_fno if self.fno_skips is not None else x_fno

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        if self.use_channel_mlp:
            if self.channel_mlp_skips is not None:
                x = self.channel_mlp[index](x) + x_skip_channel_mlp
            else:
                x = self.channel_mlp[index](x)

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        for i in range(self.n_layers):
            self.convs[i].n_modes = n_modes
        self._n_modes = n_modes

    def __getitem__(self, indices):
        return SubModule(self, indices)


class FNO(nn.Module):
    """
    针对水文任务适配的 Fourier Neural Operator (FNO)。
    假定输入 x 已经包含了所有必要的特征（如气象驱动 + 提前处理好的位置编码）。

    输入形状: (sequence, batch, in_channels)
    输出形状: (sequence, batch, out_channels)
    """

    def __init__(
        self,
        n_modes: Tuple[int],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        lifting_channel_ratio: float = 2,
        projection_channel_ratio: float = 2,
        non_linearity=F.gelu,
        norm: str = None,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: str = "soft-gating",
        fno_skip: str = "linear",
        resolution_scaling_factor: Union[float, List[float]] = None,
        domain_padding: float = None,
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = None,
        separable: bool = False,
        preactivation: bool = False,
        ada_in_features: int = None,
    ):
        super().__init__()
        if decomposition_kwargs is None:
            decomposition_kwargs = {}

        self.n_dim = len(n_modes)
        self._n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.preactivation = preactivation

        # 1. Lifting 层
        # 直接使用传入的 in_channels，不再额外增加位置维度
        self.lifting = ChannelMLP(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=int(lifting_channel_ratio * self.hidden_channels),
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

        # 2. Domain Padding
        if domain_padding is not None and domain_padding > 0:
            self.domain_padding = DomainPadding(
                domain_padding, resolution_scaling_factor
            )
        else:
            self.domain_padding = None

        # 3. FNO Blocks
        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self._n_modes,
            resolution_scaling_factor=resolution_scaling_factor,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            ada_in_features=ada_in_features,
            preactivation=preactivation,
            fno_skip=fno_skip,
            channel_mlp_skip=channel_mlp_skip,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            n_layers=n_layers,
        )

        # 4. Projection 层
        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=int(projection_channel_ratio * self.hidden_channels),
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

    def forward(self, x, output_shape=None, **kwargs):
        """
        x 形状要求: (sequence, batch, channels)
        """
        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        # 直接进入 Lifting 层
        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes
