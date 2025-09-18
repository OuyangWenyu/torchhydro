# torchhydro/models/mts_lstm.py
import warnings
from typing import List, Union, Optional, Dict, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


class MTSLSTM(nn.Module):
    """
    多时间尺度 LSTM（不做上采样）：从“小时”输入中按配置对不同频率分支做聚合（小时→天/周）。

    用法1（推荐，多频聚合）：
        model = MTSLSTM(
            input_sizes=...  # 可给一个 int（若 shared），或留空让 feature_buckets 推导
            hidden_sizes=[H_low, H_mid, H_high],
            output_size=1,
            feature_buckets=[2,2,1,0,...],          # D 维，每个特征的目标/原生频率：0=低(周)、1=中(日)、2=高(时)
            frequency_factors=[7, 24],               # 低->中×7，中->高×24
            seq_lengths=[T_low, T_mid, T_high],      # NH 风格切片用
            per_feature_aggs_map=[...],              # 长度 D，"mean"/"sum"（可省）
            down_aggregate_all_to_each_branch=True,  # 分支 f 包含 bucket>=f 的特征并下采样聚合
            ...
        )
        # x_hour: (T_hour, B, D)
        y = model(x_hour)  # 传一个张量即可；内部自动拆分/聚合到各分支

    用法2（兼容旧版）：
        # 你自己准备好每个分支的输入 xs=(x_low, x_mid, x_high)
        y = model(*xs)
        # 或仅传一个日频张量，配合 auto_build_lowfreq=True 自动构造低频（仅两频）

    仅做“向下聚合”，不做上采样：
        - 低频分支不会看到更低频（bucket<f）的特征（避免上采样）
        - 高频分支不会看到更低频特征（同理）
        - 若 down_aggregate_all_to_each_branch=True，则高频特征可出现在低频分支（通过聚合）
    """

    def __init__(
        self,
        input_sizes: Union[int, List[int], None] = None,
        hidden_sizes: Union[int, List[int]] = 64,
        output_size: int = 1,
        shared_mtslstm: bool = False,
        transfer: Union[None, str, Dict[str, Optional[Literal["identity", "linear"]]]] = "linear",
        dropout: float = 0.0,
        return_all: bool = False,
        add_freq_one_hot_if_shared: bool = True,

        # ---- 自动两频（旧路径，保留） ----
        auto_build_lowfreq: bool = False,
        build_factor: int = 7,
        agg_reduce: Literal["mean", "sum"] = "mean",
        per_feature_aggs: Optional[List[Literal["mean", "sum"]]] = None,
        truncate_incomplete: bool = True,

        # ---- 切片传递 ----
        slice_transfer: bool = True,
        slice_use_ceil: bool = True,
        seq_lengths: Optional[List[int]] = None,         # NH：每个分支的窗口长度（按低→高）
        frequency_factors: Optional[List[int]] = None,   # NH：相邻频率倍数，长度 nf-1

        # ---- 新增：从“小时”统一聚合到多频 ----
        feature_buckets: Optional[List[int]] = None,     # 长度 D，每个特征的原生/目标频率：0=低、...、nf-1=高(小时)
        per_feature_aggs_map: Optional[List[Literal["mean","sum"]]] = None,  # 长度 D
        down_aggregate_all_to_each_branch: bool = True,  # 分支 f 包含 bucket>=f 的特征并聚合；bucket<f 的丢弃
    ):
        super().__init__()

        # ------- 频率个数 nf -------
        # 若给了 feature_buckets，则 nf 由其最大值+1 推断（否则由 input_sizes/list 长度推断）
        if feature_buckets is not None:
            assert len(feature_buckets) > 0, "feature_buckets 不能为空"
            self.nf = max(feature_buckets) + 1
        else:
            # 若未给 feature_buckets，则从 input_sizes 推断 nf
            if isinstance(input_sizes, int) or input_sizes is None:
                self.nf = 2 if auto_build_lowfreq else 2  # 至少两频（与旧逻辑一致）
            else:
                assert len(input_sizes) >= 2, "需要至少两个频率"
                self.nf = len(input_sizes)

        # ------- 输入维度配置 -------
        self.feature_buckets = list(feature_buckets) if feature_buckets is not None else None
        self.per_feature_aggs_map = list(per_feature_aggs_map) if per_feature_aggs_map is not None else None
        self.down_agg_all = down_aggregate_all_to_each_branch
        self.auto_build_lowfreq = auto_build_lowfreq

        # 若使用 feature_buckets，则按“只向下聚合”的规则预计算每个分支的输入维度
        if self.feature_buckets is not None:
            D = len(self.feature_buckets)
            # 分支 f 的列数：包含所有 bucket>=f 的特征（仅当 down_agg_all=True）
            if self.down_agg_all:
                base_input_sizes = [
                    sum(1 for k in range(D) if self.feature_buckets[k] >= f)
                    for f in range(self.nf)
                ]
            else:
                base_input_sizes = [
                    sum(1 for k in range(D) if self.feature_buckets[k] == f)
                    for f in range(self.nf)
                ]
        else:
            # 旧路径：直接使用传入的 input_sizes
            if isinstance(input_sizes, int):
                base_input_sizes = [input_sizes] * self.nf
            else:
                base_input_sizes = list(input_sizes)

        assert len(base_input_sizes) == self.nf, "input_sizes 与频率数不一致"
        self.base_input_sizes = base_input_sizes

        # ------- 隐层 -------
        if isinstance(hidden_sizes, int):
            self.hidden_sizes = [hidden_sizes] * self.nf
        else:
            assert len(hidden_sizes) == self.nf, "hidden_sizes 长度需与频率数一致"
            self.hidden_sizes = list(hidden_sizes)

        self.output_size = output_size
        self.shared = shared_mtslstm
        self.return_all_default = return_all
        self.add_freq1hot = (add_freq_one_hot_if_shared and self.shared)

        # ------- 迁移设置 -------
        if transfer is None or isinstance(transfer, str):
            transfer = {"h": transfer, "c": transfer}
        self.transfer_mode: Dict[str, Optional[str]] = {
            "h": transfer.get("h", None),
            "c": transfer.get("c", None),
        }
        for k in ("h", "c"):
            assert self.transfer_mode[k] in (None, "identity", "linear"), \
                "transfer 仅支持 None/'identity'/'linear'"

        # ------- 聚合/因子/切片 -------
        assert build_factor >= 2, "build_factor 必须 >=2"
        self.build_factor = int(build_factor)
        self.agg_reduce = agg_reduce
        self.per_feature_aggs = per_feature_aggs
        self.truncate_incomplete = truncate_incomplete

        self.slice_transfer = slice_transfer
        self.slice_use_ceil = slice_use_ceil

        # NH：seq_lengths / frequency_factors / slice_timestep
        self.seq_lengths = list(seq_lengths) if (seq_lengths is not None) else None
        if self.seq_lengths is not None:
            assert len(self.seq_lengths) == self.nf, "seq_lengths 长度必须等于 nf"

        if frequency_factors is not None:
            assert len(frequency_factors) == self.nf - 1, "frequency_factors 长度必须为 nf-1"
            self.frequency_factors = list(map(int, frequency_factors))
        elif self.nf == 2 and auto_build_lowfreq:
            self.frequency_factors = [int(self.build_factor)]
        else:
            self.frequency_factors = None

        self.slice_timesteps: Optional[List[int]] = None
        if self.seq_lengths is not None and self.frequency_factors is not None:
            self.slice_timesteps = []
            for i in range(self.nf - 1):
                fac = int(self.frequency_factors[i])
                next_len = int(self.seq_lengths[i + 1])
                st = int(next_len / fac)  # NH: floor
                self.slice_timesteps.append(max(0, st))
        self._warned_slice_fallback = False

        # ------- 组装 LSTM 输入维度（考虑 one-hot） -------
        eff_input_sizes = self.base_input_sizes[:]
        if self.add_freq1hot:
            eff_input_sizes = [d + self.nf for d in eff_input_sizes]

        # 若 shared=True 要求各分支 input_size 一致
        if self.shared and len(set(eff_input_sizes)) != 1:
            raise ValueError("shared_mtslstm=True 要求各分支输入维度一致。请调整 feature_buckets 或关闭 shared。")

        # ------- 构建各层 -------
        self.lstms = nn.ModuleList()
        if self.shared:
            self.lstms.append(nn.LSTM(eff_input_sizes[0], self.hidden_sizes[0]))
        else:
            for i in range(self.nf):
                self.lstms.append(nn.LSTM(eff_input_sizes[i], self.hidden_sizes[i]))

        self.heads = nn.ModuleList()
        if self.shared:
            self.heads.append(nn.Linear(self.hidden_sizes[0], self.output_size))
        else:
            for i in range(self.nf):
                self.heads.append(nn.Linear(self.hidden_sizes[i], self.output_size))

        self.transfer_h = nn.ModuleList()
        self.transfer_c = nn.ModuleList()
        for i in range(self.nf - 1):
            hs_i = self.hidden_sizes[i]
            hs_j = self.hidden_sizes[i + 1]
            if self.transfer_mode["h"] == "linear":
                self.transfer_h.append(nn.Linear(hs_i, hs_j))
            elif self.transfer_mode["h"] == "identity":
                assert hs_i == hs_j, "identity 迁移要求相同隐藏维度"
                self.transfer_h.append(nn.Identity())
            else:
                self.transfer_h.append(None)

            if self.transfer_mode["c"] == "linear":
                self.transfer_c.append(nn.Linear(hs_i, hs_j))
            elif self.transfer_mode["c"] == "identity":
                assert hs_i == hs_j, "identity 迁移要求相同隐藏维度"
                self.transfer_c.append(nn.Identity())
            else:
                self.transfer_c.append(None)

        self.dropout = nn.Dropout(p=dropout)

        # 标记：是否走“从小时统一聚合”的路径（根据 feature_buckets）
        self.use_hourly_unified = (self.feature_buckets is not None)

    # ----------------- 工具 -----------------
    def _append_one_hot(self, x: torch.Tensor, freq_idx: int) -> torch.Tensor:
        T, B, _ = x.shape
        oh = x.new_zeros((T, B, self.nf))
        oh[:, :, freq_idx] = 1
        return torch.cat([x, oh], dim=-1)

    def _run_lstm(self, x, lstm, head, h0, c0):
        out, (h_n, c_n) = lstm(x, (h0, c0)) if (h0 is not None and c0 is not None) else lstm(x)
        y = head(self.dropout(out))  # (T,B,O)
        return y, h_n, c_n

    def _aggregate_lowfreq(self, x_high, factor, agg_reduce, per_feature_aggs, truncate_incomplete):
        T, B, D = x_high.shape
        if factor <= 1:
            return x_high
        if truncate_incomplete:
            T_trim = (T // factor) * factor
            xh = x_high[:T_trim]
            groups = xh.view(T_trim // factor, factor, B, D)
        else:
            pad = (factor - (T % factor)) % factor
            if pad > 0:
                pad_tensor = x_high.new_zeros((pad, B, D))
                xh = torch.cat([x_high, pad_tensor], dim=0)
            else:
                xh = x_high
            groups = xh.view(xh.shape[0] // factor, factor, B, D)

        if per_feature_aggs is None:
            if agg_reduce == "mean":
                return groups.mean(dim=1)
            elif agg_reduce == "sum":
                return groups.sum(dim=1)
            else:
                raise ValueError("agg_reduce 仅支持 'mean' 或 'sum'")

        assert len(per_feature_aggs) == D, "per_feature_aggs 长度需等于特征维 D"
        mean_agg = groups.mean(dim=1)
        sum_agg = groups.sum(dim=1)
        mask_sum = x_high.new_tensor([1.0 if a == "sum" else 0.0 for a in per_feature_aggs]).view(1, 1, D)
        mask_mean = 1.0 - mask_sum
        return mean_agg * mask_mean + sum_agg * mask_sum

    def _multi_factor_to_high(self, f: int) -> int:
        """分支 f 到最高频（nf-1）的连乘因子（例如 周->小时 7*24）。"""
        if self.frequency_factors is None:
            raise RuntimeError("需要 frequency_factors 来执行多级聚合。")
        fac = 1
        for k in range(f, self.nf - 1):
            fac *= int(self.frequency_factors[k])
        return fac

    def _build_from_hourly(self, x_hour: torch.Tensor) -> List[torch.Tensor]:
        """
        从单一小时输入 (T_h,B,D) 生成各分支输入：
        - 分支 f 选取列 S_f：
            * 若 down_agg_all=True ：S_f = {k | bucket[k] >= f}
            * 否则 S_f = {k | bucket[k] == f}
        - 将 x_hour[:, :, S_f] 以因子 factor = ∏_{k=f}^{nf-2} frequency_factors[k] 聚合到分支时间轴
        - 不做上采样（bucket<f 的特征不会进入分支 f）
        """
        assert self.feature_buckets is not None, "需提供 feature_buckets"
        T_h, B, D = x_hour.shape
        assert D == len(self.feature_buckets), "x_hour 的特征维与 feature_buckets 长度不一致"

        xs = []
        for f in range(self.nf):
            if self.down_agg_all:
                cols = [i for i in range(D) if self.feature_buckets[i] >= f]
            else:
                cols = [i for i in range(D) if self.feature_buckets[i] == f]

            if len(cols) == 0:
                # 没有特征时放一个占位全零列，避免尺寸为 0
                x_sub = x_hour.new_zeros((T_h, B, 1))
                per_aggs = ["mean"]
            else:
                x_sub = x_hour[:, :, cols]
                per_aggs = None
                if self.per_feature_aggs_map is not None:
                    per_aggs = [self.per_feature_aggs_map[i] for i in cols]

            factor = self._multi_factor_to_high(f) if (self.nf >= 2) else 1
            x_f = self._aggregate_lowfreq(
                x_high=x_sub,
                factor=factor,
                agg_reduce=self.agg_reduce,
                per_feature_aggs=per_aggs,
                truncate_incomplete=self.truncate_incomplete,
            )
            xs.append(x_f)
        return xs

    def _get_slice_len_low(self, i: int, T_low: int, T_high: int) -> int:
        if self.slice_timesteps is not None:
            return max(0, min(self.slice_timesteps[i], T_low))
        if (not self._warned_slice_fallback) and self.slice_transfer:
            warnings.warn(
                "[MTSLSTM] 未提供 seq_lengths/frequency_factors，切片位置采用启发式估计（ceil/floor），"
                "与 NeuralHydrology 的固定切片定义不完全一致。建议提供这两个超参以完全对齐。",
                RuntimeWarning
            )
            self._warned_slice_fallback = True
        factor = self.build_factor if (self.nf == 2 and self.feature_buckets is None and self.auto_build_lowfreq) else \
                 max(int(round(T_high / max(1, T_low))), 1)
        if self.slice_use_ceil:
            slice_len_low = int((T_high + factor - 1) // factor)
        else:
            slice_len_low = int(T_high // factor)
        return max(0, min(slice_len_low, T_low))

    # ----------------- 前向 -----------------
    def forward(self, *xs: torch.Tensor, return_all: Optional[bool] = None, **kwargs):
        """
        支持三种入口：
        1) 新路径（推荐）：仅传 1 个小时张量 x_hour，且在 __init__ 传入 feature_buckets
           -> 内部调用 _build_from_hourly() 生成各分支输入
        2) 旧路径 A：仅传 1 个“日频”张量，且 auto_build_lowfreq=True（仅两频）
        3) 旧路径 B：直接传各分支张量 xs=(x_f0,...,x_f{nf-1})
        """
        if return_all is None:
            return_all = self.return_all_default

        # 新路径：单一小时输入 + feature_buckets
        if len(xs) == 1 and self.use_hourly_unified:
            xs = tuple(self._build_from_hourly(xs[0]))

        # 旧路径：单一高频输入 + 自动构造低频（仅两频）
        elif len(xs) == 1 and self.auto_build_lowfreq and self.nf == 2 and self.feature_buckets is None:
            x_high = xs[0]
            assert x_high.dim() == 3, "输入必须是 (time,batch,features)"
            x_low = self._aggregate_lowfreq(
                x_high=x_high,
                factor=self.build_factor,
                agg_reduce=self.agg_reduce,
                per_feature_aggs=self.per_feature_aggs,
                truncate_incomplete=self.truncate_incomplete,
            )
            xs = (x_low, x_high)

        # 常规校验
        assert len(xs) == self.nf, f"收到 {len(xs)} 个频率，但模型期望 {self.nf}"
        for i, x in enumerate(xs):
            assert x.dim() == 3, f"第 {i} 个输入必须是 (time,batch,features)"
            # 允许输入的特征维 == 预期维（若共享且拼 one-hot，会在内部拼接）
            exp_d = self.base_input_sizes[i]
            assert x.shape[-1] == exp_d, f"第 {i} 个输入特征维 {x.shape[-1]} 与期望 {exp_d} 不一致"

        device = xs[0].device
        B0 = xs[0].shape[1]
        H0 = self.hidden_sizes[0]
        h_transfer = torch.zeros(1, B0, H0, device=device)
        c_transfer = torch.zeros(1, B0, H0, device=device)

        outputs: Dict[str, torch.Tensor] = {}
        lstm_shared = self.lstms[0] if self.shared else None
        head_shared = self.heads[0] if self.shared else None

        for i in range(self.nf):
            x_i = xs[i]
            if self.add_freq1hot:
                x_i = self._append_one_hot(x_i, i)

            lstm_i = lstm_shared if self.shared else self.lstms[i]
            head_i = head_shared if self.shared else self.heads[i]

            if (i < self.nf - 1) and self.slice_transfer:
                T_low = x_i.shape[0]
                T_high = xs[i + 1].shape[0]
                slice_len_low = self._get_slice_len_low(i, T_low=T_low, T_high=T_high)

                if slice_len_low == 0:
                    y_all, h_all, c_all = self._run_lstm(x_i, lstm_i, head_i, h_transfer, c_transfer)
                    outputs[f"f{i}"] = y_all
                    Bn = xs[i + 1].shape[1]
                    Hn = self.hidden_sizes[i + 1]
                    h_transfer = torch.zeros(1, Bn, Hn, device=device)
                    c_transfer = torch.zeros(1, Bn, Hn, device=device)
                    if self.transfer_h[i] is not None:
                        h_transfer = self.transfer_h[i](h_all[0]).unsqueeze(0)
                    if self.transfer_c[i] is not None:
                        c_transfer = self.transfer_c[i](c_all[0]).unsqueeze(0)
                else:
                    x_part1 = x_i[:-slice_len_low] if slice_len_low < x_i.shape[0] else x_i[:0]
                    if x_part1.shape[0] > 0:
                        y1, h1, c1 = self._run_lstm(x_part1, lstm_i, head_i, h_transfer, c_transfer)
                    else:
                        y1 = x_i.new_zeros((0, x_i.shape[1], self.output_size))
                        h1, c1 = h_transfer, c_transfer

                    Bn = xs[i + 1].shape[1]
                    Hn = self.hidden_sizes[i + 1]
                    h_transfer = torch.zeros(1, Bn, Hn, device=device)
                    c_transfer = torch.zeros(1, Bn, Hn, device=device)
                    if self.transfer_h[i] is not None:
                        h_transfer = self.transfer_h[i](h1[0]).unsqueeze(0)
                    if self.transfer_c[i] is not None:
                        c_transfer = self.transfer_c[i](c1[0]).unsqueeze(0)

                    x_part2 = x_i[-slice_len_low:] if slice_len_low > 0 else x_i[:0]
                    if x_part2.shape[0] > 0:
                        y2, _, _ = self._run_lstm(x_part2, lstm_i, head_i, h1, c1)
                        y_all = torch.cat([y1, y2], dim=0)
                    else:
                        y_all = y1

                    outputs[f"f{i}"] = y_all
            else:
                y_i, h_i, c_i = self._run_lstm(x_i, lstm_i, head_i, h_transfer, c_transfer)
                outputs[f"f{i}"] = y_i
                if i < self.nf - 1:
                    Bn = xs[i + 1].shape[1]
                    Hn = self.hidden_sizes[i + 1]
                    h_transfer = torch.zeros(1, Bn, Hn, device=device)
                    c_transfer = torch.zeros(1, Bn, Hn, device=device)
                    if self.transfer_h[i] is not None:
                        h_transfer = self.transfer_h[i](h_i[0]).unsqueeze(0)
                    if self.transfer_c[i] is not None:
                        c_transfer = self.transfer_c[i](c_i[0]).unsqueeze(0)

        return outputs if return_all else outputs[f"f{self.nf - 1}"]
