# torchhydro/models/mts_lstm.py
from typing import List, Union, Optional, Dict, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


class MTSLSTM(nn.Module):
    """
    多时间尺度 LSTM，保持接口：
        output = model(*xs)
    其中 xs 是按从低频到高频的序列列表，每个张量形状 (time, batch, features)。
    若只传 1 个日频张量且 auto_build_lowfreq=True，会在 forward 内部自动构造低频序列。

    初始化参数（通过 model_hyperparam 传入）：
    -------------------------------------------------
    input_sizes: List[int] | int
        各频率输入维度；若仅给一个 int 且 auto_build_lowfreq=True，会在内部扩展为 [D_low, D_day]=[D,D]
    hidden_sizes: int | List[int]
        LSTM 隐层维度；可为统一 int 或逐频 list（len==n_freqs）
    output_size: int
        输出维度
    shared_mtslstm: bool = False
        共享权重（sMTS-LSTM）；会给输入拼接 one-hot(freq)（可关）
    transfer: None | str | Dict[str,str] = "linear"
        低->高 频率的状态迁移方式：None / "identity" / "linear"
        也可传 {"h":"linear","c":"identity"} 这种
    dropout: float = 0.0
        LSTM 输出后的 dropout
    return_all: bool = False
        True 时返回所有频率输出 dict；否则只返回最高频输出
    add_freq_one_hot_if_shared: bool = True
        shared 时是否拼接 one-hot(freq)

    # 自动构造低频相关
    auto_build_lowfreq: bool = False
        只传日频 xs 时是否在 forward 内部自动生成低频
    build_factor: int = 7
        高频到低频的聚合因子（如 7 表示 7 天聚一周）
    agg_reduce: Literal["mean","sum"] = "mean"
        若不提供 per_feature_aggs，则对所有特征统一用此聚合
    per_feature_aggs: Optional[List[Literal["mean","sum"]]] = None
        每个特征的聚合方式，长度需等于 D
    truncate_incomplete: bool = True
        T_day 不是 build_factor 整数倍时，是否截断末尾残段

    # 切片传递（slice/factor）相关
    slice_transfer: bool = True
        是否启用切片传递
    slice_use_ceil: bool = True
        切片长度 = ceil(T_high/build_factor)（否则 floor）
    """

    def __init__(
        self,
        input_sizes: Union[int, List[int]],
        hidden_sizes: Union[int, List[int]],
        output_size: int,
        shared_mtslstm: bool = False,
        transfer: Union[None, str, Dict[str, Optional[Literal["identity", "linear"]]]] = "linear",
        dropout: float = 0.0,
        return_all: bool = False,
        add_freq_one_hot_if_shared: bool = True,
        # 自动构造低频
        auto_build_lowfreq: bool = False,
        build_factor: int = 7,
        agg_reduce: Literal["mean", "sum"] = "mean",
        per_feature_aggs: Optional[List[Literal["mean", "sum"]]] = None,
        truncate_incomplete: bool = True,
        # 切片传递
        slice_transfer: bool = True,
        slice_use_ceil: bool = True,
    ):
        super().__init__()
        # 规范 input_sizes
        if isinstance(input_sizes, int):
            base_input_sizes = [input_sizes]
        else:
            base_input_sizes = list(input_sizes)

        # 如果只给了日频尺寸，但希望自动构造低频，则扩成两层 [低频, 高频(日)]
        self.auto_build_lowfreq = auto_build_lowfreq
        if self.auto_build_lowfreq and len(base_input_sizes) == 1:
            base_input_sizes = [base_input_sizes[0], base_input_sizes[0]]

        assert len(base_input_sizes) >= 2, "MTSLSTM 需要至少两个频率（或开启 auto_build_lowfreq 并只传日频尺寸）。"
        self.base_input_sizes = base_input_sizes
        self.nf = len(self.base_input_sizes)

        # 隐层尺寸
        if isinstance(hidden_sizes, int):
            self.hidden_sizes = [hidden_sizes] * self.nf
        else:
            assert len(hidden_sizes) == self.nf, "hidden_sizes 长度需与频率数一致"
            self.hidden_sizes = list(hidden_sizes)

        self.output_size = output_size
        self.shared = shared_mtslstm
        self.return_all_default = return_all
        self.add_freq1hot = (add_freq_one_hot_if_shared and self.shared)

        # 迁移选项
        if transfer is None or isinstance(transfer, str):
            transfer = {"h": transfer, "c": transfer}
        self.transfer_mode: Dict[str, Optional[str]] = {
            "h": transfer.get("h", None),
            "c": transfer.get("c", None),
        }
        for k in ("h", "c"):
            assert self.transfer_mode[k] in (None, "identity", "linear"), \
                "transfer 仅支持 None/'identity'/'linear'"

        # 自动构造低频设置
        assert build_factor >= 2, "build_factor 必须 >=2"
        self.build_factor = int(build_factor)
        self.agg_reduce = agg_reduce
        self.per_feature_aggs = per_feature_aggs
        self.truncate_incomplete = truncate_incomplete

        # 切片传递设置
        self.slice_transfer = slice_transfer
        self.slice_use_ceil = slice_use_ceil

        # 有共享时，实际输入维度需要加 one-hot
        eff_input_sizes = self.base_input_sizes[:]
        if self.add_freq1hot:
            eff_input_sizes = [d + self.nf for d in eff_input_sizes]

        # LSTMs
        self.lstms = nn.ModuleList()
        if self.shared:
            assert len(set(self.hidden_sizes)) == 1, "shared=True 要求各频率 hidden_size 相等"
            self.lstms.append(nn.LSTM(eff_input_sizes[0], self.hidden_sizes[0]))
        else:
            for i in range(self.nf):
                self.lstms.append(nn.LSTM(eff_input_sizes[i], self.hidden_sizes[i]))

        # Heads（线性）
        self.heads = nn.ModuleList()
        if self.shared:
            self.heads.append(nn.Linear(self.hidden_sizes[0], self.output_size))
        else:
            for i in range(self.nf):
                self.heads.append(nn.Linear(self.hidden_sizes[i], self.output_size))

        # 低->高 迁移层（i -> i+1）
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

    # ---------- 工具方法 ----------
    def _append_one_hot(self, x: torch.Tensor, freq_idx: int) -> torch.Tensor:
        T, B, _ = x.shape
        oh = x.new_zeros((T, B, self.nf))
        oh[:, :, freq_idx] = 1
        return torch.cat([x, oh], dim=-1)

    def _run_lstm(
        self,
        x: torch.Tensor,            # (T,B,D)
        lstm: nn.LSTM,
        head: nn.Module,
        h0: Optional[torch.Tensor], # (1,B,H) or None
        c0: Optional[torch.Tensor], # (1,B,H) or None
    ):
        out, (h_n, c_n) = lstm(x, (h0, c0)) if (h0 is not None and c0 is not None) else lstm(x)
        y = head(self.dropout(out))  # (T,B,O)
        return y, h_n, c_n

    def _aggregate_lowfreq(
        self,
        x_high: torch.Tensor,  # (T_high,B,D)
        factor: int,
        agg_reduce: str,
        per_feature_aggs: Optional[List[str]],
        truncate_incomplete: bool,
    ) -> torch.Tensor:
        """把高频聚合成低频，支持全局 mean/sum 或 per-feature 控制。"""
        T, B, D = x_high.shape
        if truncate_incomplete:
            T_trim = (T // factor) * factor
            xh = x_high[:T_trim]
            groups = xh.view(T_trim // factor, factor, B, D)  # (T_low,factor,B,D)
        else:
            # 不截断：按需在时间维补零（mean/sum 的定义会受影响）
            pad = (factor - (T % factor)) % factor
            if pad > 0:
                pad_tensor = x_high.new_zeros((pad, B, D))
                xh = torch.cat([x_high, pad_tensor], dim=0)
            else:
                xh = x_high
            groups = xh.view(xh.shape[0] // factor, factor, B, D)

        if per_feature_aggs is None:
            if agg_reduce == "mean":
                return groups.mean(dim=1)  # (T_low,B,D)
            elif agg_reduce == "sum":
                return groups.sum(dim=1)
            else:
                raise ValueError("agg_reduce 仅支持 'mean' 或 'sum'")

        # per-feature：一次性算 mean 和 sum，再按列拣选
        assert len(per_feature_aggs) == D, "per_feature_aggs 长度需等于特征维 D"
        mean_agg = groups.mean(dim=1)  # (T_low,B,D)
        sum_agg = groups.sum(dim=1)    # (T_low,B,D)
        # 构造选择掩码
        mask_sum = x_high.new_tensor([1.0 if a == "sum" else 0.0 for a in per_feature_aggs]).view(1, 1, D)
        mask_mean = 1.0 - mask_sum
        return mean_agg * mask_mean + sum_agg * mask_sum

    # ---------- 前向 ----------
    def forward(self, *xs: torch.Tensor, return_all: Optional[bool] = None, **kwargs):
        """
        xs: 形如 [x_f0, x_f1, ...]，按从低频到高频排序；每个张量 (T_i,B,D_i)。
        若只传 1 个张量且 auto_build_lowfreq=True：会把该张量视为“日频”，内部聚合生成低频序列。
        **kwargs 为兼容其它模型的接口，占位不用。
        """
        if return_all is None:
            return_all = self.return_all_default

        # 只传了日频？
        if len(xs) == 1 and self.auto_build_lowfreq:
            x_day = xs[0]
            assert x_day.dim() == 3, "输入必须是 (time,batch,features)"
            # 低频聚合
            x_low = self._aggregate_lowfreq(
                x_high=x_day,
                factor=self.build_factor,
                agg_reduce=self.agg_reduce,
                per_feature_aggs=self.per_feature_aggs,
                truncate_incomplete=self.truncate_incomplete,
            )
            xs = (x_low, x_day)  # 低频在前，高频(日)在后

        # 常规校验
        assert len(xs) == self.nf, f"收到 {len(xs)} 个频率，但模型期望 {self.nf}"
        for i, x in enumerate(xs):
            assert x.dim() == 3, f"第 {i} 个输入必须是 (time,batch,features)"
            assert x.shape[-1] == self.base_input_sizes[i], \
                f"第 {i} 个输入特征维 {x.shape[-1]} 与 input_sizes[{i}]={self.base_input_sizes[i]} 不一致"

        device = xs[0].device
        B0 = xs[0].shape[1]
        H0 = self.hidden_sizes[0]
        h_transfer = torch.zeros(1, B0, H0, device=device)
        c_transfer = torch.zeros(1, B0, H0, device=device)

        outputs: Dict[str, torch.Tensor] = {}
        lstm_shared = self.lstms[0] if self.shared else None
        head_shared = self.heads[0] if self.shared else None

        for i in range(self.nf):
            x_i = xs[i]  # (T_i,B,D_i)
            if self.add_freq1hot:
                x_i = self._append_one_hot(x_i, i)

            lstm_i = lstm_shared if self.shared else self.lstms[i]
            head_i = head_shared if self.shared else self.heads[i]

            # 不是最高频，并且启用切片传递：按“slice/factor”切
            if (i < self.nf - 1) and self.slice_transfer:
                # 根据 “下一频率长度 / 因子” 估算需要在低频末尾保留多少步
                T_high = xs[i + 1].shape[0]
                factor = self.build_factor if (len(xs) == 2 and self.auto_build_lowfreq) else max(
                    int(round(T_high / max(1, x_i.shape[0]))), 1
                )
                # 若已知严格因子，可改为：factor = 显式配置（例如多级因子列表）
                if self.slice_use_ceil:
                    slice_len_low = int((T_high + factor - 1) // factor)
                else:
                    slice_len_low = int(T_high // factor)
                slice_len_low = max(0, min(slice_len_low, x_i.shape[0]))

                if slice_len_low == 0:
                    # 没有需要切片的长度，整段跑 + 迁移
                    y_all, h_all, c_all = self._run_lstm(x_i, lstm_i, head_i, h_transfer, c_transfer)
                    outputs[f"f{i}"] = y_all
                    # 迁移到下一频率
                    Bn = xs[i + 1].shape[1]
                    Hn = self.hidden_sizes[i + 1]
                    h_transfer = torch.zeros(1, Bn, Hn, device=device)
                    c_transfer = torch.zeros(1, Bn, Hn, device=device)
                    if self.transfer_h[i] is not None:
                        h_transfer = self.transfer_h[i](h_all[0]).unsqueeze(0)
                    if self.transfer_c[i] is not None:
                        c_transfer = self.transfer_c[i](c_all[0]).unsqueeze(0)
                else:
                    # 第一段：到切片点之前
                    x_part1 = x_i[:-slice_len_low] if slice_len_low < x_i.shape[0] else x_i[:0]
                    if x_part1.shape[0] > 0:
                        y1, h1, c1 = self._run_lstm(x_part1, lstm_i, head_i, h_transfer, c_transfer)
                    else:
                        # 没有前段，则前段输出为空，状态仍用初值
                        y1 = x_i.new_zeros((0, x_i.shape[1], self.output_size))
                        h1, c1 = h_transfer, c_transfer

                    # 把前段末状态迁移到“下一频率”的初态（供下一频率使用）
                    Bn = xs[i + 1].shape[1]
                    Hn = self.hidden_sizes[i + 1]
                    h_transfer = torch.zeros(1, Bn, Hn, device=device)
                    c_transfer = torch.zeros(1, Bn, Hn, device=device)
                    if self.transfer_h[i] is not None:
                        h_transfer = self.transfer_h[i](h1[0]).unsqueeze(0)
                    if self.transfer_c[i] is not None:
                        c_transfer = self.transfer_c[i](c1[0]).unsqueeze(0)

                    # 第二段：低频的最后 slice_len_low 步（把低频自己也跑完）
                    x_part2 = x_i[-slice_len_low:] if slice_len_low > 0 else x_i[:0]
                    if x_part2.shape[0] > 0:
                        y2, _, _ = self._run_lstm(x_part2, lstm_i, head_i, h1, c1)
                        y_all = torch.cat([y1, y2], dim=0)
                    else:
                        y_all = y1

                    outputs[f"f{i}"] = y_all

            else:
                # 最高频（或未启用切片传递）：整段跑
                y_i, h_i, c_i = self._run_lstm(x_i, lstm_i, head_i, h_transfer, c_transfer)
                outputs[f"f{i}"] = y_i
                # 为下一频率准备（若还有下一频率且未切片，下传最终态）
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
