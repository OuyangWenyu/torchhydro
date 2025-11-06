# torchhydro/models/mts_lstm.py
import warnings
import os
from typing import List, Tuple, Union, Optional, Dict, Literal, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class MTSLSTM(nn.Module):
    """Multi-Temporal-Scale LSTM (MTS-LSTM).

    This model processes multi-frequency time-series data (hour/day/week) by
    aggregating high-frequency (hourly) inputs into lower-frequency branches.
    It supports per-feature down-aggregation, optional state transfer between
    frequency branches, and loading pretrained weights for the daily branch.

    Example usage:
        # Unified hourly input (recommended)
        model = MTSLSTM(
            hidden_sizes=[64, 64, 64],
            output_size=1,
            feature_buckets=[2, 2, 1, 0, ...],
            frequency_factors=[7, 24],   # week->day ×7, day->hour ×24
            seq_lengths=[T_week, T_day, T_hour],
            slice_transfer=True,
        )
        y = model(x_hour)  # x_hour: (T_hour, B, D)

        # Legacy: explicitly provide each frequency branch
        y = model(x_week, x_day, x_hour)
    """

    def __init__(
        self,
        input_sizes: Union[int, List[int], None] = None,
        hidden_sizes: Union[int, List[int]] = 64,
        output_size: int = 1,
        shared_mtslstm: bool = False,
        transfer: Union[
            None, str, Dict[str, Optional[Literal["identity", "linear"]]]
        ] = "linear",
        dropout: float = 0.0,
        return_all: bool = False,
        add_freq_one_hot_if_shared: bool = True,
        auto_build_lowfreq: bool = False,
        build_factor: int = 7,
        agg_reduce: Literal["mean", "sum"] = "mean",
        per_feature_aggs: Optional[List[Literal["mean", "sum"]]] = None,
        truncate_incomplete: bool = True,
        slice_transfer: bool = True,
        slice_use_ceil: bool = True,
        seq_lengths: Optional[List[int]] = None,
        frequency_factors: Optional[List[int]] = None,
        feature_buckets: Optional[List[int]] = None,
        per_feature_aggs_map: Optional[List[Literal["mean", "sum"]]] = None,
        down_aggregate_all_to_each_branch: bool = True,
        pretrained_day_path: Optional[str] = None,
        pretrained_lstm_prefix: Optional[str] = None,
        pretrained_head_prefix: Optional[str] = None,
    ):
        """Initializes an MTSLSTM model.

        Args:
            input_sizes: Input feature dimension(s). Can be:
                * int: shared across all frequency branches
                * list: per-frequency input sizes
                * None: inferred from `feature_buckets`
            hidden_sizes: Hidden dimension(s) for each LSTM branch.
            output_size: Output dimension per timestep.
            shared_mtslstm: If True, all frequency branches share one LSTM.
            transfer: Hidden state transfer mode between frequencies.
                * None: no transfer
                * "identity": copy states directly (same dim required)
                * "linear": learn linear projection between dims
            dropout: Dropout probability applied before heads.
            return_all: If True, return all branch outputs (dict f0,f1,...).
                If False, return only the highest-frequency output.
            add_freq_one_hot_if_shared: If True and `shared_mtslstm=True`,
                append frequency one-hot encoding to inputs.
            auto_build_lowfreq: Legacy 2-frequency path (high->low).
            build_factor: Aggregation factor for auto low-frequency.
            agg_reduce: Aggregation method for downsampling ("mean" or "sum").
            per_feature_aggs: Optional list of per-feature aggregation methods.
            truncate_incomplete: Whether to drop remainder timesteps when
                aggregating (vs. zero-padding).
            slice_transfer: If True, transfer LSTM states at slice boundaries
                computed by seq_lengths × frequency_factors.
            slice_use_ceil: If True, use ceil for slice length calculation.
            seq_lengths: Per-frequency sequence lengths [low, ..., high].
            frequency_factors: Multipliers between adjacent frequencies.
                Example: [7,24] means week->day ×7, day->hour ×24.
            feature_buckets: Per-feature frequency assignment (len = D).
                0 = lowest (week), nf-1 = highest (hour).
            per_feature_aggs_map: Per-feature aggregation method ("mean"/"sum").
            down_aggregate_all_to_each_branch: If True, branch f includes all
                features with bucket >= f (down-aggregate); else only == f.
            pretrained_day_path: Optional path to pretrained checkpoint. If set,
                loads weights for the daily (f1) LSTM and head.

        Raises:
            AssertionError: If configuration is inconsistent.
        """
        super().__init__()

        self.pretrained_day_path = pretrained_day_path

        # -------- Frequency count (nf) --------
        if feature_buckets is not None:
            # nf = max bucket index + 1
            assert len(feature_buckets) > 0, "feature_buckets cannot be empty"
            self.nf = max(feature_buckets) + 1
        else:
            # Infer nf from input_sizes if feature_buckets not given
            if isinstance(input_sizes, int) or input_sizes is None:
                self.nf = 2 if auto_build_lowfreq else 2
            else:
                assert len(input_sizes) >= 2, "At least 2 frequencies required"
                self.nf = len(input_sizes)

        # -------- Input dimension config --------
        self.feature_buckets = (
            list(feature_buckets) if feature_buckets is not None else None
        )
        self.per_feature_aggs_map = (
            list(per_feature_aggs_map) if per_feature_aggs_map is not None else None
        )
        self.down_agg_all = down_aggregate_all_to_each_branch
        self.auto_build_lowfreq = auto_build_lowfreq

        if self.feature_buckets is not None:
            # Compute input sizes per branch (down-aggregate rule)
            D = len(self.feature_buckets)
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
            # Legacy: use input_sizes directly
            if isinstance(input_sizes, int):
                base_input_sizes = [input_sizes] * self.nf
            else:
                base_input_sizes = list(input_sizes)

        assert len(base_input_sizes) == self.nf, "input_sizes mismatch with nf"
        self.base_input_sizes = base_input_sizes
        print(self.base_input_sizes)
        # -------- Hidden layers --------
        if isinstance(hidden_sizes, int):
            self.hidden_sizes = [hidden_sizes] * self.nf
        else:
            assert len(hidden_sizes) == self.nf, "hidden_sizes length mismatch"
            self.hidden_sizes = list(hidden_sizes)
        print(self.hidden_sizes)
        self.output_size = output_size
        print(self.output_size)
        self.shared = shared_mtslstm
        self.return_all_default = return_all
        self.add_freq1hot = add_freq_one_hot_if_shared and self.shared

        # -------- Transfer configuration --------
        if transfer is None or isinstance(transfer, str):
            transfer = {"h": transfer, "c": transfer}
        self.transfer_mode: Dict[str, Optional[str]] = {
            "h": transfer.get("h", None),
            "c": transfer.get("c", None),
        }
        for k in ("h", "c"):
            assert self.transfer_mode[k] in (
                None,
                "identity",
                "linear",
            ), "transfer must be None/'identity'/'linear'"

        # -------- Aggregation / factors / slicing --------
        assert build_factor >= 2, "build_factor must be >=2"
        self.build_factor = int(build_factor)
        self.agg_reduce = agg_reduce
        self.per_feature_aggs = per_feature_aggs
        self.truncate_incomplete = truncate_incomplete

        self.slice_transfer = slice_transfer
        self.slice_use_ceil = slice_use_ceil

        # seq_lengths and frequency_factors setup
        self.seq_lengths = list(seq_lengths) if (seq_lengths is not None) else None
        if self.seq_lengths is not None:
            assert len(self.seq_lengths) == self.nf, "seq_lengths length must match nf"

        if frequency_factors is not None:
            assert (
                len(frequency_factors) == self.nf - 1
            ), "frequency_factors length must be nf-1"
            self.frequency_factors = list(map(int, frequency_factors))
        elif self.nf == 2 and auto_build_lowfreq:
            self.frequency_factors = [int(self.build_factor)]
        else:
            self.frequency_factors = None

        # Pre-compute slice positions if seq_lengths and frequency_factors are provided
        self.slice_timesteps: Optional[List[int]] = None
        if self.seq_lengths is not None and self.frequency_factors is not None:
            self.slice_timesteps = []
            for i in range(self.nf - 1):
                fac = int(self.frequency_factors[i])
                next_len = int(self.seq_lengths[i + 1])
                st = int(next_len / fac)  # floor
                self.slice_timesteps.append(max(0, st))
        self._warned_slice_fallback = False

        # -------- Effective input sizes --------
        eff_input_sizes = self.base_input_sizes[:]
        if self.add_freq1hot:
            eff_input_sizes = [d + self.nf for d in eff_input_sizes]

        if self.shared and len(set(eff_input_sizes)) != 1:
            raise ValueError("shared_mtslstm=True requires equal input sizes.")

        """self.input_linears = nn.ModuleList([
            nn.Linear(eff_input_sizes[i], self.hidden_sizes[i]) for i in range(self.nf)
        ])"""
        self.linear1 = nn.ModuleList(
            [nn.Linear(eff_input_sizes[i], 41) for i in range(self.nf)]
        )
        self.linear2 = nn.ModuleList([nn.Linear(41, 128) for i in range(self.nf)])

        # -------- Build LSTM and heads --------
        """self.lstms = nn.ModuleList()
        if self.shared:
            self.lstms.append(nn.LSTM(eff_input_sizes[0], self.hidden_sizes[0]))
        else:
            for i in range(self.nf):
                self.lstms.append(nn.LSTM(eff_input_sizes[i], self.hidden_sizes[i]))"""

        self.lstms = nn.ModuleList()
        if self.shared:
            self.lstms.append(nn.LSTM(self.hidden_sizes[0], self.hidden_sizes[0]))
        else:
            for i in range(self.nf):
                self.lstms.append(nn.LSTM(self.hidden_sizes[i], self.hidden_sizes[i]))

        self.heads = nn.ModuleList()
        if self.shared:
            self.heads.append(nn.Linear(self.hidden_sizes[0], self.output_size))
        else:
            for i in range(self.nf):
                self.heads.append(nn.Linear(self.hidden_sizes[i], self.output_size))

        # -------- Transfer projection layers --------
        self.transfer_h = nn.ModuleList()
        self.transfer_c = nn.ModuleList()
        for i in range(self.nf - 1):
            hs_i = self.hidden_sizes[i]
            hs_j = self.hidden_sizes[i + 1]
            if self.transfer_mode["h"] == "linear":
                self.transfer_h.append(nn.Linear(hs_i, hs_j))
            elif self.transfer_mode["h"] == "identity":
                assert hs_i == hs_j, "identity requires same hidden size"
                self.transfer_h.append(nn.Identity())
            else:
                self.transfer_h.append(None)

            if self.transfer_mode["c"] == "linear":
                self.transfer_c.append(nn.Linear(hs_i, hs_j))
            elif self.transfer_mode["c"] == "identity":
                assert hs_i == hs_j, "identity requires same hidden size"
                self.transfer_c.append(nn.Identity())
            else:
                self.transfer_c.append(None)

        self.dropout = nn.Dropout(p=dropout)

        # -------- Unified hourly aggregation flag --------
        self.use_hourly_unified = self.feature_buckets is not None

        # -------- Pretrained weights for daily branch --------
        if self.pretrained_day_path is not None:
            if not os.path.isfile(self.pretrained_day_path):
                warnings.warn(
                    f"[MTSLSTM] Pretrained file not found: {self.pretrained_day_path}"
                )
            elif self.shared:
                warnings.warn(
                    "[MTSLSTM] shared_mtslstm=True: skip daily-only pretrained load."
                )
            elif self.nf < 2:
                warnings.warn("[MTSLSTM] nf<2: no daily branch, skip pretrained load.")
            else:
                try:
                    state = torch.load(self.pretrained_day_path, map_location="cpu")
                    if isinstance(state, dict):
                        if "state_dict" in state:
                            state = state["state_dict"]
                        elif "model" in state:
                            state = state["model"]

                    day_lstm = self.lstms[1]
                    day_head = self.heads[1]
                    lstm_state = day_lstm.state_dict()
                    head_state = day_head.state_dict()

                    matched = skipped = shape_mismatch = 0

                    def try_load(
                        prefix: str, target_state: Dict[str, torch.Tensor]
                    ) -> None:
                        nonlocal matched, skipped, shape_mismatch
                        for k_pre, v in state.items():
                            if not k_pre.startswith(prefix):
                                continue
                            k = k_pre[len(prefix) :]
                            if k in target_state:
                                if target_state[k].shape == v.shape:
                                    target_state[k].copy_(v)
                                    matched += 1
                                else:
                                    shape_mismatch += 1
                            else:
                                skipped += 1

                    try_load(pretrained_lstm_prefix, lstm_state)
                    try_load(pretrained_head_prefix, head_state)

                    # -------- Load pretrained linearIn into linear2[1] --------
                    if "linearIn.weight" in state and "linearIn.bias" in state:
                        linear_weight = state["linearIn.weight"]
                        linear_bias = state["linearIn.bias"]

                        target_linear = self.linear2[1]  # 你想加载的线性层
                        if (
                            target_linear.weight.shape == linear_weight.shape
                            and target_linear.bias.shape == linear_bias.shape
                        ):
                            target_linear.weight.data.copy_(linear_weight)
                            target_linear.bias.data.copy_(linear_bias)
                            print(
                                f"[MTSLSTM] Pretrained linearIn loaded into linear2[1]"
                            )
                        else:
                            warnings.warn(
                                f"[MTSLSTM] linearIn shape mismatch: pretrained {linear_weight.shape}, "
                                f"current {target_linear.weight.shape}"
                            )
                    else:
                        warnings.warn(
                            "[MTSLSTM] linearIn keys not found in pretrained state"
                        )

                    if matched == 0:
                        # Try raw state_dict without prefix
                        for k, v in state.items():
                            if k in lstm_state and lstm_state[k].shape == v.shape:
                                lstm_state[k].copy_(v)
                                matched += 1
                            elif k in head_state and head_state[k].shape == v.shape:
                                head_state[k].copy_(v)
                                matched += 1
                            else:
                                shape_mismatch += 1
                    print("=== Pretrained keys ===")
                    for k in list(state.keys())[:10]:
                        print(k)

                    print("\n=== Current LSTM keys ===")
                    for k in list(day_lstm.state_dict().keys())[:10]:
                        print(k)

                    print("\n=== Current Head keys ===")
                    for k in list(day_head.state_dict().keys())[:10]:
                        print(k)

                    print("Head.bias pretrained:", state["linearOut.bias"].shape)
                    print("Head.bias current:", day_head.state_dict()["bias"].shape)

                    day_lstm.load_state_dict(lstm_state)
                    day_head.load_state_dict(head_state)
                    print(
                        f"[MTSLSTM] Daily pretrained loaded: matched={matched}, "
                        f"shape_mismatch={shape_mismatch}, skipped={skipped}"
                    )
                except Exception as e:
                    warnings.warn(f"[MTSLSTM] Failed to load daily pretrained: {e}")

    def _append_one_hot(self, x: torch.Tensor, freq_idx: int) -> torch.Tensor:
        """Appends a one-hot frequency indicator to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (T, B, D).
            freq_idx (int): Frequency index to mark as 1 in the one-hot vector.

        Returns:
            torch.Tensor: Tensor of shape (T, B, D + nf), where `nf` is the
            number of frequency branches. The appended one-hot encodes the
            branch identity.
        """
        T, B, _ = x.shape
        oh = x.new_zeros((T, B, self.nf))
        oh[:, :, freq_idx] = 1
        return torch.cat([x, oh], dim=-1)

    def _run_lstm(
        self,
        x: torch.Tensor,
        lstm: nn.LSTM,
        head: nn.Linear,
        h0: Optional[torch.Tensor],
        c0: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs an LSTM followed by a linear head with optional initial states.

        Args:
            x (torch.Tensor): Input tensor of shape (T, B, D).
            lstm (nn.LSTM): LSTM module for this branch.
            head (nn.Linear): Linear output layer.
            h0 (torch.Tensor): Optional initial hidden state (1, B, H).
            c0 (torch.Tensor): Optional initial cell state (1, B, H).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - y (torch.Tensor): Output sequence (T, B, O).
                - h_n (torch.Tensor): Final hidden state (1, B, H).
                - c_n (torch.Tensor): Final cell state (1, B, H).
        """
        out, (h_n, c_n) = (
            lstm(x, (h0, c0)) if (h0 is not None and c0 is not None) else lstm(x)
        )
        y = head(self.dropout(out))  # Project to output size
        return y, h_n, c_n

    def _aggregate_lowfreq(
        self,
        x_high: torch.Tensor,
        factor: int,
        agg_reduce: Literal["mean", "sum"],
        per_feature_aggs: Optional[List[Literal["mean", "sum"]]],
        truncate_incomplete: bool,
    ) -> torch.Tensor:
        """Aggregates high-frequency input into lower-frequency sequences.

        Args:
            x_high (torch.Tensor): Input tensor (T, B, D) at high frequency.
            factor (int): Aggregation factor (e.g., 24 for daily from hourly).
            agg_reduce (str): Default aggregation method, "mean" or "sum".
            per_feature_aggs (List[str] | None):
                Optional per-feature aggregation strategies ("mean"/"sum").
            truncate_incomplete (bool): If True, drop incomplete groups;
                if False, pad to make groups complete.

        Returns:
            torch.Tensor: Aggregated tensor of shape (T_low, B, D),
            where T_low = floor(T / factor) if truncate_incomplete,
            else ceil(T / factor).

        Raises:
            ValueError: If `agg_reduce` is not "mean" or "sum".
            AssertionError: If `per_feature_aggs` length mismatches feature dim.
        """
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
                raise ValueError("agg_reduce must be 'mean' or 'sum'")

        assert len(per_feature_aggs) == D, "per_feature_aggs length must match D"
        mean_agg = groups.mean(dim=1)
        sum_agg = groups.sum(dim=1)
        mask_sum = x_high.new_tensor(
            [1.0 if a == "sum" else 0.0 for a in per_feature_aggs]
        ).view(1, 1, D)
        mask_mean = 1.0 - mask_sum
        return mean_agg * mask_mean + sum_agg * mask_sum

    def _multi_factor_to_high(self, f: int) -> int:
        """Computes cumulative factor from branch f to the highest frequency.

        Args:
            f (int): Branch index (0 = lowest frequency).

        Returns:
            int: Product of frequency factors from branch f to highest branch.

        Raises:
            RuntimeError: If `frequency_factors` is not defined.
        """
        if self.frequency_factors is None:
            raise RuntimeError("frequency_factors must be provided.")
        fac = 1
        for k in range(f, self.nf - 1):
            fac *= int(self.frequency_factors[k])
        return fac

    def _build_from_hourly(self, x_hour: torch.Tensor) -> List[torch.Tensor]:
        """Builds multi-frequency inputs from raw hourly features.

        Each branch selects features based on bucket assignment and aggregates
        them down to its frequency using frequency factors.

        Args:
            x_hour (torch.Tensor): Hourly input tensor (T_h, B, D).
                - T_h: number of hourly timesteps
                - B: batch size
                - D: feature dimension (must match `feature_buckets`)

        Returns:
            List[torch.Tensor]: List of tensors, one per branch,
            with shapes (T_f, B, D_f).

        Raises:
            AssertionError: If feature dimension mismatches `feature_buckets`.
        """
        assert self.feature_buckets is not None, "feature_buckets must be provided"
        T_h, B, D = x_hour.shape
        assert D == len(self.feature_buckets), "Mismatch between features and buckets"

        xs = []
        for f in range(self.nf):
            if self.down_agg_all:
                cols = [i for i in range(D) if self.feature_buckets[i] >= f]
            else:
                cols = [i for i in range(D) if self.feature_buckets[i] == f]

            if len(cols) == 0:
                # Insert placeholder if branch has no features
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
        """Computes the slice length for a low-frequency branch.

        This method determines how many timesteps from the low-frequency input
        should be aligned with the high-frequency branch during state transfer.

        Priority:
            1. If `slice_timesteps` is precomputed, return the clamped value.
            2. Otherwise, fall back to heuristic estimation using ceil/floor.

        Args:
            i (int): Index of the low-frequency branch.
            T_low (int): Sequence length of the low-frequency input.
            T_high (int): Sequence length of the high-frequency input.

        Returns:
            int: Number of timesteps to slice from the low-frequency input,
            clamped between [0, T_low].

        Raises:
            RuntimeWarning: If `seq_lengths` and `frequency_factors` are not
            provided, a warning is issued since the fallback may not perfectly
            match NeuralHydrology's fixed slicing definition.
        """
        if self.slice_timesteps is not None:
            return max(0, min(self.slice_timesteps[i], T_low))
        if (not self._warned_slice_fallback) and self.slice_transfer:
            warnings.warn(
                "[MTSLSTM] 未提供 seq_lengths/frequency_factors，切片位置采用启发式估计（ceil/floor），"
                "与 NeuralHydrology 的固定切片定义不完全一致。建议提供这两个超参以完全对齐。",
                RuntimeWarning,
            )
            self._warned_slice_fallback = True
        factor = (
            self.build_factor
            if (
                self.nf == 2
                and self.feature_buckets is None
                and self.auto_build_lowfreq
            )
            else max(int(round(T_high / max(1, T_low))), 1)
        )
        if self.slice_use_ceil:
            slice_len_low = int((T_high + factor - 1) // factor)
        else:
            slice_len_low = int(T_high // factor)
        return max(0, min(slice_len_low, T_low))

    def forward(
        self, *xs: torch.Tensor, return_all: Optional[bool] = None, **kwargs: Any
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass of the Multi-Time-Scale LSTM (MTSLSTM).

        This method supports three types of input pipelines:

        1. New unified hourly input (recommended):
            - Pass a single hourly tensor (T, B, D).
            - Requires `feature_buckets` set in the constructor.
            - Internally calls `_build_from_hourly()` to build multi-scale inputs.

        2. Legacy path A (two-frequency auto build):
            - Pass a single high-frequency tensor (daily).
            - Requires `auto_build_lowfreq=True` and `nf=2`.
            - Automatically constructs the low-frequency branch.

        3. Legacy path B (manual multi-frequency input):
            - Pass a tuple of tensors: (x_f0, x_f1, ..., x_f{nf-1}).

        Args:
            *xs (torch.Tensor): Input tensors. Can be:
                - One hourly tensor of shape (T, B, D).
                - One daily tensor (legacy auto-build).
                - A tuple of nf tensors, each shaped (T_f, B, D_f).
            return_all (Optional[bool], default=None): Whether to return outputs
                from all frequency branches. If None, uses the class default.
            **kwargs: Additional unused keyword arguments.

        Returns:
            Dict[str, torch.Tensor] | torch.Tensor:
                - If `return_all=True`: Dictionary mapping branch names to outputs,
                  e.g. {"f0": y_low, "f1": y_mid, "f2": y_high}.
                - If `return_all=False`: Only returns the highest-frequency output
                  tensor of shape (T_high, B, output_size).

        Raises:
            AssertionError: If the number of provided inputs does not match `nf`,
            or if input feature dimensions do not match the expected configuration.
        """
        if return_all is None:
            return_all = self.return_all_default

        # 新路径：单一小时输入 + feature_buckets
        if len(xs) == 1 and self.use_hourly_unified:
            xs = tuple(self._build_from_hourly(xs[0]))

        # 旧路径：单一高频输入 + 自动构造低频（仅两频）
        elif (
            len(xs) == 1
            and self.auto_build_lowfreq
            and self.nf == 2
            and self.feature_buckets is None
        ):
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
            assert (
                x.shape[-1] == exp_d
            ), f"第 {i} 个输入特征维 {x.shape[-1]} 与期望 {exp_d} 不一致"

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
            # x_i = self.input_linears[i](x_i)
            x_i = self.linear1[i](x_i)
            x_i = F.relu(x_i)
            x_i = self.linear2[i](x_i)
            x_i = F.relu(x_i)
            if self.add_freq1hot:
                x_i = self._append_one_hot(x_i, i)

            lstm_i = lstm_shared if self.shared else self.lstms[i]
            head_i = head_shared if self.shared else self.heads[i]

            if (i < self.nf - 1) and self.slice_transfer:
                T_low = x_i.shape[0]
                T_high = xs[i + 1].shape[0]
                slice_len_low = self._get_slice_len_low(i, T_low=T_low, T_high=T_high)

                if slice_len_low == 0:
                    y_all, h_all, c_all = self._run_lstm(
                        x_i, lstm_i, head_i, h_transfer, c_transfer
                    )
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
                    x_part1 = (
                        x_i[:-slice_len_low]
                        if slice_len_low < x_i.shape[0]
                        else x_i[:0]
                    )
                    if x_part1.shape[0] > 0:
                        y1, h1, c1 = self._run_lstm(
                            x_part1, lstm_i, head_i, h_transfer, c_transfer
                        )
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
                y_i, h_i, c_i = self._run_lstm(
                    x_i, lstm_i, head_i, h_transfer, c_transfer
                )
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
