#!/usr/bin/env python3
"""
Model comparison plotting script
Compare multiple model result folders (e.g. gnn_only, lstm_only, lstm_gnn) on Songliao test set (normal scenario)

Features:
- Flexible epoch identifier (e.g. 30, 60, best_model, best_model.pth)
- Automatic NetCDF file discovery (glob fallback)
- Flood event detection using Songliao labels (via flood_utils) if available, fallback simple detector
- Unit conversion (mm/3h -> m^3/s) using basin area (via flood_utils)
- Epoch-specific output folder: <output_dir>/epoch{epoch}/model_comparison_plots
- Metrics (NSE per model per event) -> CSV
- Summary statistics (NSE only) per model -> TXT
- Figure filenames include epoch string; rainfall-runoff style (rain bars + multi-model hydrographs)

Usage example:
python plot_model_comparison.py \
  --model_dirs /home/lizilin/code/torchhydro/results/3d_gnn_only,/home/lizilin/code/torchhydro/results/3d_gnn_experiment_lstm_only,/home/lizilin/code/torchhydro/results/3d_gnn_experiment_lstm_gnn \
  --model_names GNN,LSTM,"LSTM+GNN" \
  --epoch best_model.pth

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Matplotlib settings
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
# 增大基础字号（原 10 -> 11）
plt.rcParams['font.size'] = 12
# 定义统一字号常量 (再次增大 1)
BASE_FONT = 12       # 原 11
LABEL_FONT = 14      # 原 13
TITLE_FONT = 17      # 原 16
ANNOT_FONT = 12      # 原 11

# Try importing flood utilities
USE_FLOOD_LABELS = True
try:
    from flood_utils import (
        load_netcdf_data,
        load_basin_attributes,
        convert_flow_units_batch,
        find_flood_events_from_labels,
        load_songliao_timeseries_data,  # rainfall loader
    )
except Exception as e:
    print(f"Warning: flood_utils import failed ({e}), falling back to internal loaders")
    USE_FLOOD_LABELS = False
    import xarray as xr
    def load_netcdf_data(file_path):
        try:
            ds = xr.open_dataset(file_path)
            data_dict = {var: ds[var].values for var in ds.data_vars}
            for coord in ['time','basin']:
                if coord in ds.coords:
                    data_dict[coord] = ds.coords[coord].values
            return data_dict
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return None
    def load_basin_attributes(basin_id, songliao_data_path=None):
        return None  # no attributes
    def convert_flow_units_batch(data, basin_areas, time_unit="3h"):
        return data, 'mm/3h'
    def load_songliao_timeseries_data(*args, **kwargs):
        return None

# Simple fallback flood event detector (percentile based)
def simple_find_flood_events(flow_data, time_array=None, min_duration=6, percentile_threshold=70):
    if flow_data.ndim == 1:
        flow_data = flow_data.reshape(1,-1)
    n_basins, n_time = flow_data.shape
    events = []
    for b in range(n_basins):
        series = flow_data[b]
        valid = series[~np.isnan(series)]
        if len(valid) < 10:
            continue
        thr = np.percentile(valid, percentile_threshold)
        above = series >= thr
        start = None
        for i, flag in enumerate(above):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                if i - start >= min_duration:
                    seg = series[start:i]
                    peak_idx_rel = np.argmax(seg)
                    peak_idx = start + peak_idx_rel
                    events.append({
                        'basin': b,
                        'start': start,
                        'end': i,
                        'duration': i-start,
                        'peak_idx': peak_idx,
                        'peak_value': series[peak_idx],
                        'event_name': f'Event_{b}_{start}_{i}'
                    })
                start = None
        if start is not None:
            if n_time - start >= min_duration:
                seg = series[start:]
                peak_idx_rel = np.argmax(seg)
                peak_idx = start + peak_idx_rel
                events.append({
                    'basin': b,
                    'start': start,
                    'end': n_time,
                    'duration': n_time-start,
                    'peak_idx': peak_idx,
                    'peak_value': series[peak_idx],
                    'event_name': f'Event_{b}_{start}_{n_time}'
                })
    return events

FLOW_VAR_CANDIDATES = ['inflow','flow','streamflow','discharge','qobs','qsim']
PRECIP_VAR_CANDIDATES = ['p','precip','precipitation','rain','rainfall','prcp','precipitationCal','precipitationFlux']

# Additional helpers for precipitation / shape alignment
def select_precip_array(dataset: Dict[str, np.ndarray], verbose=False):
    # Direct candidates
    for vn in PRECIP_VAR_CANDIDATES:
        if vn in dataset:
            if verbose:
                print(f"[precip] Found direct variable: {vn} shape={getattr(dataset[vn],'shape',None)}")
            return dataset[vn], vn
    # Fuzzy search
    for vn in dataset.keys():
        low = vn.lower()
        if 'precip' in low or 'rain' in low:
            if verbose:
                print(f"[precip] Using fuzzy variable: {vn} shape={getattr(dataset[vn],'shape',None)}")
            return dataset[vn], vn
    if verbose:
        print('[precip] No precipitation variable detected.')
    return None, None

def harmonize_time_dimension(obs_flow: np.ndarray, model_flows: Dict[str,np.ndarray], time_array: np.ndarray, verbose=False):
    """Crop all arrays to the minimum common time length if mismatched.
    Returns new (obs_flow, model_flows, time_array, cropped) where cropped is True if any change occurred.
    """
    lengths = [obs_flow.shape[1]]
    for arr in model_flows.values():
        if arr is not None:
            lengths.append(arr.shape[1])
    min_len = min(lengths)
    max_len = max(lengths)
    cropped = False
    if min_len != max_len:
        cropped = True
        if verbose:
            print(f"[align] Time length mismatch detected. Cropping all to {min_len} (was {lengths}).")
        obs_flow = obs_flow[:, :min_len]
        if time_array is not None and len(time_array) >= min_len:
            time_array = time_array[:min_len]
        new_model_flows = {}
        for k, arr in model_flows.items():
            if arr is None:
                new_model_flows[k] = None
            else:
                if arr.shape[1] < min_len:  # pad with NaN if shorter (corner case)
                    pad_width = min_len - arr.shape[1]
                    if verbose:
                        print(f"[align] Padding model {k} with {pad_width} NaNs to reach {min_len}.")
                    pad = np.full((arr.shape[0], pad_width), np.nan, dtype=float)
                    arr = np.concatenate([arr, pad], axis=1)
                new_model_flows[k] = arr[:, :min_len]
        model_flows = new_model_flows
    else:
        if verbose:
            print('[align] All model time lengths already consistent.')
    return obs_flow, model_flows, time_array, cropped

def _find_first_match(folder: Path, patterns: List[str]):
    for pat in patterns:
        matches = list(folder.glob(pat))
        if matches:
            return matches[0]
    return None

def discover_epoch_files(model_dir: Path, epoch: str):
    """Return (obs_path, pred_path) for given model and epoch (flexible).
    Enhanced: also try epoch stem if an extension (e.g., .pth) is present.
    """
    normal_dir = model_dir / 'songliao_3h_test_normal'
    if not normal_dir.exists():
        return None, None
    epoch_str = str(epoch)
    epoch_stem = Path(epoch_str).stem  # e.g. best_model.pth -> best_model
    # Exact style (legacy)
    obs_exact = normal_dir / f"epoch{epoch_str}flow_obs.nc"
    pred_exact = normal_dir / f"epoch{epoch_str}flow_pred.nc"
    # Also consider stem version
    obs_exact_stem = normal_dir / f"epoch{epoch_stem}flow_obs.nc"
    pred_exact_stem = normal_dir / f"epoch{epoch_stem}flow_pred.nc"

    def _find_any(pattern_groups):
        for patterns in pattern_groups:
            for pat in patterns:
                matches = list(normal_dir.glob(pat))
                if matches:
                    return matches[0]
        return None

    if obs_exact.exists():
        obs_path = obs_exact
    elif obs_exact_stem.exists():
        obs_path = obs_exact_stem
    else:
        obs_path = _find_any([
            [f"*{epoch_str}*flow_obs*.nc", f"*{epoch_str}*obs*.nc", f"*{epoch_str}*.nc"],
            [f"*{epoch_stem}*flow_obs*.nc", f"*{epoch_stem}*obs*.nc", f"*{epoch_stem}*.nc"],
        ])

    if pred_exact.exists():
        pred_path = pred_exact
    elif pred_exact_stem.exists():
        pred_path = pred_exact_stem
    else:
        pred_path = _find_any([
            [f"*{epoch_str}*flow_pred*.nc", f"*{epoch_str}*pred*.nc", f"*{epoch_str}*.nc"],
            [f"*{epoch_stem}*flow_pred*.nc", f"*{epoch_stem}*pred*.nc", f"*{epoch_stem}*.nc"],
        ])
    return obs_path, pred_path

def load_model_comparison_data(model_dirs: List[str], model_names: List[str], epoch: str):
    data = {}
    obs_loaded = False
    obs_meta = None
    for i, md in enumerate(model_dirs):
        model_dir = Path(md).expanduser().resolve()
        model_name = model_names[i]  # Use provided model name
        obs_path, pred_path = discover_epoch_files(model_dir, epoch)
        if not obs_loaded and obs_path and obs_path.exists():
            print(f"Using observations from {obs_path}")
            data['obs'] = load_netcdf_data(obs_path)
            obs_loaded = True
            obs_meta = model_name
        if pred_path and pred_path.exists():
            print(f"Loading {model_name} predictions: {pred_path}")
            data[model_name] = load_netcdf_data(pred_path)
        else:
            print(f"Prediction file missing for {model_name} (from dir {model_dir.name}) epoch={epoch}")
            data[model_name] = None
    if 'obs' not in data:
        print("ERROR: No observation file found in provided model directories")
    else:
        print(f"Observations sourced from model folder: {obs_meta}")
    return data

def select_flow_array(dataset: Dict[str, np.ndarray]):
    for vn in FLOW_VAR_CANDIDATES:
        if vn in dataset:
            return dataset[vn], vn
    # fallback
    first = list(dataset.keys())[0]
    return dataset[first], first

# --- Metrics (only NSE kept for display) ---
def calculate_metrics(obs, pred):
    mask = ~(np.isnan(obs) | np.isnan(pred))
    if mask.sum() < 2:
        return {'error':'insufficient'}
    o = obs[mask]; p = pred[mask]
    denom = np.sum((o - o.mean())**2)
    nse = 1 - np.sum((o-p)**2)/denom if denom>0 else -np.inf
    return {'NSE': nse}

# ============================================================================
# NEW: Multi-model rainfall-runoff plotting cloned from plot_normal_comparison
# ============================================================================

def _load_rainfall_for_window(basin_name, time_subset_dt, verbose=False):
    """Try to load rainfall (and flood_event mask) via load_songliao_timeseries_data to match time subset.
    Returns (rain, flood_mask) or (None, None) if not available."""
    if not callable(load_songliao_timeseries_data):
        return None, None
    if time_subset_dt is None or len(time_subset_dt)==0:
        return None, None
    try:
        t0 = pd.to_datetime(time_subset_dt[0])
        t1 = pd.to_datetime(time_subset_dt[-1])
        raw = load_songliao_timeseries_data(basin_name, (t0, t1))
        if raw is None or len(raw)==0:
            return None, None
        # Ensure datetime index
        if 'time' in raw.columns:
            raw = raw.set_index(pd.to_datetime(raw['time']))
        target_index = pd.to_datetime(time_subset_dt)
        aligned = raw.reindex(target_index, method='nearest', tolerance=pd.Timedelta(hours=1))
        # Rain column search
        rain_col = None
        for c in ['rain','precip','precipitation']:
            if c in aligned.columns:
                rain_col = c; break
        if rain_col is None:
            cand = [c for c in aligned.columns if 'rain' in c.lower() or 'precip' in c.lower()]
            if cand:
                rain_col = cand[0]
        rain = aligned[rain_col].to_numpy() if rain_col else None
        flood_mask = None
        if 'flood_event' in aligned.columns:
            flood_mask = (aligned['flood_event'].fillna(0).astype(int).to_numpy()==1)
        if verbose:
            if rain is not None:
                print(f"[rain] Loaded rainfall for {basin_name}: len={len(rain)} range={np.nanmin(rain):.3f}-{np.nanmax(rain):.3f}")
            else:
                print(f"[rain] No rainfall column found for {basin_name}")
        return rain, flood_mask
    except Exception as e:
        if verbose:
            print(f"[rain] Failed loading rainfall for {basin_name}: {e}")
        return None, None


def plot_multi_model_rainfall_runoff_event(
    obs_flow: np.ndarray,
    model_flows: Dict[str, dict],
    period: dict,
    basin_names: List[str],
    time_array: np.ndarray,
    save_dir: Path,
    epoch_str: str,
    flow_unit: str,
    songliao_path: str,
    precip_fallback: np.ndarray,
    verbose: bool=False,
):
    """Replicated style of plot_normal_comparison: top inverted rainfall bars, bottom hydrographs.
    Shows observed + all model predictions (dashed). Only NSE per model annotated.
    """
    import pandas as pd

    basin_idx = period['basin']
    basin_name = basin_names[basin_idx] if basin_names is not None else f"Basin_{basin_idx}"

    # Context window (±48 timesteps)
    start_idx = max(0, period['start'] - 48)
    end_idx = min(obs_flow.shape[1], period['end'] + 48)

    # Time subset
    if time_array is not None:
        time_subset = time_array[start_idx:end_idx]
        try:
            time_subset_dt = pd.to_datetime(time_subset)
            use_dt = True
        except Exception:
            time_subset_dt = time_subset
            use_dt = False
    else:
        time_subset_dt = np.arange(start_idx, end_idx)
        use_dt = False

    obs_seg = obs_flow[basin_idx, start_idx:end_idx]

    # Rainfall (primary: Songliao loader; fallback: obs precip array)
    rain_seg, flood_mask = _load_rainfall_for_window(basin_name, time_subset_dt if use_dt else None, verbose=verbose)
    if rain_seg is None and precip_fallback is not None:
        rain_seg = precip_fallback[basin_idx, start_idx:end_idx]
        if verbose:
            print(f"[rain] Using fallback precip for {basin_name} range={np.nanmin(rain_seg):.3f}-{np.nanmax(rain_seg):.3f}")
    # Ensure length alignment
    if rain_seg is not None and len(rain_seg) != len(obs_seg):
        min_len = min(len(rain_seg), len(obs_seg))
        rain_seg = rain_seg[:min_len]
        obs_seg = obs_seg[:min_len]
        time_subset_dt = time_subset_dt[:min_len]
        if flood_mask is not None:
            flood_mask = flood_mask[:min_len]
    if flood_mask is None:
        # Build a simple mask only for the main event window within subset
        flood_mask = np.zeros(len(time_subset_dt), dtype=bool)
        # Map event indices into subset range
        ev_s = period['start'] - start_idx
        ev_e = period['end'] - start_idx
        ev_s = max(0, ev_s); ev_e = min(len(flood_mask), ev_e)
        flood_mask[ev_s:ev_e] = True

    # Prepare figure
    fig, (ax_rain, ax_flow) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios':[1,3]}, sharex=True)

    # Rainfall bars (inverted). Different colors for flood vs normal
    if rain_seg is not None:
        if len(time_subset_dt) > 1:
            try:
                if use_dt:
                    delta_h = (time_subset_dt[1]-time_subset_dt[0]).astype('timedelta64[h]').astype(int) if not isinstance(time_subset_dt[0], pd.Timestamp) else (time_subset_dt[1]-time_subset_dt[0]).total_seconds()/3600
                    bar_width = delta_h * 0.8 / 24
                else:
                    bar_width = 0.6
            except Exception:
                bar_width = 0.6
        else:
            bar_width = 0.6
        colors = ['orange' if f else 'skyblue' for f in flood_mask]
        ax_rain.bar(time_subset_dt, rain_seg, color=colors, alpha=0.75, width=bar_width, edgecolor='none')
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='skyblue', alpha=0.75, label='Normal Period'), Patch(facecolor='orange', alpha=0.75, label='Flood Period')]
        ax_rain.legend(handles=legend_elements, loc='upper right')
        ax_rain.set_ylabel('Precipitation (mm/3h)', fontsize=LABEL_FONT)
        ax_rain.invert_yaxis()
        ax_rain.grid(True, alpha=0.3)

    # Flow panel
    ax_flow.plot(time_subset_dt, obs_seg, color='black', linewidth=2.2, label='Observed')

    metrics_rows = []
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    idx_color = 0
    for model_name, meta in model_flows.items():
        pred_arr = meta.get('flow')
        model_basin_names = meta.get('basins')
        if pred_arr is None:
            if verbose:
                print(f"[event-debug] Skip {model_name}: prediction array is None")
            continue
        # --- FIX & diagnostics: avoid truth-value check on numpy array; convert to list if needed ---
        if isinstance(model_basin_names, np.ndarray):
            model_basin_names = model_basin_names.tolist()
        if model_basin_names is not None and not isinstance(model_basin_names, list):
            try:
                model_basin_names = list(model_basin_names)
            except Exception:
                model_basin_names = None
        if model_basin_names is not None and len(model_basin_names) == pred_arr.shape[0]:
            try:
                mb_idx = model_basin_names.index(basin_name)
            except ValueError:
                if verbose:
                    print(f"[event-debug] {model_name}: basin {basin_name} not found in its basin list -> skip")
                continue
        else:
            mb_idx = basin_idx if basin_idx < pred_arr.shape[0] else None
        if mb_idx is None:
            if verbose:
                print(f"[event-debug] {model_name}: computed basin index None -> skip")
            continue
        if pred_arr.shape[1] < end_idx:
            # Instead of skipping, clip to available length and warn
            if verbose:
                print(f"[event-debug] {model_name}: shorter time length ({pred_arr.shape[1]}) than required end_idx {end_idx}, clipping.")
            local_end = pred_arr.shape[1]
        else:
            local_end = end_idx
        pred_seg = pred_arr[mb_idx, start_idx:local_end]
        # Align lengths with obs_seg
        max_common = min(len(pred_seg), len(obs_seg))
        pred_seg = pred_seg[:max_common]
        obs_aligned = obs_seg[:max_common]
        time_aligned = time_subset_dt[:max_common]
        color = color_cycle[idx_color % len(color_cycle)] if color_cycle else None
        idx_color += 1
        ax_flow.plot(time_aligned, pred_seg, linestyle='--', linewidth=1.8, label=model_name, color=color, alpha=0.95)
        m = calculate_metrics(obs_aligned, pred_seg)
        if 'error' not in m:
            m.update({'model': model_name})
            metrics_rows.append(m)

    # Highlight main event span once
    ev_s = period['start'] - start_idx
    ev_e = period['end'] - start_idx
    if ev_s < len(time_subset_dt) and ev_e > 0:
        ev_s_clamped = max(0, ev_s)
        ev_e_clamped = min(len(time_subset_dt)-1, ev_e-1)
        ax_flow.axvspan(time_subset_dt[ev_s_clamped], time_subset_dt[ev_e_clamped], color='red', alpha=0.15, label='Flood Event')

    event_name = period.get('event_name', f"event_{period['start']}_{period['end']}")
    fig.suptitle(f"{basin_name} - {event_name}\nEpoch: {Path(str(epoch_str)).stem}", fontsize=TITLE_FONT, fontweight='bold')
    ax_flow.set_ylabel(f"Flow ({flow_unit})", fontsize=LABEL_FONT)
    ax_flow.set_xlabel('Time', fontsize=LABEL_FONT)
    ax_flow.grid(True, alpha=0.3)
    ax_flow.legend(loc='upper right')

    # NSE annotation
    if metrics_rows:
        text = 'NSE per model:\n' + '\n'.join([f"{r['model']}: {r['NSE']:.3f}" for r in metrics_rows])
        ax_flow.text(0.01, 0.99, text, transform=ax_flow.transAxes, ha='left', va='top', fontsize=ANNOT_FONT,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    if use_dt:
        plt.setp(ax_flow.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout(rect=[0,0,1,0.95])

    filename = f"{basin_name}_{event_name}_epoch{Path(str(epoch_str)).stem}.png".replace(' ', '_')
    plt.savefig(Path(save_dir)/filename, dpi=300, bbox_inches='tight')
    plt.close()

    for r in metrics_rows:
        r.update({
            'basin': basin_name,
            'event_name': event_name,
            'period_start': period['start'],
            'period_end': period['end'],
            'duration': period['duration'],
            'epoch': Path(str(epoch_str)).stem
        })
    return metrics_rows

# ============================================================================

def run_model_comparison(model_dirs: List[str], model_names: List[str], epoch: str, output_dir: str=None, songliao_path: str='/mnt/data/ClassC/songliaorrevent', verbose: bool=False):
    data = load_model_comparison_data(model_dirs, model_names, epoch)
    if 'obs' not in data or data['obs'] is None:
        print('Abort: observation data missing.')
        return
    if verbose:
        print(f"[debug] Observation variables: {list(data['obs'].keys())}")
    obs_flow, obs_var = select_flow_array(data['obs'])

    # Precip fallback
    precip_fallback = np.zeros_like(obs_flow)
    for cand in PRECIP_VAR_CANDIDATES:
        if cand in data['obs']:
            var = data['obs'][cand]
            if isinstance(var, np.ndarray) and var.shape[:2] == obs_flow.shape:
                precip_fallback = var
                if verbose:
                    print(f"[rain-fallback] Using {cand} as fallback precipitation.")
                break

    time_array = data['obs'].get('time')
    basin_names = data['obs'].get('basin')
    if basin_names is not None and len(basin_names)>0:
        try:
            if hasattr(basin_names[0],'decode'):
                basin_names = [b.decode('utf-8') if hasattr(b,'decode') else str(b) for b in basin_names]
        except Exception:
            basin_names = [f"Basin_{i}" for i in range(obs_flow.shape[0])]
    else:
        basin_names = [f"Basin_{i}" for i in range(obs_flow.shape[0])]

    flow_unit = 'mm/3h'
    basin_areas = None
    if 'convert_flow_units_batch' in globals():
        basin_areas = {}
        for i, bid in enumerate(basin_names):
            attrs = load_basin_attributes(bid, songliao_path)
            basin_areas[i] = attrs.get('area') if attrs and 'area' in attrs else None
        obs_flow, flow_unit = convert_flow_units_batch(obs_flow, basin_areas, time_unit='3h')

    # Gather model predictions
    temp_arrays = []
    model_flows = {}
    for i, md in enumerate(model_dirs):
        name = model_names[i]
        pred_data = data.get(name)
        if pred_data is None:
            if verbose:
                print(f"[model] {name} prediction data missing")
            model_flows[name] = {'flow': None, 'basins': None}
            continue
        model_basin_names = pred_data.get('basin')
        if model_basin_names is not None and len(model_basin_names)>0:
            try:
                if hasattr(model_basin_names[0],'decode'):
                    model_basin_names = [b.decode('utf-8') if hasattr(b,'decode') else str(b) for b in model_basin_names]
            except Exception:
                model_basin_names = [str(i) for i in range(len(model_basin_names))]
        arr, _ = select_flow_array(pred_data)
        if 'convert_flow_units_batch' in globals() and basin_areas is not None:
            arr,_ = convert_flow_units_batch(arr, basin_areas, time_unit='3h')
        temp_arrays.append(arr)
        model_flows[name] = {'flow': arr, 'basins': model_basin_names}
        if verbose:
            print(f"[model] {name} flow shape {arr.shape}; basins loaded {len(model_basin_names) if model_basin_names is not None else 'None'}")

    # Align time lengths
    lengths = [obs_flow.shape[1]] + [a.shape[1] for a in temp_arrays if a is not None]
    min_len = min(lengths)
    if any(l != min_len for l in lengths):
        if verbose:
            print(f"[align] Cropping all time dims to {min_len}")
        obs_flow = obs_flow[:, :min_len]
        if time_array is not None and len(time_array) >= min_len:
            time_array = time_array[:min_len]
        precip_fallback = precip_fallback[:, :min_len]
        for k, meta in model_flows.items():
            arr = meta['flow']
            if arr is not None and arr.shape[1] >= min_len:
                model_flows[k]['flow'] = arr[:, :min_len]

    # Flood events
    if USE_FLOOD_LABELS and 'find_flood_events_from_labels' in globals():
        print('Detecting flood events using Songliao labels...')
        periods = find_flood_events_from_labels(obs_flow, time_array, basin_names, songliao_path)
    else:
        print('Detecting flood events using simple percentile method...')
        periods = simple_find_flood_events(obs_flow, time_array)
    if not periods:
        print('No flood events detected.')
        return

    if output_dir is None:
        output_dir = Path(model_dirs[0]) / 'songliao_3h_test_normal'
    output_dir = Path(output_dir)
    epoch_folder = Path(str(epoch)).stem
    plots_dir = output_dir / f'epoch{epoch_folder}' / 'model_comparison_plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {plots_dir}')

    all_metrics = []
    for period in periods:
        basin_dir = plots_dir / basin_names[period['basin']].replace(' ','_')
        basin_dir.mkdir(exist_ok=True, parents=True)
        metrics_rows = plot_multi_model_rainfall_runoff_event(
            obs_flow, model_flows, period, basin_names, time_array, basin_dir, epoch, flow_unit, songliao_path, precip_fallback, verbose=verbose
        )
        all_metrics.extend(metrics_rows)

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        (plots_dir / 'model_comparison_metrics.csv').write_text(df.to_csv(index=False), encoding='utf-8')
        summary_lines = [f'Epoch: {epoch}', f'Total events: {df.event_name.nunique()}', f'Models: {", ".join(model_flows.keys())}']
        for model_name in model_flows.keys():
            sub = df[df['model']==model_name]
            if len(sub)==0: continue
            summary_lines.append(f'{model_name}: NSE_mean={sub.NSE.mean():.3f} NSE_std={sub.NSE.std():.3f}')
        (plots_dir / 'model_comparison_summary.txt').write_text('\n'.join(summary_lines), encoding='utf-8')
        print('Saved metrics (NSE) and summary.')
    print('Done.')

# ===================== 新增：同一模型多情景 (normal/sim1/sim2/sim3) 对比功能 =====================

def discover_epoch_files_for_scenario(model_dir: Path, scenario: str, epoch: str):
    """Return (obs_path, pred_path) for a given model root, scenario name and epoch.
    Scenario directory pattern: songliao_3h_test_{scenario}
    Epoch file pattern kept consistent with discover_epoch_files().
    """
    scen_dir = model_dir / f'songliao_3h_test_{scenario}'
    if not scen_dir.exists():
        return None, None
    epoch_str = str(epoch)
    epoch_stem = Path(epoch_str).stem
    obs_exact = scen_dir / f"epoch{epoch_str}flow_obs.nc"
    pred_exact = scen_dir / f"epoch{epoch_str}flow_pred.nc"
    obs_exact_stem = scen_dir / f"epoch{epoch_stem}flow_obs.nc"
    pred_exact_stem = scen_dir / f"epoch{epoch_stem}flow_pred.nc"

    def _find_any(pattern_groups):
        for patterns in pattern_groups:
            for pat in patterns:
                matches = list(scen_dir.glob(pat))
                if matches:
                    return matches[0]
        return None

    if obs_exact.exists():
        obs_path = obs_exact
    elif obs_exact_stem.exists():
        obs_path = obs_exact_stem
    else:
        obs_path = _find_any([
            [f"*{epoch_str}*flow_obs*.nc", f"*{epoch_str}*obs*.nc", f"*{epoch_str}*.nc"],
            [f"*{epoch_stem}*flow_obs*.nc", f"*{epoch_stem}*obs*.nc", f"*{epoch_stem}*.nc"],
        ])

    if pred_exact.exists():
        pred_path = pred_exact
    elif pred_exact_stem.exists():
        pred_path = pred_exact_stem
    else:
        pred_path = _find_any([
            [f"*{epoch_str}*flow_pred*.nc", f"*{epoch_str}*pred*.nc", f"*{epoch_str}*.nc"],
            [f"*{epoch_stem}*flow_pred*.nc", f"*{epoch_stem}*pred*.nc", f"*{epoch_stem}*.nc"],
        ])
    return obs_path, pred_path


def load_scenario_comparison_data(model_root: str, scenarios: List[str], epoch: str):
    """Load observation (once) + prediction datasets for each scenario for a single model.
    Returns dict with keys: 'obs' (dataset dict) and scenario display keys mapping to dataset dicts.
    """
    data = {}
    obs_loaded = False
    obs_source = None
    model_dir = Path(model_root).expanduser().resolve()
    for scen in scenarios:
        obs_path, pred_path = discover_epoch_files_for_scenario(model_dir, scen, epoch)
        if not obs_loaded and obs_path and obs_path.exists():
            data['obs'] = load_netcdf_data(obs_path)
            obs_loaded = True
            obs_source = scen
        if pred_path and pred_path.exists():
            data[scen] = load_netcdf_data(pred_path)
        else:
            data[scen] = None
    if 'obs' not in data:
        print('ERROR: No observation file found among scenarios.')
    else:
        print(f"Observations sourced from scenario: {obs_source}")
    return data


def run_scenario_comparison(
    model_root_dir: str,
    scenarios: List[str],
    scenario_names: List[str],
    epoch: str,
    output_dir: str | None = None,
    songliao_path: str = '/mnt/data/ClassC/songliaorrevent',
    verbose: bool = False,
):
    """Compare the SAME model under different scenario result folders (normal/sim1/...).
    Reuses rainfall-runoff multi-line plotting style; each scenario is treated as a 'model'.
    """
    if len(scenarios) != len(scenario_names):
        raise ValueError('scenarios 与 scenario_names 长度需一致')
    data = load_scenario_comparison_data(model_root_dir, scenarios, epoch)
    if 'obs' not in data or data['obs'] is None:
        print('Abort: observation data missing.')
        return
    obs_flow, _ = select_flow_array(data['obs'])

    # Precip fallback from obs dataset
    precip_fallback = np.zeros_like(obs_flow)
    for cand in PRECIP_VAR_CANDIDATES:
        if cand in data['obs']:
            var = data['obs'][cand]
            if isinstance(var, np.ndarray) and var.shape[:2] == obs_flow.shape:
                precip_fallback = var
                if verbose:
                    print(f"[rain-fallback] Using {cand} as fallback precipitation.")
                break

    time_array = data['obs'].get('time')
    basin_names = data['obs'].get('basin')
    if basin_names is not None and len(basin_names) > 0:
        try:
            if hasattr(basin_names[0], 'decode'):
                basin_names = [b.decode('utf-8') if hasattr(b, 'decode') else str(b) for b in basin_names]
        except Exception:
            basin_names = [f"Basin_{i}" for i in range(obs_flow.shape[0])]
    else:
        basin_names = [f"Basin_{i}" for i in range(obs_flow.shape[0])]

    flow_unit = 'mm/3h'
    basin_areas = None
    if 'convert_flow_units_batch' in globals():
        basin_areas = {}
        for i, bid in enumerate(basin_names):
            attrs = load_basin_attributes(bid, songliao_path)
            basin_areas[i] = attrs.get('area') if attrs and 'area' in attrs else None
        obs_flow, flow_unit = convert_flow_units_batch(obs_flow, basin_areas, time_unit='3h')

    # Gather scenario prediction arrays
    scenario_flows = {}
    temp_arrays = []
    for scen, scen_name in zip(scenarios, scenario_names):
        pred_data = data.get(scen)
        if pred_data is None:
            scenario_flows[scen_name] = {'flow': None, 'basins': None}
            if verbose:
                print(f"[scenario] {scen} prediction data missing")
            continue
        model_basin_names = pred_data.get('basin')
        if model_basin_names is not None and len(model_basin_names) > 0:
            try:
                if hasattr(model_basin_names[0], 'decode'):
                    model_basin_names = [b.decode('utf-8') if hasattr(b, 'decode') else str(b) for b in model_basin_names]
            except Exception:
                model_basin_names = [str(i) for i in range(len(model_basin_names))]
        arr, _ = select_flow_array(pred_data)
        if 'convert_flow_units_batch' in globals() and basin_areas is not None:
            arr, _ = convert_flow_units_batch(arr, basin_areas, time_unit='3h')
        temp_arrays.append(arr)
        scenario_flows[scen_name] = {'flow': arr, 'basins': model_basin_names}
        if verbose:
            print(f"[scenario] {scen_name} flow shape {arr.shape}; basins: {len(model_basin_names) if model_basin_names is not None else 'None'}")

    # Align time lengths
    lengths = [obs_flow.shape[1]] + [a.shape[1] for a in temp_arrays if a is not None]
    min_len = min(lengths)
    if any(l != min_len for l in lengths):
        if verbose:
            print(f"[align] Cropping all time dims to {min_len}")
        obs_flow = obs_flow[:, :min_len]
        if time_array is not None and len(time_array) >= min_len:
            time_array = time_array[:min_len]
        precip_fallback = precip_fallback[:, :min_len]
        for k, meta in scenario_flows.items():
            arr = meta['flow']
            if arr is not None and arr.shape[1] >= min_len:
                scenario_flows[k]['flow'] = arr[:, :min_len]

    # Flood events detection
    if USE_FLOOD_LABELS and 'find_flood_events_from_labels' in globals():
        print('Detecting flood events using Songliao labels...')
        periods = find_flood_events_from_labels(obs_flow, time_array, basin_names, songliao_path)
    else:
        print('Detecting flood events using simple percentile method...')
        periods = simple_find_flood_events(obs_flow, time_array)
    if not periods:
        print('No flood events detected.')
        return

    # Output directory
    if output_dir is None:
        output_dir = Path(model_root_dir) / 'scenario_comparison'
    output_dir = Path(output_dir)
    epoch_folder = Path(str(epoch)).stem
    plots_dir = output_dir / f'epoch{epoch_folder}' / 'scenario_comparison_plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {plots_dir}')

    all_metrics = []
    for period in periods:
        basin_dir = plots_dir / basin_names[period['basin']].replace(' ', '_')
        basin_dir.mkdir(exist_ok=True, parents=True)
        metrics_rows = plot_multi_model_rainfall_runoff_event(
            obs_flow, scenario_flows, period, basin_names, time_array, basin_dir, epoch, flow_unit, songliao_path, precip_fallback, verbose=verbose
        )
        all_metrics.extend(metrics_rows)

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        (plots_dir / 'scenario_comparison_metrics.csv').write_text(df.to_csv(index=False), encoding='utf-8')
        summary_lines = [f'Epoch: {epoch}', f'Total events: {df.event_name.nunique()}', f'Scenarios: {", ".join(scenario_flows.keys())}']
        for scen_name in scenario_flows.keys():
            sub = df[df['model'] == scen_name]
            if len(sub) == 0:
                continue
            summary_lines.append(f'{scen_name}: NSE_mean={sub.NSE.mean():.3f} NSE_std={sub.NSE.std():.3f}')
        (plots_dir / 'scenario_comparison_summary.txt').write_text('\n'.join(summary_lines), encoding='utf-8')
        print('Saved scenario metrics (NSE) and summary.')
    print('Done.')

    return all_metrics


def compare_scenarios(model_root_dir: str, epoch: str, scenarios: List[str] | None = None, scenario_names: List[str] | None = None, **kwargs):
    """Notebook-friendly wrapper. Returns pandas DataFrame of metrics."""
    import pandas as pd
    if scenarios is None:
        scenarios = ['normal', 'sim1', 'sim2', 'sim3']
    if scenario_names is None:
        scenario_names = [s.capitalize() if s != 'normal' else 'Normal' for s in scenarios]
    metrics = run_scenario_comparison(model_root_dir, scenarios, scenario_names, epoch, **kwargs)
    if not metrics:
        return pd.DataFrame()
    return pd.DataFrame(metrics)

# ===================== 原有 parse_list 保留 =====================

def parse_list(s: str):
    return [p.strip() for p in s.split(',') if p.strip()]

# ===================== 修改 main：默认执行情景对比 =====================

def main():
    # ===== 可修改区域 =====
    compare_mode = 'scenario'  # 'scenario' 或 'model'
    epoch = 'best_model.pth'
    Songliao_path = '/mnt/data/ClassC/songliaorrevent'
    verbose = True

    if compare_mode == 'scenario':
        # 单模型多情景对比
        model_root_dir = './results/3d_lstm_gnn'
        scenarios = ['normal', 'sim1', 'sim2', 'sim3']
        scenario_names = ['Normal', 'Sim1', 'Sim2', 'Sim3']  # 可自定义显示名称
        output_dir = None  # 或指定目录如 './results/3d_lstm_gnn/scenario_compare_custom'
        run_scenario_comparison(
            model_root_dir=model_root_dir,
            scenarios=scenarios,
            scenario_names=scenario_names,
            epoch=epoch,
            output_dir=output_dir,
            songliao_path=Songliao_path,
            verbose=verbose,
        )
    else:
        # 多模型同一情景对比（原逻辑）
        model_dirs = [
            './results/3d_gnn_only',
            './results/3d_lstm_only',
            './results/3d_lstm_gnn'
        ]
        model_names = ['GNN-Only', 'LSTM-Only', 'LSTM+GNN']
        output_dir = None
        run_model_comparison(
            model_dirs=model_dirs,
            model_names=model_names,
            epoch=epoch,
            output_dir=output_dir,
            songliao_path=Songliao_path,
            verbose=verbose,
        )

    print('\n运行结束，如需重新计算修改上方参数后再次运行本单元。')

if __name__ == '__main__':
    main()
