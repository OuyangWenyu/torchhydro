#!/usr/bin/env python3
"""
Updated spatial sensitivity test results comparison plotting script (rainfall-runoff style fully cloned from plot_model_comparison.py)
Compare observed values with multiple spatial scenario predictions (e.g., normal, sim1, sim2, sim3).

This script is structured based on the `plot_normal_comparison.py` template, providing:
- Consistent flood event analysis using `flood_utils`.
- Unit conversion from mm/3h to m³/s.
- Rainfall-runoff style plotting.
- Detailed metrics calculation for each scenario.
- Standardized output structure with epoch-specific folders.

Updated paths:
- Observations: /path/to/results/XXX/songliao_3h_test_normal/epoch...flow_obs.nc
- Predictions: /path/to/results/XXX/songliao_3h_test_{scenario}/epoch...flow_pred.nc
- Output: /path/to/results/XXX/songliao_3h_test_normal/epoch{epoch}/spatial_sensitivity_plots/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
import warnings
from typing import Dict, List
warnings.filterwarnings('ignore')

# === 统一字体风格（与 plot_model_comparison.py 保持一致） ===
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
BASE_FONT = 12
LABEL_FONT = 14
TITLE_FONT = 17
ANNOT_FONT = 12

# Import flood event analysis utilities from flood_utils
try:
    from flood_utils import (
        load_songliao_timeseries_data,
        load_basin_attributes,
        convert_flow_units_batch,
        find_flood_events_from_labels,
        load_netcdf_data,
        calculate_metrics,
        find_flood_events_threshold_based
    )
    FLOOD_UTILS_AVAILABLE = True
except ImportError:
    print("Warning: flood_utils not available, using simplified internal functions.")
    FLOOD_UTILS_AVAILABLE = False
    # Define fallback functions if flood_utils is not present
    def load_netcdf_data(file_path):
        try:
            ds = xr.open_dataset(file_path)
            data_dict = {var: ds[var].values for var in ds.data_vars}
            for coord in ['time', 'basin']:
                if coord in ds.coords:
                    data_dict[coord] = ds.coords[coord].values
            return data_dict
        except Exception as e:
            print(f"Failed to load file {file_path}: {e}")
            return None
    def calculate_metrics(obs, pred):
        valid_mask = ~(np.isnan(obs) | np.isnan(pred))
        if valid_mask.sum() < 2: return {'error': 'insufficient_data'}
        obs_valid, pred_valid = obs[valid_mask], pred[valid_mask]
        numerator = np.sum((obs_valid - pred_valid) ** 2)
        denominator = np.sum((obs_valid - np.mean(obs_valid)) ** 2)
        nse = 1 - (numerator / denominator) if denominator > 0 else -np.inf
        r2 = np.corrcoef(obs_valid, pred_valid)[0, 1] ** 2 if len(obs_valid) > 1 else 0
        rmse = np.sqrt(np.mean((obs_valid - pred_valid) ** 2))
        mae = np.mean(np.abs(obs_valid - pred_valid))
        obs_peak, pred_peak = np.max(obs_valid), np.max(pred_valid)
        peak_error = abs(pred_peak - obs_peak) / obs_peak * 100 if obs_peak > 0 else 0
        return {'NSE': nse, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'obs_peak': obs_peak, 'pred_peak': pred_peak, 'peak_error_pct': peak_error}
    def find_flood_events_threshold_based(*args, **kwargs): return []
    def load_basin_attributes(*args, **kwargs): return None
    def convert_flow_units_batch(flow, *args, **kwargs): return flow, "mm/3h"
    def load_songliao_timeseries_data(*args, **kwargs): return None


# Import rainfall-runoff plotting functionality
try:
    from hydroutils.hydro_plot import plot_rainfall_runoff
    HYDROUTILS_AVAILABLE = True
except ImportError:
    print("Warning: hydroutils not available, using simplified plotting")
    HYDROUTILS_AVAILABLE = False

def _find_first_match(folder: Path, patterns: List[str]):
    """Find the first file in a folder matching a list of glob patterns."""
    for pat in patterns:
        matches = list(folder.glob(pat))
        if matches:
            return matches[0]
    return None

def load_spatial_sensitivity_data(base_dir, epoch: str = "30", scenarios: List[str] = None):
    """
    Load spatial sensitivity test data with a flexible epoch string.
    This is modeled after `load_normal_comparison_data`.
    """
    base_dir = Path(base_dir)
    if scenarios is None:
        scenarios = ["normal", "sim1", "sim2", "sim3"]

    data: Dict[str, dict] = {}
    
    # --- Load Observation Data ---
    # Observation is assumed to be in the 'normal' scenario directory
    obs_dir = base_dir / "songliao_3h_test_normal"
    obs_path = _find_first_match(obs_dir, [
        f"epoch{epoch}flow_obs.nc",
        f"*{epoch}*flow_obs*.nc",
        f"*{epoch}*obs*.nc"
    ])
    
    if obs_path and obs_path.exists():
        print(f"Loading observations: {obs_path}")
        data['obs'] = load_netcdf_data(obs_path)
    else:
        print(f"ERROR: Observation file not found for epoch='{epoch}' in {obs_dir}")
        data['obs'] = None

    # --- Load Prediction Data for Each Scenario ---
    for sc in scenarios:
        sc_dir = base_dir / f"songliao_3h_test_{sc}"
        pred_path = _find_first_match(sc_dir, [
            f"epoch{epoch}flow_pred.nc",
            f"*{epoch}*flow_pred*.nc",
            f"*{epoch}*pred*.nc"
        ])
        
        if pred_path and pred_path.exists():
            print(f"Loading '{sc}' predictions: {pred_path}")
            data[sc] = load_netcdf_data(pred_path)
        else:
            print(f"Warning: Prediction file not found for scenario '{sc}' epoch='{epoch}' in {sc_dir}")
            data[sc] = None
            
    return data

def plot_spatial_comparison_period(
    obs_flow, sim_flows, period, basin_names, time_array, save_dir, 
    songliao_data_path=None, flow_unit="m³/s", scenarios=None
):
    """
    Plot spatial sensitivity comparison for a single period with rainfall.
    This is modeled after `plot_normal_comparison_period_with_rainfall`.
    """
    if scenarios is None:
        scenarios = ["normal", "sim1", "sim2", "sim3"]
        
    basin_idx = period['basin']
    # Extend window to show context around the event
    start_idx = max(0, period['start'] - 48)
    end_idx = min(obs_flow.shape[1], period['end'] + 48)
    
    # Extract time and flow data for the window
    time_subset = time_array[start_idx:end_idx] if time_array is not None else np.arange(start_idx, end_idx)
    try:
        time_subset_dt = pd.to_datetime(time_subset)
        use_datetime = True
    except Exception:
        time_subset_dt = time_subset
        use_datetime = False
        
    obs_subset = obs_flow[basin_idx, start_idx:end_idx]
    
    # --- Load Rainfall Data ---
    rainfall_data = None
    if FLOOD_UTILS_AVAILABLE and basin_names is not None:
        try:
            basin_id = basin_names[basin_idx]
            time_range = (time_subset_dt[0], time_subset_dt[-1]) if use_datetime and len(time_subset_dt) > 0 else None
            
            songliao_data = load_songliao_timeseries_data(basin_id, time_range, data_path=songliao_data_path)
            if songliao_data is not None and not songliao_data.empty:
                songliao_data = songliao_data.set_index('time')
                target_times = pd.to_datetime(time_subset_dt)
                aligned_data = songliao_data.reindex(target_times, method='nearest', tolerance='1H')
                
                if not aligned_data.empty:
                    rainfall_data = aligned_data['rain'].values
                    # Ensure consistent length
                    min_len = min(len(rainfall_data), len(time_subset_dt))
                    if len(rainfall_data) != len(time_subset_dt):
                        rainfall_data = rainfall_data[:min_len]
                        time_subset_dt = time_subset_dt[:min_len]
                        obs_subset = obs_subset[:min_len]
        except Exception as e:
            print(f"Failed to load rainfall data for basin {basin_id}: {e}")

    # --- Plotting ---
    basin_name_str = basin_names[basin_idx] if basin_names is not None else f"Basin_{basin_idx}"
    event_name_str = period.get('event_name', f"Event_{period['start']}_{period['end']}")
    title = f'{basin_name_str} - Spatial Sensitivity Comparison\nFlood Event: {event_name_str}'
    
    # Define plot styles for different scenarios
    style_map = {
        'normal': {'color': '#ff7f0e', 'linestyle': '-', 'label': 'Normal (Baseline)'},
        'sim1': {'color': '#1f77b4', 'linestyle': '--', 'label': 'Sim1 (Upstream)'},
        'sim2': {'color': '#2ca02c', 'linestyle': '-.', 'label': 'Sim2 (Midstream)'},
        'sim3': {'color': '#d62728', 'linestyle': ':', 'label': 'Sim3 (Downstream)'},
    }
    
    flow_list = [obs_subset]
    leg_list = ["Observed"]
    color_list = ["black"]
    linestyle_list = ["-"]
    
    all_metrics = []

    for sc in scenarios:
        if sc in sim_flows and sim_flows[sc] is not None:
            sim_subset = sim_flows[sc][basin_idx, start_idx:end_idx]
            # Ensure consistent length
            sim_subset = sim_subset[:len(time_subset_dt)]
            
            style = style_map.get(sc, {})
            flow_list.append(sim_subset)
            leg_list.append(style.get('label', sc))
            color_list.append(style.get('color', 'gray'))
            linestyle_list.append(style.get('linestyle', '--'))
            
            # Calculate metrics for this scenario
            metrics = calculate_metrics(obs_subset, sim_subset)
            if 'error' not in metrics:
                metrics['scenario'] = sc
                all_metrics.append(metrics)

    if HYDROUTILS_AVAILABLE and rainfall_data is not None:
        fig, ax = plot_rainfall_runoff(
            time_subset_dt, rainfall_data, flow_list,
            fig_size=(16, 10), title=title, ylabel=f"Flow ({flow_unit})",
            prcp_ylabel="Precipitation (mm/3h)", leg_lst=leg_list,
            linewidth=2, color_list=color_list, linestyle_list=linestyle_list
        )
    else:
        # Simplified plot without rainfall
        fig, ax = plt.subplots(figsize=(14, 8))
        for i, (flow, label, color, ls) in enumerate(zip(flow_list, leg_list, color_list, linestyle_list)):
            ax.plot(time_subset_dt, flow, color=color, linestyle=ls, linewidth=2.5 if i==0 else 2, label=label, alpha=0.9)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel(f"Flow ({flow_unit})")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.4)

    # Add metrics text box
    if all_metrics:
        metrics_text = "Evaluation Metrics:\n"
        for m in all_metrics:
            sc_label = style_map.get(m['scenario'], {}).get('label', m['scenario'])
            metrics_text += f"{sc_label}: NSE={m['NSE']:.3f}, R²={m['R2']:.3f}\n"
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if use_datetime:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    filename = f"{basin_name_str.replace(' ', '_')}_{event_name_str}.png"
    filepath = save_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return all_metrics

def plot_spatial_sensitivity_comparison(data, save_dir, epoch="30", scenarios=None, songliao_data_path=None):
    """
    Main function to drive the spatial sensitivity comparison plotting.
    This is modeled after `plot_normal_comparison`.
    """
    if scenarios is None:
        scenarios = ["normal", "sim1", "sim2", "sim3"]
        
    save_dir = Path(save_dir)
    epoch_folder = Path(str(epoch)).stem
    plots_dir = save_dir / f"epoch{epoch_folder}" / "spatial_sensitivity_plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    obs_data = data.get('obs')
    if obs_data is None:
        print("ERROR: Missing observation data. Aborting.")
        return

    # --- Extract and Prepare Data ---
    flow_var_names = ['inflow', 'flow', 'streamflow', 'discharge', 'qobs', 'qsim']
    
    def get_flow_array(d):
        if d is None: return None
        for name in flow_var_names:
            if name in d: return d[name]
        # Fallback to first variable
        key = next(iter(d))
        if key not in ['time', 'basin']: return d[key]
        return None

    obs_flow = get_flow_array(obs_data)
    time_array = obs_data.get('time')
    basin_names = obs_data.get('basin')
    
    if basin_names is not None:
        try:
            if hasattr(basin_names[0], 'decode'):
                basin_names = [name.decode('utf-8') for name in basin_names]
        except Exception:
            basin_names = [f"Basin_{i}" for i in range(obs_flow.shape[0])]
    else:
        basin_names = [f"Basin_{i}" for i in range(obs_flow.shape[0])]

    # --- Unit Conversion ---
    flow_unit = "mm/3h"
    if FLOOD_UTILS_AVAILABLE:
        basin_areas = {i: (load_basin_attributes(bid, songliao_data_path) or {}).get('area') for i, bid in enumerate(basin_names)}
        obs_flow, flow_unit = convert_flow_units_batch(obs_flow, basin_areas, time_unit="3h")
    
    sim_flows = {}
    for sc in scenarios:
        sim_data = data.get(sc)
        if sim_data:
            sim_flow_raw = get_flow_array(sim_data)
            if sim_flow_raw is not None:
                if FLOOD_UTILS_AVAILABLE:
                    sim_flows[sc], _ = convert_flow_units_batch(sim_flow_raw, basin_areas, time_unit="3h")
                else:
                    sim_flows[sc] = sim_flow_raw
            else:
                sim_flows[sc] = None
        else:
            sim_flows[sc] = None

    # --- Find Flood Events ---
    if FLOOD_UTILS_AVAILABLE:
        print("\n=== Using Songliao dataset labels to identify flood events ===")
        periods = find_flood_events_from_labels(obs_flow, time_array, basin_names, songliao_data_path)
    else:
        print("\n=== Using threshold-based method to identify flood events ===")
        periods = find_flood_events_threshold_based(obs_flow, time_array, basin_names)

    if not periods:
        print("No flood events found. Exiting.")
        return
    print(f"Found {len(periods)} flood events to analyze.")

    # --- Plot Each Event and Collect Metrics ---
    all_metrics_df = []
    for period in periods:
        basin_name_str = basin_names[period['basin']].replace(' ', '_')
        print(f"\nPlotting {basin_name_str} - {period['event_name']}")
        
        basin_dir = plots_dir / basin_name_str
        basin_dir.mkdir(exist_ok=True)
        
        period_metrics = plot_spatial_comparison_period(
            obs_flow, sim_flows, period, basin_names, time_array, basin_dir,
            songliao_data_path=songliao_data_path, flow_unit=flow_unit, scenarios=scenarios
        )
        
        if period_metrics:
            for m in period_metrics:
                m.update({
                    'basin': basin_names[period['basin']],
                    'event_name': period['event_name'],
                    'year': period.get('year', 'N/A'),
                })
                all_metrics_df.append(m)

    # --- Save Overall Metrics and Summary ---
    if all_metrics_df:
        df = pd.DataFrame(all_metrics_df)
        csv_path = plots_dir / "spatial_sensitivity_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved detailed metrics to {csv_path}")

        summary_text = f"Spatial Sensitivity Analysis Summary (Epoch: {epoch_folder})\n"
        summary_text += f"Analyzed {df['event_name'].nunique()} events across {df['basin'].nunique()} basins.\n\n"
        
        for sc in scenarios:
            sc_df = df[df['scenario'] == sc]
            if not sc_df.empty:
                summary_text += f"--- Scenario: {sc} ---\n"
                summary_text += f"  Avg NSE: {sc_df['NSE'].mean():.3f} ± {sc_df['NSE'].std():.3f}\n"
                summary_text += f"  Avg R²: {sc_df['R2'].mean():.3f} ± {sc_df['R2'].std():.3f}\n"
                summary_text += f"  Avg RMSE: {sc_df['RMSE'].mean():.2f} ± {sc_df['RMSE'].std():.2f}\n"
                summary_text += f"  Avg Peak Error: {sc_df['peak_error_pct'].mean():.1f}% ± {sc_df['peak_error_pct'].std():.1f}%\n\n"

        summary_path = plots_dir / "spatial_sensitivity_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"Saved summary report to {summary_path}")

    print(f"\nAll plots and reports saved to: {plots_dir}")

def _load_rainfall_for_window(basin_name, time_subset_dt, verbose=False):
    """Load rainfall aligned to a time subset using load_songliao_timeseries_data if available.
    Returns (rain_array, flood_mask_or_None). Flood mask only if label exists in dataframe (column flood_event)."""
    if not FLOOD_UTILS_AVAILABLE or not callable(load_songliao_timeseries_data):
        return None, None
    if time_subset_dt is None or len(time_subset_dt) == 0:
        return None, None
    try:
        t0 = pd.to_datetime(time_subset_dt[0])
        t1 = pd.to_datetime(time_subset_dt[-1])
        raw = load_songliao_timeseries_data(basin_name, (t0, t1))
        if raw is None or len(raw) == 0:
            return None, None
        if 'time' in raw.columns:
            raw = raw.set_index(pd.to_datetime(raw['time']))
        target_index = pd.to_datetime(time_subset_dt)
        aligned = raw.reindex(target_index, method='nearest', tolerance=pd.Timedelta(hours=1))
        # find rainfall column
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
        if verbose and rain is not None:
            print(f"[rain] {basin_name} rainfall loaded len={len(rain)}")
        return rain, flood_mask
    except Exception as e:
        if verbose:
            print(f"[rain] load fail {basin_name}: {e}")
        return None, None

# === Simplified NSE-only metrics (replicating plot_model_comparison style) ===
def _nse_metric(obs, pred):
    mask = ~(np.isnan(obs) | np.isnan(pred))
    if mask.sum() < 2:
        return {'error':'insufficient'}
    o = obs[mask]; p = pred[mask]
    denom = np.sum((o - o.mean())**2)
    nse = 1 - np.sum((o-p)**2)/denom if denom>0 else -np.inf
    return {'NSE': nse}

# === File discovery for spatial scenarios ===
FLOW_FILE_PATTERNS = [
    "epoch{epoch}flow_obs.nc", "*{epoch}*flow_obs*.nc", "*{epoch}*obs*.nc"
]
PRED_FILE_PATTERNS = [
    "epoch{epoch}flow_pred.nc", "*{epoch}*flow_pred*.nc", "*{epoch}*pred*.nc"
]

def _discover_obs_file(base_dir: Path, epoch: str):
    normal_dir = base_dir / 'songliao_3h_test_normal'
    epoch_str = str(epoch)
    epoch_stem = Path(epoch_str).stem
    patterns = FLOW_FILE_PATTERNS + [p.replace('{epoch}', epoch_stem) for p in FLOW_FILE_PATTERNS]
    for pat in patterns:
        p = normal_dir / pat.format(epoch=epoch_str)
        for m in normal_dir.glob(p.name):
            return m
    return None

def _discover_pred_file(base_dir: Path, scenario: str, epoch: str):
    scen_dir = base_dir / f'songliao_3h_test_{scenario}'
    epoch_str = str(epoch)
    epoch_stem = Path(epoch_str).stem
    patterns = PRED_FILE_PATTERNS + [p.replace('{epoch}', epoch_stem) for p in PRED_FILE_PATTERNS]
    for pat in patterns:
        p = scen_dir / pat.format(epoch=epoch_str)
        for m in scen_dir.glob(p.name):
            return m
    return None

FLOW_VAR_CANDIDATES_SCEN = ['inflow','flow','streamflow','discharge','qobs','qsim']
PRECIP_VAR_CANDIDATES_SCEN = ['p','precip','precipitation','rain','rainfall','prcp']

def _select_flow_var(d: Dict[str, np.ndarray]):
    for vn in FLOW_VAR_CANDIDATES_SCEN:
        if vn in d:
            return d[vn], vn
    # fallback
    for k,v in d.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            return v, k
    first_key = next(iter(d))
    return d[first_key], first_key

# === Event plotting for scenarios (rainfall + observed + scenario lines) ===
from matplotlib.patches import Patch

def plot_spatial_scenario_event(obs_flow, scen_flows, period, basin_names, time_array, save_dir, epoch, flow_unit, precip_fallback=None, verbose=False):
    basin_idx = period['basin']
    basin_name = basin_names[basin_idx] if basin_names is not None else f'Basin_{basin_idx}'
    # context window
    start_idx = max(0, period['start'] - 48)
    end_idx = min(obs_flow.shape[1], period['end'] + 48)

    # time subset
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

    # rainfall load
    rain_seg, flood_mask = _load_rainfall_for_window(basin_name, time_subset_dt if use_dt else None, verbose=verbose)
    if rain_seg is None and precip_fallback is not None:
        rain_seg = precip_fallback[basin_idx, start_idx:end_idx]
    if rain_seg is not None and len(rain_seg) != len(obs_seg):
        L = min(len(rain_seg), len(obs_seg))
        rain_seg = rain_seg[:L]; obs_seg = obs_seg[:L]; time_subset_dt = time_subset_dt[:L]
    if flood_mask is None:
        flood_mask = np.zeros(len(time_subset_dt), dtype=bool)
        ev_s = period['start'] - start_idx
        ev_e = period['end'] - start_idx
        flood_mask[max(0,ev_s):min(len(flood_mask), ev_e)] = True

    fig, (ax_rain, ax_flow) = plt.subplots(2,1, figsize=(16,10), gridspec_kw={'height_ratios':[1,3]}, sharex=True)

    # Rain panel (CLONED STYLE)
    if rain_seg is not None:
        # 动态计算条宽（与 model comparison 相同逻辑，避免重叠）
        if len(time_subset_dt) > 1:
            try:
                if use_dt:
                    delta = time_subset_dt[1] - time_subset_dt[0]
                    if hasattr(delta, 'total_seconds'):
                        delta_h = delta.total_seconds() / 3600.0
                    else:  # numpy timedelta64
                        delta_h = pd.to_timedelta(delta).total_seconds() / 3600.0
                    bar_width = (delta_h * 0.8) / 24.0  # 转为“天”为单位的宽度
                else:
                    bar_width = 0.6
            except Exception:
                bar_width = 0.6
        else:
            bar_width = 0.6
        colors = ['orange' if f else 'skyblue' for f in flood_mask]
        ax_rain.bar(time_subset_dt, rain_seg, color=colors, alpha=0.75, width=bar_width, edgecolor='none')
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='skyblue', alpha=0.75, label='Normal Period'),
            Patch(facecolor='orange', alpha=0.75, label='Flood Period')
        ]
        ax_rain.legend(handles=legend_elements, loc='upper right')
        ax_rain.set_ylabel('Precipitation (mm/3h)', fontsize=LABEL_FONT)
        ax_rain.invert_yaxis()
        ax_rain.grid(True, alpha=0.3)

    # Flow panel
    ax_flow.plot(time_subset_dt, obs_seg, color='black', linewidth=2.2, label='Observed')

    metrics_rows = []
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    idx_color = 0
    for scen_name, meta in scen_flows.items():
        pred_arr = meta.get('flow')
        scen_basin_names = meta.get('basins')
        if pred_arr is None:
            continue
        if isinstance(scen_basin_names, np.ndarray):
            scen_basin_names = scen_basin_names.tolist()
        if scen_basin_names is not None and len(scen_basin_names) == pred_arr.shape[0]:
            try:
                mb_idx = scen_basin_names.index(basin_name)
            except ValueError:
                mb_idx = basin_idx if basin_idx < pred_arr.shape[0] else None
        else:
            mb_idx = basin_idx if basin_idx < pred_arr.shape[0] else None
        if mb_idx is None:
            continue
        local_end = min(end_idx, pred_arr.shape[1])
        pred_seg = pred_arr[mb_idx, start_idx:local_end]
        L = min(len(pred_seg), len(obs_seg))
        pred_seg = pred_seg[:L]; obs_aligned = obs_seg[:L]; time_aligned = time_subset_dt[:L]
        color = color_cycle[idx_color % len(color_cycle)] if color_cycle else None
        idx_color += 1
        ax_flow.plot(time_aligned, pred_seg, linestyle='--', linewidth=1.8, label=scen_name, color=color, alpha=0.95)
        m = _nse_metric(obs_aligned, pred_seg)
        if 'error' not in m:
            m.update({'scenario': scen_name})
            metrics_rows.append(m)

    # Highlight event window
    ev_s = period['start'] - start_idx
    ev_e = period['end'] - start_idx
    if ev_s < len(time_subset_dt) and ev_e > 0:
        ax_flow.axvspan(time_subset_dt[max(0,ev_s)], time_subset_dt[min(len(time_subset_dt)-1, ev_e-1)], color='red', alpha=0.15, label='Flood Event')

    event_name = period.get('event_name', f"event_{period['start']}_{period['end']}")
    fig.suptitle(f"{basin_name} - {event_name}\nEpoch: {Path(str(epoch)).stem}", fontsize=17, fontweight='bold')
    ax_flow.set_ylabel(f"Flow ({flow_unit})", fontsize=14)
    ax_flow.set_xlabel('Time', fontsize=14)
    ax_flow.grid(True, alpha=0.3)
    ax_flow.legend(loc='upper right')

    if metrics_rows:
        text = 'NSE per scenario:\n' + '\n'.join([f"{r['scenario']}: {r['NSE']:.3f}" for r in metrics_rows])
        ax_flow.text(0.01, 0.99, text, transform=ax_flow.transAxes, ha='left', va='top', fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    if use_dt:
        plt.setp(ax_flow.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout(rect=[0,0,1,0.95])

    filename = f"{basin_name}_{event_name}_epoch{Path(str(epoch)).stem}.png".replace(' ', '_')
    plt.savefig(Path(save_dir)/filename, dpi=300, bbox_inches='tight')
    plt.close()

    for r in metrics_rows:
        r.update({
            'basin': basin_name,
            'event_name': event_name,
            'period_start': period['start'],
            'period_end': period['end'],
            'duration': period['duration'],
            'epoch': Path(str(epoch)).stem
        })
    return metrics_rows

# === Run comparison (core) ===

def run_spatial_scenario_comparison(base_dir: str, scenarios: List[str], epoch: str, songliao_path: str='/mnt/data/ClassC/songliaorrevent', verbose: bool=False):
    base_dir = Path(base_dir).expanduser().resolve()
    if scenarios is None or len(scenarios)==0:
        scenarios = ['normal','sim1','sim2','sim3']

    # Load observation
    obs_file = _discover_obs_file(base_dir, epoch)
    if obs_file is None or not obs_file.exists():
        print(f"ERROR: observation file not found under {base_dir}/songliao_3h_test_normal for epoch={epoch}")
        return None
    print(f"Using observations: {obs_file}")
    obs_ds = load_netcdf_data(obs_file)
    obs_flow, _ = _select_flow_var(obs_ds)
    time_array = obs_ds.get('time')
    basin_names = obs_ds.get('basin')
    if basin_names is not None and len(basin_names)>0:
        try:
            if hasattr(basin_names[0],'decode'):
                basin_names = [b.decode('utf-8') if hasattr(b,'decode') else str(b) for b in basin_names]
        except Exception:
            basin_names = [f'Basin_{i}' for i in range(obs_flow.shape[0])]
    else:
        basin_names = [f'Basin_{i}' for i in range(obs_flow.shape[0])]

    flow_unit = 'mm/3h'
    basin_areas = None
    if FLOOD_UTILS_AVAILABLE:
        basin_areas = {}
        for i,bid in enumerate(basin_names):
            attrs = load_basin_attributes(bid, songliao_path)
            basin_areas[i] = attrs.get('area') if attrs and 'area' in attrs else None
        obs_flow, flow_unit = convert_flow_units_batch(obs_flow, basin_areas, time_unit='3h')

    # Precip fallback (if any precip-like variable in obs file)
    precip_fallback = np.zeros_like(obs_flow)
    for cand in PRECIP_VAR_CANDIDATES_SCEN:
        if cand in obs_ds and isinstance(obs_ds[cand], np.ndarray) and obs_ds[cand].shape[:2] == obs_flow.shape:
            precip_fallback = obs_ds[cand]
            if verbose:
                print(f"[rain-fallback] Using {cand}")
            break

    # Load scenario predictions
    scen_flows = {}
    temp_arrays = []
    for scen in scenarios:
        pred_file = _discover_pred_file(base_dir, scen, epoch)
        if pred_file and pred_file.exists():
            if verbose:
                print(f"Loading scenario {scen} predictions: {pred_file}")
            ds = load_netcdf_data(pred_file)
            arr, _ = _select_flow_var(ds)
            scen_basin_names = ds.get('basin')
            if scen_basin_names is not None and len(scen_basin_names)>0:
                try:
                    if hasattr(scen_basin_names[0],'decode'):
                        scen_basin_names = [b.decode('utf-8') if hasattr(b,'decode') else str(b) for b in scen_basin_names]
                except Exception:
                    scen_basin_names = [str(i) for i in range(len(scen_basin_names))]
            if FLOOD_UTILS_AVAILABLE and basin_areas is not None:
                arr,_ = convert_flow_units_batch(arr, basin_areas, time_unit='3h')
            temp_arrays.append(arr)
            scen_flows[scen] = {'flow': arr, 'basins': scen_basin_names}
        else:
            print(f"Warning: scenario '{scen}' prediction file missing for epoch={epoch}")
            scen_flows[scen] = {'flow': None, 'basins': None}

    # Align lengths (crop to min)
    lengths = [obs_flow.shape[1]] + [a.shape[1] for a in temp_arrays if a is not None]
    min_len = min(lengths)
    if any(l!=min_len for l in lengths):
        if verbose:
            print(f"[align] Cropping all series to {min_len}")
        obs_flow = obs_flow[:, :min_len]
        if time_array is not None and len(time_array) >= min_len:
            time_array = time_array[:min_len]
        precip_fallback = precip_fallback[:, :min_len]
        for k, meta in scen_flows.items():
            arr = meta['flow']
            if arr is not None and arr.shape[1] >= min_len:
                scen_flows[k]['flow'] = arr[:, :min_len]

    # Flood events
    if FLOOD_UTILS_AVAILABLE and 'find_flood_events_from_labels' in globals():
        print('Detecting flood events using Songliao labels...')
        periods = find_flood_events_from_labels(obs_flow, time_array, basin_names, songliao_path)
    else:
        print('Detecting flood events using threshold method (fallback)...')
        periods = find_flood_events_threshold_based(obs_flow, time_array, basin_names) if 'find_flood_events_threshold_based' in globals() else []
    if not periods:
        print('No flood events detected.')
        return None

    # Output dirs
    normal_dir = base_dir / 'songliao_3h_test_normal'
    epoch_folder = Path(str(epoch)).stem
    plots_dir = normal_dir / f'epoch{epoch_folder}' / 'spatial_sensitivity_plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {plots_dir}')

    all_metrics = []
    for period in periods:
        basin_dir = plots_dir / basin_names[period['basin']].replace(' ','_')
        basin_dir.mkdir(exist_ok=True, parents=True)
        rows = plot_spatial_scenario_event(
            obs_flow, scen_flows, period, basin_names, time_array, basin_dir, epoch, flow_unit, precip_fallback=precip_fallback, verbose=verbose
        )
        all_metrics.extend(rows)

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        df_path = plots_dir / 'spatial_sensitivity_metrics.csv'
        df.to_csv(df_path, index=False)
        # Summary NSE per scenario
        summary_lines = [f'Epoch: {epoch}', f'Total events: {df.event_name.nunique()}', f'Scenarios: {", ".join(scen_flows.keys())}']
        for scen in scen_flows.keys():
            sub = df[df['scenario']==scen]
            if len(sub)==0: continue
            summary_lines.append(f'{scen}: NSE_mean={sub.NSE.mean():.3f} NSE_std={sub.NSE.std():.3f}')
        (plots_dir / 'spatial_sensitivity_summary.txt').write_text('\n'.join(summary_lines), encoding='utf-8')
        print('Saved metrics (NSE) and summary.')
        return df
    print('Finished without metrics.')
    return None

# Notebook-friendly wrapper

def compare_spatial_scenarios(base_dir: str, scenarios='normal,sim1,sim2,sim3', epoch='30', songliao_path='/mnt/data/ClassC/songliaorrevent', verbose=False):
    if isinstance(scenarios, str):
        scenarios = [s.strip() for s in scenarios.split(',') if s.strip()]
    return run_spatial_scenario_comparison(base_dir, scenarios, epoch, songliao_path=songliao_path, verbose=verbose)

# === Editable-parameters main (no argparse) ===

def main():
    # ===== 可修改区域 =====
    base_dir = './results/3d_lstm_gnn'  # 实验根目录 (含 songliao_3h_test_normal 等子目录)
    scenarios = ['normal', 'sim1', 'sim2', 'sim3']     # 情景名称列表
    epoch = 'best_model.pth'                                       # 可换成 'best_model.pth' 等
    songliao_path = '/mnt/data/ClassC/songliaorrevent' # 数据根目录
    verbose = True                                     # 调试输出
    # =====================

    print('=== Spatial Scenario Comparison (manual main) ===')
    print('Base Dir:', base_dir)
    print('Scenarios:', scenarios)
    print('Epoch:', epoch)
    df = run_spatial_scenario_comparison(base_dir, scenarios, epoch, songliao_path=songliao_path, verbose=verbose)
    if df is not None:
        print('\n前5行指标:')
        print(df.head())
    print('\n运行结束，如需重新计算请修改上方参数后再次运行。')

if __name__ == '__main__':
    main()
