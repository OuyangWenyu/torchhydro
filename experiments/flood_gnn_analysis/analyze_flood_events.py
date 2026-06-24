#!/usr/bin/env python3
"""
GNN Flood Event Analysis Script
Analyze flood events from GNN model predictions vs observations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def explore_nc_structure(file_path):
    """Explore NetCDF file structure using different methods"""
    import scipy.io.netcdf as netcdf
    
    print(f"\n=== Exploring {file_path} ===")
    
    try:
        # Try scipy.io.netcdf first
        with netcdf.netcdf_file(file_path, 'r') as f:
            print("Variables:", list(f.variables.keys()))
            print("Dimensions:", list(f.dimensions.keys()))
            
            for var_name in f.variables.keys():
                var = f.variables[var_name]
                print(f"{var_name}: shape {var.shape}, dtype {var.dtype}")
                if hasattr(var, 'dimensions'):
                    print(f"  dimensions: {var.dimensions}")
            
            # Get inflow data if exists
            if 'inflow' in f.variables:
                inflow = f.variables['inflow'][:]
                print(f"Inflow data range: [{np.nanmin(inflow):.4f}, {np.nanmax(inflow):.4f}]")
                return inflow, f.variables.get('flood_event', None)
            
    except Exception as e:
        print(f"scipy.io.netcdf failed: {e}")
        
    try:
        # Try h5py as NetCDF4 alternative
        import h5py
        with h5py.File(file_path, 'r') as f:
            print("HDF5 keys:", list(f.keys()))
            
            def explore_group(group, indent=""):
                for key in group.keys():
                    item = group[key]
                    if hasattr(item, 'shape'):
                        print(f"{indent}{key}: shape {item.shape}, dtype {item.dtype}")
                    elif hasattr(item, 'keys'):
                        print(f"{indent}{key}/ (group)")
                        explore_group(item, indent + "  ")
                        
            explore_group(f)
            
    except Exception as e:
        print(f"h5py failed: {e}")
        
    return None, None

def load_data_simple(obs_file, pred_file):
    """Load data using h5py (NetCDF4/HDF5 format)"""
    import h5py
    
    obs_data = {}
    pred_data = {}
    
    try:
        # Load observation data
        with h5py.File(obs_file, 'r') as f:
            for var_name in f.keys():
                obs_data[var_name] = f[var_name][:]
                print(f"Obs {var_name}: shape {obs_data[var_name].shape}")
        
        # Load prediction data  
        with h5py.File(pred_file, 'r') as f:
            for var_name in f.keys():
                pred_data[var_name] = f[var_name][:]
                print(f"Pred {var_name}: shape {pred_data[var_name].shape}")
                
        return obs_data, pred_data
        
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None, None

def identify_flood_events_from_inflow(inflow_array, min_duration=3):
    """Identify continuous flood events from inflow data (when inflow > 0)"""
    if inflow_array is None:
        return []
    
    flood_events = []
    inflow_array = np.array(inflow_array)
    
    # Data is [basin, time] format
    n_basins = inflow_array.shape[0] 
    n_timesteps = inflow_array.shape[1]
    
    print(f"Processing {n_basins} basins with {n_timesteps} timesteps")
    print(f"Identifying flood events where inflow > 0 for at least {min_duration} consecutive timesteps")
    
    for basin in range(n_basins):
        basin_inflow = inflow_array[basin, :]
        basin_events = identify_single_basin_inflow_events(basin_inflow, basin, min_duration)
        flood_events.extend(basin_events)
        print(f"Basin {basin}: found {len(basin_events)} flood events (inflow > 0)")
    
    return flood_events

def identify_single_basin_inflow_events(inflow_series, basin_id, min_duration=3):
    """Identify flood events for a single basin based on inflow > 0"""
    events = []
    inflow_series = np.array(inflow_series).flatten()
    
    in_event = False
    event_start = None
    event_count = 0
    
    for i, inflow_val in enumerate(inflow_series):
        if inflow_val > 0 and not in_event:
            # Start of new flood event (inflow > 0)
            in_event = True
            event_start = i
        elif inflow_val <= 0 and in_event:
            # End of flood event (inflow <= 0)
            in_event = False
            duration = i - event_start
            if duration >= min_duration:  # Only keep events with minimum duration
                event_count += 1
                events.append({
                    'basin_id': basin_id,
                    'event_id': event_count,
                    'start_idx': event_start,
                    'end_idx': i - 1,
                    'duration': duration
                })
    
    # Handle case where flood event continues to the end
    if in_event:
        duration = len(inflow_series) - event_start
        if duration >= min_duration:
            event_count += 1
            events.append({
                'basin_id': basin_id,
                'event_id': event_count,
                'start_idx': event_start,
                'end_idx': len(inflow_series) - 1,
                'duration': duration
            })
    
    return events

def calculate_metrics(obs, pred):
    """Calculate evaluation metrics"""
    obs = np.array(obs).flatten()
    pred = np.array(pred).flatten()
    
    # Remove NaN values
    valid_mask = ~(np.isnan(obs) | np.isnan(pred))
    obs_valid = obs[valid_mask]
    pred_valid = pred[valid_mask]
    
    if len(obs_valid) == 0:
        return {"error": "No valid data points"}
    
    metrics = {}
    
    # NSE
    obs_mean = np.mean(obs_valid)
    sse = np.sum((obs_valid - pred_valid) ** 2)
    sst = np.sum((obs_valid - obs_mean) ** 2)
    metrics['NSE'] = 1 - (sse / sst) if sst != 0 else float('-inf')
    
    # R²
    corr_matrix = np.corrcoef(obs_valid, pred_valid)
    metrics['R2'] = corr_matrix[0, 1] ** 2 if not np.isnan(corr_matrix[0, 1]) else 0
    
    # RMSE
    metrics['RMSE'] = np.sqrt(np.mean((obs_valid - pred_valid) ** 2))
    
    # MAE
    metrics['MAE'] = np.mean(np.abs(obs_valid - pred_valid))
    
    return metrics

def plot_flood_events(obs_data, pred_data, flood_events, save_dir=None, period_suffix=''):
    """Plot flood events comparison"""
    
    # Extract inflow data - format is [basin, time]
    obs_inflow = obs_data.get('inflow')
    pred_inflow = pred_data.get('inflow')
    basin_names = obs_data.get('basin', None)
    
    if obs_inflow is None or pred_inflow is None:
        print("No inflow data found!")
        return
    
    print(f"Obs inflow shape: {obs_inflow.shape}")
    print(f"Pred inflow shape: {pred_inflow.shape}")
    
    # Decode basin names if they are byte strings
    if basin_names is not None:
        try:
            basin_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) 
                          for name in basin_names]
            print(f"Basin names: {basin_names}")
        except:
            basin_names = [f"Basin_{i}" for i in range(len(basin_names))]
    else:
        basin_names = [f"Basin_{i}" for i in range(obs_inflow.shape[0])]
    
    # Group events by basin
    basin_events = {}
    for event in flood_events:
        basin_id = event['basin_id']
        if basin_id not in basin_events:
            basin_events[basin_id] = []
        basin_events[basin_id].append(event)

    print(f"Found {len(basin_events)} basins with flood events")

    # For each basin, plot up to 5 events (if available)
    for basin_id, events in basin_events.items():
        if len(events) == 0:
            continue

        basin_name = basin_names[basin_id] if basin_id < len(basin_names) else f"Basin_{basin_id}"
        print(f"\n{basin_name}: {len(events)} flood events")

        n_events = min(len(events), 5)
        events_to_plot = events[:n_events]
        if n_events == 0:
            continue

        fig, axes = plt.subplots(n_events, 1, figsize=(15, 4*n_events))
        if n_events == 1:
            axes = [axes]

        fig.suptitle(f'{basin_name} Flood Events Comparison ({period_suffix.replace("_", " ").strip()})', fontsize=14, fontweight='bold')

        overall_metrics = {'NSE': [], 'R2': [], 'RMSE': [], 'MAE': []}

        for i, event in enumerate(events_to_plot):
            start_idx = event['start_idx']
            end_idx = event['end_idx']
            event_id = event['event_id']

            obs_event = obs_inflow[basin_id, start_idx:end_idx+1]
            pred_event = pred_inflow[basin_id, start_idx:end_idx+1]

            metrics = calculate_metrics(obs_event, pred_event)

            if 'error' not in metrics:
                for key in overall_metrics:
                    if key in metrics:
                        overall_metrics[key].append(metrics[key])

                ax = axes[i]
                time_steps = range(len(obs_event))
                ax.plot(time_steps, obs_event, 'b-', linewidth=2, label='Observed', alpha=0.8)
                ax.plot(time_steps, pred_event, 'r--', linewidth=2, label='Predicted', alpha=0.8)
                ax.set_title(f'Event {event_id} (Duration: {event["duration"]} steps) - '
                             f'NSE: {metrics["NSE"]:.3f}, R²: {metrics["R2"]:.3f}')
                ax.set_xlabel('Time Steps (3-hour intervals)')
                ax.set_ylabel('Inflow')
                ax.legend()
                ax.grid(True, alpha=0.3)
                obs_max = np.max(obs_event)
                pred_max = np.max(pred_event)
                ax.text(0.02, 0.98, f'Obs Max: {obs_max:.2f}\nPred Max: {pred_max:.2f}',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        # Save plot for this basin and period
        if save_dir:
            save_path = Path(save_dir) / f'{basin_name.replace(" ", "_")}_flood_events{period_suffix}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"{basin_name} plot saved to: {save_path}")
        plt.close(fig)

        # Print overall metrics for this basin
        if any(len(v) > 0 for v in overall_metrics.values()):
            print(f"\n{basin_name} Overall Metrics ({len(events)} events):")
            for metric_name, values in overall_metrics.items():
                if len(values) > 0:
                    print(f"  {metric_name}: mean={np.mean(values):.4f} ± {np.std(values):.4f}")

    # Create and save summary plot for this period
    create_summary_plot(basin_events, obs_data, pred_data, basin_names, save_dir, period_suffix)

def create_summary_plot(basin_events, obs_data, pred_data, basin_names, save_dir=None, period_suffix=''):
    """Create a summary plot showing metrics across all basins"""
    
    # Collect metrics for all basins
    all_metrics = {}
    basin_summary = []
    
    obs_inflow = obs_data.get('inflow')
    pred_inflow = pred_data.get('inflow')
    
    for basin_id, events in basin_events.items():
        if len(events) == 0:
            continue
        basin_name = basin_names[basin_id] if basin_id < len(basin_names) else f"Basin_{basin_id}"
        # Ensure basin_name is str, not bytes
        if isinstance(basin_name, bytes):
            basin_name = basin_name.decode('utf-8')
        basin_metrics = {'NSE': [], 'R2': [], 'RMSE': [], 'MAE': []}
        for event in events:
            start_idx = event['start_idx']
            end_idx = event['end_idx']
            obs_event = obs_inflow[basin_id, start_idx:end_idx+1]
            pred_event = pred_inflow[basin_id, start_idx:end_idx+1]
            metrics = calculate_metrics(obs_event, pred_event)
            if 'error' not in metrics:
                for key in basin_metrics:
                    if key in metrics:
                        basin_metrics[key].append(metrics[key])
        # Calculate mean metrics for this basin
        basin_summary.append({
            'basin_name': basin_name,
            'basin_id': basin_id,
            'n_events': len(events),
            'NSE_mean': np.mean(basin_metrics['NSE']) if basin_metrics['NSE'] else 0,
            'R2_mean': np.mean(basin_metrics['R2']) if basin_metrics['R2'] else 0,
            'RMSE_mean': np.mean(basin_metrics['RMSE']) if basin_metrics['RMSE'] else 0,
            'MAE_mean': np.mean(basin_metrics['MAE']) if basin_metrics['MAE'] else 0,
        })
    
    if not basin_summary:
        print("No valid basin summary data to plot")
        return
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Summary Across All Basins', fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    basin_names_plot = [item['basin_name'] for item in basin_summary]
    nse_values = [item['NSE_mean'] for item in basin_summary]
    r2_values = [item['R2_mean'] for item in basin_summary]
    rmse_values = [item['RMSE_mean'] for item in basin_summary]
    n_events = [item['n_events'] for item in basin_summary]
    
    # NSE plot
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(basin_names_plot)), nse_values, alpha=0.7, color='skyblue')
    ax1.set_title('Nash-Sutcliffe Efficiency (NSE)')
    ax1.set_xlabel('Basin')
    ax1.set_ylabel('NSE')
    ax1.set_xticks(range(len(basin_names_plot)))
    ax1.set_xticklabels(basin_names_plot, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='NSE=0.5')
    ax1.legend()
    
    # Add value labels on bars
    for i, v in enumerate(nse_values):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # R² plot
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(basin_names_plot)), r2_values, alpha=0.7, color='lightgreen')
    ax2.set_title('Coefficient of Determination (R²)')
    ax2.set_xlabel('Basin')
    ax2.set_ylabel('R²')
    ax2.set_xticks(range(len(basin_names_plot)))
    ax2.set_xticklabels(basin_names_plot, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    for i, v in enumerate(r2_values):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # RMSE plot
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(basin_names_plot)), rmse_values, alpha=0.7, color='salmon')
    ax3.set_title('Root Mean Square Error (RMSE)')
    ax3.set_xlabel('Basin')
    ax3.set_ylabel('RMSE')
    ax3.set_xticks(range(len(basin_names_plot)))
    ax3.set_xticklabels(basin_names_plot, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    for i, v in enumerate(rmse_values):
        ax3.text(i, v + max(rmse_values)*0.01, f'{v:.2f}', ha='center', va='bottom')
    
    # Number of events plot
    ax4 = axes[1, 1]
    bars4 = ax4.bar(range(len(basin_names_plot)), n_events, alpha=0.7, color='gold')
    ax4.set_title('Number of Flood Events')
    ax4.set_xlabel('Basin')
    ax4.set_ylabel('Count')
    ax4.set_xticks(range(len(basin_names_plot)))
    ax4.set_xticklabels(basin_names_plot, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    for i, v in enumerate(n_events):
        ax4.text(i, v + 0.5, f'{v}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot with period-specific filename
    if save_dir:
        summary_name = f'model_performance_summary{period_suffix}.png' if period_suffix else 'model_performance_summary.png'
        save_path = Path(save_dir) / summary_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary plot saved to: {save_path}")
    plt.close(fig)
    
    # Print summary table
    print("\n=== Model Performance Summary ===")
    print(f"{'Basin':<15} {'Events':<8} {'NSE':<8} {'R²':<8} {'RMSE':<8}")
    print("-" * 55)
    for item in basin_summary:
        print(f"{item['basin_name']:<15} {item['n_events']:<8} "
              f"{item['NSE_mean']:<8.3f} {item['R2_mean']:<8.3f} {item['RMSE_mean']:<8.2f}")

def analyze_period_floods(obs_data, pred_data, period_name, start_ratio, end_ratio):
    """
    Analyze flood events for a specific time period based on inflow > 0
    
    Based on training config:
    - Train period: 1980-2012 (33 years) = 0.0 to ~0.73
    - Valid period: 2013-2019 (7 years) = ~0.73 to ~0.89  
    - Test period: 2020-2024 (5 years) = ~0.89 to 1.0
    
    Args:
        period_name: 'train', 'valid', or 'test'
        start_ratio: start of period as fraction of total time
        end_ratio: end of period as fraction of total time
    """
    obs_inflow = obs_data.get('inflow')
    pred_inflow = pred_data.get('inflow')
    
    if obs_inflow is None or pred_inflow is None:
        print("Missing required data (inflow)")
        return None
    
    # Data shape is [basin, time]
    n_basins, n_timesteps = obs_inflow.shape
    start_idx = int(n_timesteps * start_ratio)
    end_idx = int(n_timesteps * end_ratio)
    
    print(f"\n=== {period_name.upper()} Period Analysis ===")
    print(f"Total timesteps: {n_timesteps}")
    print(f"{period_name} period: timesteps {start_idx} to {end_idx}")
    print(f"{period_name} period length: {end_idx - start_idx} timesteps")
    
    # Extract period data
    period_obs = obs_inflow[:, start_idx:end_idx]
    period_pred = pred_inflow[:, start_idx:end_idx]
    
    # Create modified data dict for this period only
    period_obs_data = {
        'inflow': period_obs,
        'basin': obs_data.get('basin', None)
    }
    
    period_pred_data = {
        'inflow': period_pred,
        'basin': pred_data.get('basin', None)
    }
    
    # Identify flood events in this period based on inflow > 0
    flood_events = identify_flood_events_from_inflow(period_obs)
    
    # Adjust event indices to account for period offset
    for event in flood_events:
        event['global_start_idx'] = event['start_idx'] + start_idx
        event['global_end_idx'] = event['end_idx'] + start_idx
        event['period'] = period_name
    
    print(f"Found {len(flood_events)} flood events in {period_name} period")
    
    return period_obs_data, period_pred_data, flood_events

def get_time_period_ratios():
    """
    Get time period ratios based on training configuration
    
    Training config periods:
    - Train: 1980-01-01-02 to 2012-12-31-23 (33 years)
    - Valid: 2013-01-01-02 to 2019-12-31-23 (7 years)  
    - Test:  2020-01-01-02 to 2024-12-30-23 (5 years)
    Total: 45 years (1980-2024)
    """
    # Calculate exact time boundaries
    from datetime import datetime
    
    # Define exact periods from training config
    train_start = datetime(1980, 1, 1)
    train_end = datetime(2012, 12, 31)
    valid_start = datetime(2013, 1, 1)
    valid_end = datetime(2019, 12, 31)
    test_start = datetime(2020, 1, 1)
    test_end = datetime(2024, 12, 30)
    
    # Total time span
    total_start = train_start
    total_end = test_end
    total_days = (total_end - total_start).days
    
    # Calculate ratios based on actual time spans
    train_days = (train_end - train_start).days
    valid_days = (valid_end - valid_start).days
    test_days = (test_end - test_start).days
    
    train_ratio_end = train_days / total_days
    valid_ratio_end = (train_days + valid_days) / total_days
    
    print(f"Period calculations:")
    print(f"  Train: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')} ({train_days} days, {train_days/365.25:.1f} years)")
    print(f"  Valid: {valid_start.strftime('%Y-%m-%d')} to {valid_end.strftime('%Y-%m-%d')} ({valid_days} days, {valid_days/365.25:.1f} years)")
    print(f"  Test:  {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')} ({test_days} days, {test_days/365.25:.1f} years)")
    print(f"  Total: {total_start.strftime('%Y-%m-%d')} to {total_end.strftime('%Y-%m-%d')} ({total_days} days, {total_days/365.25:.1f} years)")
    print(f"  Ratios: train=0.0-{train_ratio_end:.3f}, valid={train_ratio_end:.3f}-{valid_ratio_end:.3f}, test={valid_ratio_end:.3f}-1.0")
    
    return {
        'train': (0.0, train_ratio_end),
        'valid': (train_ratio_end, valid_ratio_end), 
        'test': (valid_ratio_end, 1.0)
    }

def main():
    """Main analysis function"""
    # Set up paths
    result_dir = "/home/lizilin/code/torchhydro/results/gnn_experiment/songliao_3h_test_"
    obs_file = Path(result_dir) / "epoch60flow_obs.nc"
    pred_file = Path(result_dir) / "epoch60flow_pred.nc"
    # result_dir = "/home/lizilin/code/torchhydro/results/events_train_3h/regional_weighted_mse_loss"
    # obs_file = Path(result_dir) / "epoch60flow_obs.nc"
    # pred_file = Path(result_dir) / "epoch60flow_pred.nc"
    
    print("=== GNN Flood Event Analysis (Test Period Focus) ===")
    print(f"Observation file: {obs_file}")
    print(f"Prediction file: {pred_file}")
    
    if not obs_file.exists():
        print(f"Observation file not found: {obs_file}")
        return
    
    if not pred_file.exists():
        print(f"Prediction file not found: {pred_file}")
        return
    
    # Load data
    print("\n1. Loading data...")
    obs_data, pred_data = load_data_simple(obs_file, pred_file)
    
    if obs_data is None or pred_data is None:
        print("Failed to load data")
        return
    
    print(f"Loaded data successfully!")
    print(f"Observation variables: {list(obs_data.keys())}")
    print(f"Prediction variables: {list(pred_data.keys())}")
    
    # Analyze full dataset first
    print("\n2. Analyzing full dataset flood events...")
    obs_inflow = obs_data.get('inflow')
    if obs_inflow is not None:
        all_flood_events = identify_flood_events_from_inflow(obs_inflow)
        print(f"Total flood events in full dataset: {len(all_flood_events)}")
        
        # Show distribution by basin
        basin_event_counts = {}
        for event in all_flood_events:
            basin_id = event['basin_id']
            basin_event_counts[basin_id] = basin_event_counts.get(basin_id, 0) + 1
        
        print("Flood events per basin (full dataset):")
        for basin_id, count in sorted(basin_event_counts.items()):
            print(f"  Basin {basin_id}: {count} events")
    
    # Analyze different periods based on training configuration
    print("\n3. Analyzing periods based on training config...")
    period_ratios = get_time_period_ratios()
    
    # Analyze and plot for each period: train_recent, valid, test
    period_plot_settings = [
        ('train_recent', period_ratios['train'][1] - 5/45, period_ratios['train'][1], '_train'),
        ('valid', period_ratios['valid'][0], period_ratios['valid'][1], '_valid'),
        ('test', period_ratios['test'][0], period_ratios['test'][1], '_test'),
    ]

    all_results = []
    for period_name, start_ratio, end_ratio, period_suffix in period_plot_settings:
        print(f"\n--- {period_name.capitalize()} Period ---")
        result = analyze_period_floods(obs_data, pred_data, period_name, start_ratio, end_ratio)
        if result is not None:
            obs_data_p, pred_data_p, flood_events_p = result
            if len(flood_events_p) > 0:
                print(f"\nVisualizing {period_name} period floods...")
                print(f"{period_name.capitalize()} period flood events: {len(flood_events_p)}")
                # Plot up to 5 events per basin (if available)
                # Group by basin
                basin_events = {}
                for event in flood_events_p:
                    basin_id = event['basin_id']
                    if basin_id not in basin_events:
                        basin_events[basin_id] = []
                    basin_events[basin_id].append(event)
                # For each basin, plot up to 5 events
                events_to_plot = []
                for basin_id, events in basin_events.items():
                    events_to_plot.extend(events[:5])
                plot_flood_events(obs_data_p, pred_data_p, events_to_plot, save_dir=result_dir, period_suffix=period_suffix)
                # Also save summary plot for this period
                # Fix: avoid ambiguous truth value for numpy arrays
                basin_names = obs_data_p.get('basin', None)
                if basin_names is None:
                    basin_names = pred_data_p.get('basin', None)
                # If still None, will be handled in create_summary_plot
                create_summary_plot(basin_events, obs_data_p, pred_data_p, basin_names,
                                   save_dir=result_dir, period_suffix=period_suffix)
                all_results.append((period_name, flood_events_p))
            else:
                print(f"No flood events found in {period_name} period!")

if __name__ == "__main__":
    main()
