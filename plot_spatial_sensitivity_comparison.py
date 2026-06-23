#!/usr/bin/env python3
"""
Spatial sensitivity test results comparison plotting script
Compare observed values with sim1/sim2/sim3 spatial scenario predictions

Author: Assistant
Date: 2025-08-20
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set font and figure styles
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_netcdf_data(file_path):
    """Load NetCDF file data"""
    try:
        ds = xr.open_dataset(file_path)
        
        # Extract data
        data_dict = {}
        for var in ds.data_vars:
            data_dict[var] = ds[var].values
        
        # Extract coordinate information
        if 'time' in ds.coords:
            data_dict['time'] = ds.coords['time'].values
        if 'basin' in ds.coords:
            data_dict['basin'] = ds.coords['basin'].values
            
        return data_dict
        
    except Exception as e:
        print(f"Failed to load file {file_path}: {e}")
        return None

def load_spatial_sensitivity_data(result_dir):
    """Load all spatial sensitivity test data"""
    result_dir = Path(result_dir)
    
    # Define file paths
    files = {
        'obs': result_dir / "epoch60flow_obs.nc",
        'sim1': result_dir / "epoch60flow_pred_sim1.nc", 
        'sim2': result_dir / "epoch60flow_pred_sim2.nc",
        'sim3': result_dir / "epoch60flow_pred_sim3.nc"
    }
    
    # Check if files exist
    missing_files = []
    for name, path in files.items():
        if not path.exists():
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        return None
    
    # Load data
    data = {}
    for name, path in files.items():
        print(f"Loading {name} data: {path}")
        data[name] = load_netcdf_data(path)
        if data[name] is None:
            return None
    
    return data

def find_positive_flow_periods(flow_data, min_duration=24):
    """Find continuous periods with positive flow"""
    
    if flow_data.ndim == 1:
        flow_data = flow_data.reshape(1, -1)
    
    n_basins, n_time = flow_data.shape
    all_periods = []
    
    for basin_idx in range(n_basins):
        basin_flow = flow_data[basin_idx, :]
        
        # Find continuous positive flow periods
        positive_mask = basin_flow > 0
        periods = []
        
        start_idx = None
        for i, is_positive in enumerate(positive_mask):
            if is_positive and start_idx is None:
                start_idx = i
            elif not is_positive and start_idx is not None:
                # End a period
                duration = i - start_idx
                if duration >= min_duration:
                    periods.append({
                        'basin': basin_idx,
                        'start': start_idx,
                        'end': i,
                        'duration': duration
                    })
                start_idx = None
        
        # Handle case where period continues to the end
        if start_idx is not None:
            duration = len(positive_mask) - start_idx
            if duration >= min_duration:
                periods.append({
                    'basin': basin_idx,
                    'start': start_idx,
                    'end': len(positive_mask),
                    'duration': duration
                })
        
        all_periods.extend(periods)
    
    return all_periods

def calculate_metrics(obs, pred):
    """Calculate evaluation metrics"""
    # Remove NaN values
    valid_mask = ~(np.isnan(obs) | np.isnan(pred))
    if valid_mask.sum() < 2:
        return {'error': 'insufficient_data'}
    
    obs_valid = obs[valid_mask]
    pred_valid = pred[valid_mask]
    
    # NSE
    numerator = np.sum((obs_valid - pred_valid) ** 2)
    denominator = np.sum((obs_valid - np.mean(obs_valid)) ** 2)
    nse = 1 - (numerator / denominator) if denominator > 0 else -np.inf
    
    # R²
    r2 = np.corrcoef(obs_valid, pred_valid)[0, 1] ** 2 if len(obs_valid) > 1 else 0
    
    # RMSE
    rmse = np.sqrt(np.mean((obs_valid - pred_valid) ** 2))
    
    # MAE
    mae = np.mean(np.abs(obs_valid - pred_valid))
    
    # Peak comparison
    obs_peak = np.max(obs_valid)
    pred_peak = np.max(pred_valid)
    peak_error = abs(pred_peak - obs_peak) / obs_peak * 100 if obs_peak > 0 else 0
    
    return {
        'NSE': nse,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'obs_peak': obs_peak,
        'pred_peak': pred_peak,
        'peak_error_pct': peak_error
    }

def plot_spatial_comparison_period(obs_flow, sim_flows, period, basin_names, time_array, save_dir):
    """Plot spatial sensitivity comparison for a single period"""
    
    basin_idx = period['basin']
    start_idx = period['start'] 
    end_idx = period['end']
    
    # Extract period data
    time_subset = time_array[start_idx:end_idx] if time_array is not None else range(start_idx, end_idx)
    obs_subset = obs_flow[basin_idx, start_idx:end_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot observations
    ax.plot(time_subset, obs_subset, 'k-', linewidth=2.5, label='Observed', alpha=0.8)
    
    # Plot three scenario predictions
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # blue, green, red
    styles = ['-', '--', '-.']
    labels = ['Sim1 (Upstream Heavy Rain)', 'Sim2 (Midstream Heavy Rain)', 'Sim3 (Downstream Heavy Rain)']
    
    metrics_list = []
    
    for i, (sim_name, (color, style, label)) in enumerate(zip(['sim1', 'sim2', 'sim3'], 
                                                              zip(colors, styles, labels))):
        if sim_name in sim_flows:
            sim_subset = sim_flows[sim_name][basin_idx, start_idx:end_idx]
            ax.plot(time_subset, sim_subset, color=color, linestyle=style, 
                   linewidth=2, label=label, alpha=0.9)
            
            # Calculate metrics
            metrics = calculate_metrics(obs_subset, sim_subset)
            if 'error' not in metrics:
                metrics['scenario'] = sim_name
                metrics_list.append(metrics)
    
    # Set figure properties
    basin_name = basin_names[basin_idx] if basin_names else f"Basin_{basin_idx}"
    ax.set_title(f'{basin_name} - Spatial Sensitivity Test Comparison\n'
                f'Period: {start_idx} - {end_idx} (Duration: {period["duration"]} timesteps)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Flow Rate', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add metrics text box
    if metrics_list:
        metrics_text = "Evaluation Metrics:\n"
        for metrics in metrics_list:
            scenario_name = {'sim1': 'Sim1', 'sim2': 'Sim2', 'sim3': 'Sim3'}[metrics['scenario']]
            metrics_text += f"{scenario_name}: NSE={metrics['NSE']:.3f}, R²={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.2f}\n"
        
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{basin_name}_period_{start_idx}_{end_idx}.png"
    filepath = save_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics_list

def plot_spatial_sensitivity_comparison(data, save_dir):
    """Plot spatial sensitivity comparison analysis"""
    
    save_dir = Path(save_dir)
    plots_dir = save_dir / "spatial_sensitivity_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Extract data
    obs_data = data['obs']
    sim_flows = {
        'sim1': data['sim1']['inflow'] if 'sim1' in data else None,
        'sim2': data['sim2']['inflow'] if 'sim2' in data else None, 
        'sim3': data['sim3']['inflow'] if 'sim3' in data else None
    }
    
    obs_flow = obs_data['inflow']
    time_array = obs_data.get('time')
    basin_names = obs_data.get('basin')
    
    # Process basin names
    if basin_names is not None:
        try:
            if hasattr(basin_names[0], 'decode'):
                basin_names = [name.decode('utf-8') if hasattr(name, 'decode') else str(name) 
                              for name in basin_names]
        except:
            basin_names = [f"Basin_{i}" for i in range(obs_flow.shape[0])]
    else:
        basin_names = [f"Basin_{i}" for i in range(obs_flow.shape[0])]
    
    print(f"Processing spatial sensitivity analysis for {len(basin_names)} basins:")
    for i, name in enumerate(basin_names):
        print(f"  {i}: {name}")
    
    # Find positive flow periods
    periods = find_positive_flow_periods(obs_flow, min_duration=24)
    print(f"\nFound {len(periods)} positive flow periods")
    
    if len(periods) == 0:
        print("No sufficiently long positive flow periods found")
        return
    
    # Plot each period
    all_metrics = []
    
    for period in periods:
        print(f"Plotting basin {basin_names[period['basin']]} period {period['start']}-{period['end']}")
        
        # Create basin subfolder
        basin_name = basin_names[period['basin']].replace(' ', '_')
        basin_dir = plots_dir / basin_name
        basin_dir.mkdir(exist_ok=True)
        
        metrics_list = plot_spatial_comparison_period(
            obs_flow, sim_flows, period, basin_names, time_array, basin_dir
        )
        
        # Add period information to metrics
        for metrics in metrics_list:
            metrics.update({
                'basin': basin_names[period['basin']],
                'period_start': period['start'],
                'period_end': period['end'],
                'duration': period['duration']
            })
            all_metrics.append(metrics)
    
    # Save overall metrics statistics
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        csv_path = plots_dir / "spatial_sensitivity_metrics.csv"
        df.to_csv(csv_path, index=False)
        
        # Statistics by scenario
        print(f"\n=== Spatial Sensitivity Analysis Results ===")
        print(f"Total periods analyzed: {len(all_metrics)}")
        
        scenario_stats = df.groupby('scenario').agg({
            'NSE': ['count', 'mean', 'std'],
            'R2': ['mean', 'std'],
            'RMSE': ['mean', 'std'],
            'peak_error_pct': ['mean', 'std']
        }).round(3)
        
        print("\nAverage performance by scenario:")
        print(scenario_stats)
        
        # Generate summary report
        summary_text = f"""Spatial Sensitivity Test Summary Report
        
Number of periods analyzed: {len(all_metrics)}
Number of basins processed: {len(set(df['basin']))}

Average metrics by scenario:
"""
        for scenario in ['sim1', 'sim2', 'sim3']:
            scenario_data = df[df['scenario'] == scenario]
            if len(scenario_data) > 0:
                avg_nse = scenario_data['NSE'].mean()
                avg_r2 = scenario_data['R2'].mean()
                avg_rmse = scenario_data['RMSE'].mean()
                avg_peak_error = scenario_data['peak_error_pct'].mean()
                
                scenario_names = {'sim1': 'Upstream Heavy Rain', 'sim2': 'Midstream Heavy Rain', 'sim3': 'Downstream Heavy Rain'}
                summary_text += f"""
{scenario.upper()} ({scenario_names[scenario]}):
  NSE: {avg_nse:.3f}
  R²: {avg_r2:.3f} 
  RMSE: {avg_rmse:.3f}
  Peak Error: {avg_peak_error:.1f}%
"""
        
        # Save summary
        summary_path = plots_dir / "spatial_sensitivity_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
    
    print(f"\nAll plots and analysis results saved to: {plots_dir}")

def main():
    """Main function"""
    # Set result directory
    result_dir = Path("/home/lizilin/code/torchhydro/results/gnn_experiment/test_sim")
    
    print("=== Spatial Sensitivity Test Results Comparison Analysis ===")
    print(f"Result directory: {result_dir}")
    
    if not result_dir.exists():
        print(f"Result directory does not exist: {result_dir}")
        return
    
    # Load data
    print("\n1. Loading data...")
    data = load_spatial_sensitivity_data(result_dir)
    
    if data is None:
        print("Data loading failed")
        return
    
    print("Data loaded successfully!")
    
    # Plot comparison analysis
    print("\n2. Plotting spatial sensitivity comparison...")
    plot_spatial_sensitivity_comparison(data, result_dir)
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()
