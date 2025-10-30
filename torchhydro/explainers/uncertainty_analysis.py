import numpy as np
import matplotlib.pyplot as plt


def calculate_empirical_cdf(predictions, obs_values):
    """
    Calculate the empirical cumulative distribution function (CDF) of model predictions for each time step
    and compute the quantiles z_i of the observed values.

    Parameters
    ----------
    predictions: np.ndarray
        2D array of predictions (shape = (MC_dropout_times, time_steps))
    obs_values: np.ndarray
        Time series of observed values (1D array, shape = (time_steps,))

    Returns
    -------
    z_values: np.ndarray
        Quantiles of the observed values in the prediction distribution for each time step (1D array)
    """
    time_steps = obs_values.shape[0]
    mc_dropout_times = predictions.shape[0]

    z_values = np.zeros(time_steps)

    for t in range(time_steps):
        # Sort the predictions for each time step
        sorted_predictions = np.sort(predictions[:, t])

        # Calculate the quantile of the observed value in the prediction distribution
        z_values[t] = np.sum(sorted_predictions <= obs_values[t]) / mc_dropout_times

    return z_values


def calculate_observed_ecdf(obs_values):
    """
    Calculate the empirical cumulative distribution function (ECDF) of observed values.

    Parameters
    ----------
    obs_values: np.ndarray
        Time series of observed values (1D array, shape = (time_steps,))

    Returns
    -------
    ecdf: np.ndarray
        Proportion of observed values in the cumulative distribution (1D array, same shape as obs_values)
    """
    time_steps = obs_values.shape[0]

    # Sort observed values
    sorted_obs = np.sort(obs_values)

    # Calculate ECDF: for each observed value, the proportion of values <= that value
    ecdf = np.zeros(time_steps)
    for i in range(time_steps):
        ecdf[i] = np.sum(sorted_obs <= obs_values[i]) / time_steps

    return ecdf


def bin_aggregated_data(z_values, r_values, num_bins=10):
    """
    Aggregate the z-values and r-values into bins, and calculate the average for each bin.

    Parameters
    ----------
    z_values: np.ndarray
        Aggregated z-values (quantiles) across all basins and time steps
    r_values: np.ndarray
        Aggregated r-values (ECDF) across all basins and time steps
    num_bins: int
        Number of bins to divide the data into (default is 10)

    Returns
    -------
    binned_z: np.ndarray
        Average z-values for each bin
    binned_r: np.ndarray
        Average r-values for each bin
    """
    bins = np.linspace(0, 1, num_bins + 1)  # Create bin edges from 0 to 1
    bin_indices = np.digitize(
        z_values, bins
    )  # Find out which bin each z-value belongs to

    binned_z = np.zeros(num_bins)
    binned_r = np.zeros(num_bins)

    for i in range(1, num_bins + 1):
        # Select the z and r values that fall into the current bin
        in_bin = bin_indices == i

        if np.sum(in_bin) > 0:
            # Calculate the average z and r values for this bin
            binned_z[i - 1] = np.mean(z_values[in_bin])
            binned_r[i - 1] = np.mean(r_values[in_bin])
        else:
            # If no values in this bin, assign NaN (can also handle this differently if needed)
            binned_z[i - 1] = np.nan
            binned_r[i - 1] = np.nan

    return binned_z, binned_r


def plot_probability_plot(z_values, r_values, basin_name="Basin", scatter=True):
    """
    Plot the probability plot for a single basin.

    Parameters
    ----------
    z_values: np.ndarray
        Quantiles of the observed values for each time step (1D array)
    r_values: np.ndarray
        Empirical cumulative distribution of observed values (ECDF, 1D array)
    basin_name: str
        Name of the basin (for title of the plot)
    """
    # Sort both z_values and r_values by z_values
    sorted_indices = np.argsort(z_values)
    z_sorted = z_values[sorted_indices]
    r_sorted = r_values[sorted_indices]

    # Plot the probability plot with markers only (no lines)
    if scatter:
        plt.scatter(
            z_sorted, r_sorted, label=f"Probability Plot ({basin_name})", color="b"
        )
    else:
        plt.plot(
            z_sorted, r_sorted, label=f"Probability Plot ({basin_name})", color="b"
        )
    plt.plot([0, 1], [0, 1], "k--", label="1:1 Line")

    # Add labels and grid
    plt.xlabel("Predicted CDF Quantiles (z_i)")
    plt.ylabel("Observed ECDF (R_i/n)")
    plt.title(f"Probability Plot - {basin_name}")
    plt.legend()
    plt.grid(True)
    plt.show()


def process_basin(predictions, obs_values, basin_name="Basin"):
    """
    Process a single basin: calculate z-values, ECDF, and plot the probability plot.

    Parameters
    ----------
    predictions: np.ndarray
        2D array of predictions for the basin (shape = (MC_dropout_times, time_steps))
    obs_values: np.ndarray
        Time series of observed values for the basin (1D array, shape = (time_steps,))
    basin_name: str
        Name of the basin (for title of the plot)
    """
    # Calculate z-values (quantiles of the observed values in prediction distribution)
    z_values = calculate_empirical_cdf(predictions, obs_values)

    # Calculate observed ECDF
    r_values = calculate_observed_ecdf(obs_values)

    # Plot the probability plot for the basin
    plot_probability_plot(z_values, r_values, basin_name)


def process_multiple_basins(basins_data):
    """
    Process multiple basins: for each basin, process and plot its probability plot.

    Parameters
    ----------
    basins_data: list of dict
        List of dictionaries, each containing 'predictions', 'obs_values', and 'name' for a basin.
    """
    for basin_data in basins_data:
        predictions = basin_data["predictions"]
        obs_values = basin_data["obs_values"]
        basin_name = basin_data["name"]
        process_basin(predictions, obs_values, basin_name)


def process_and_aggregate_basins(basins_data, num_bins=0):
    """
    Process multiple basins: aggregate z-values and r-values across basins and time steps.

    Parameters
    ----------
    basins_data: list of dict
        List of dictionaries, each containing 'predictions', 'obs_values', and 'name' for a basin.
    num_bins: int
        Number of bins to divide the data into (default is 0, which does not bin the data)

    Returns
    -------
    all_z_values: np.ndarray
        Aggregated z-values (quantiles) across all basins and time steps
    all_r_values: np.ndarray
        Aggregated r-values (ECDF) across all basins and time steps
    """
    all_z_values = []
    all_r_values = []

    for basin_data in basins_data:
        predictions = basin_data["predictions"]
        obs_values = basin_data["obs_values"]

        # Calculate z-values and r-values for the current basin
        z_values = calculate_empirical_cdf(predictions, obs_values)
        r_values = calculate_observed_ecdf(obs_values)

        # Append the results to the global list
        all_z_values.extend(z_values)
        all_r_values.extend(r_values)

    # Convert lists to numpy arrays
    all_z_values = np.array(all_z_values)
    all_r_values = np.array(all_r_values)

    # Optionally bin the data
    if num_bins > 0:
        all_z_values, all_r_values = bin_aggregated_data(
            all_z_values, all_r_values, num_bins
        )
    return all_z_values, all_r_values


# Example usage with multiple basins
if __name__ == "__main__":
    np.random.seed(1234)

    # Assume we have 3 basins, each with 100 time steps and 50 Monte Carlo Dropout evaluations
    num_basins = 3
    time_steps = 100
    mc_dropout_times = 50

    basins_data = []

    # for i in range(num_basins):
    #     # Randomly generate observed values and predictions for each basin
    #     obs_values = np.random.rand(time_steps)
    #     predictions = np.random.rand(mc_dropout_times, time_steps)
    #     basin_name = f"Basin {i+1}"

    #     # Store the data in the list
    #     basins_data.append(
    #         {"predictions": predictions, "obs_values": obs_values, "name": basin_name}
    #     )

    # # Process and plot for all basins
    # process_multiple_basins(basins_data)
    for i in range(num_basins):
        # Randomly generate observed values and predictions for each basin
        obs_values = np.random.rand(time_steps)
        predictions = np.random.rand(mc_dropout_times, time_steps)
        basin_name = f"Basin {i+1}"

        # Store the data in the list
        basins_data.append(
            {"predictions": predictions, "obs_values": obs_values, "name": basin_name}
        )

    # Aggregate z-values and r-values over basins and time steps
    all_z_values, all_r_values = process_and_aggregate_basins(basins_data, num_bins=10)

    # Plot the aggregated probability plot
    plot_probability_plot(all_z_values, all_r_values, basin_name="All Basins", scatter=False)
