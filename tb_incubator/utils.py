from jax import numpy as jnp
from math import log, exp
from typing import Dict, List
import numpy as np
import pandas as pd

def create_periodic_time_series(baseline_year, rate, frequency, baseline_rate=0.0):
    time_series = {}
    baseline_year = baseline_year
    while baseline_year < 2050:
        time_series[baseline_year] = baseline_rate
        time_series[baseline_year + 0.01] = rate
        time_series[baseline_year + 1.0] = rate
        time_series[baseline_year + 1.01] = baseline_rate
        baseline_year += frequency

    return time_series

def calculate_proportional_reduction(
    scenario_outputs: Dict[str, Dict[str, pd.DataFrame]],
    indicator: str,
    baseline_year: int,
    target_year: int
) -> pd.DataFrame:
    """
    Calculate the proportional reduction of an indicator between baseline and target years.

    Args:
        scenario_outputs: Dictionary of scenario outputs (from your combined detection function).
        indicator: Indicator to calculate reduction for (e.g. 'mortality', 'incidence').
        baseline_year: Baseline year to compare from (e.g. 2015).
        target_year: Target year to compare to (e.g. 2050).

    Returns:
        DataFrame with scenario and % reduction in the indicator.
    """
    reduction_results = []

    for scenario_name, outputs in scenario_outputs.items():
        # Some scenarios may not include this indicator (defensive check)
        if indicator not in outputs:
            print(f"{scenario_name}: missing indicator {indicator}")
            continue

        df = outputs[indicator]

        # Check if both years are present
        if baseline_year in df.index and target_year in df.index:
            baseline_value = df.loc[baseline_year, 0.5]  # median
            target_value = df.loc[target_year, 0.5]       # median

            proportional_reduction = (baseline_value - target_value) / baseline_value
            percent_reduction = proportional_reduction * 100

            reduction_results.append({
                "scenario": scenario_name,
                f"percent_reduction_{indicator}_{target_year}": percent_reduction
            })
        else:
            print(f"{scenario_name}: missing data for {baseline_year} or {target_year}")

    # Convert to DataFrame
    reduction_df = pd.DataFrame(reduction_results)

    # Sort by reduction for nice viewing
    column_name = f"percent_reduction_{indicator}_{target_year}"
    reduction_df = reduction_df.sort_values(by=column_name, ascending=False)

    return reduction_df


def dict_to_markdown_table(data_dict, decimal_places=4):
    # Create table header
    markdown_table = "| Parameter | Value |\n|-----------|-------|\n"
    
    # Add each row
    for key, value in data_dict.items():
        # Round floating point values to specified decimal places
        if isinstance(value, float):
            value = round(value, decimal_places)
        markdown_table += f"| {key} | {value} |\n"
    
    return markdown_table

def get_param_table(param_info, prior_names=None):
    """
    Get parameter info in a tidy pd.Dataframe format.
    
    Args:
        param_info: Dictionary containing parameter information
        prior_names: List of parameter names that should be marked as "Calibrated"
                    rather than showing their values
    
    Returns:
        pd.DataFrame: Formatted parameter table
    """
    if prior_names is None:
        prior_names = []
        
    param_table = []
    for key in param_info["value"]:
        if isinstance(param_info["value"][key], dict):
            if key in param_info["unit"]:
                for subkey, value in param_info["value"][key].items():
                    # Create full parameter name for checking calibration status
                    param_name = f"{key}.{subkey}"
                    
                    # Format value based on whether it's calibrated
                    if param_name in prior_names:
                        value_str = "Calibrated"
                    else:
                        # Handle dictionary values
                        if isinstance(value, dict):
                            value_str = "/".join(f"{k}: {v}" for k, v in value.items())
                        else:
                            # Round numerical values
                            value_str = str(round(value, 3) if value != 0.0 else 0.0)
                    
                    param_table.append(
                        {
                            "Parameter": f"{param_info['descriptions'][key][subkey]}",
                            "Value": value_str,
                            "Unit": param_info["unit"][key][subkey],
                            "Source": param_info["sources"][key],
                        }
                    )
        else:
            # Check if this parameter is calibrated
            if key in prior_names:
                value_str = "Calibrated"
            else:
                # Round numerical values
                value = param_info["value"][key]
                value_str = str(round(value, 3) if value != 0.0 else 0.0)
            
            param_table.append(
                {
                    "Parameter": param_info["descriptions"][key],
                    "Value": value_str,
                    "Unit": param_info["unit"][key],
                    "Source": param_info["sources"][key],
                }
            )

    # Create DataFrame and set index
    fixed_param_table = pd.DataFrame(param_table)
    fixed_param_table = fixed_param_table.set_index("Parameter")
    
    # Format column names
    fixed_param_table.columns = fixed_param_table.columns.str.capitalize()
    
    # Reorder columns to match the second example if desired
    fixed_param_table = fixed_param_table[["Value", "Unit", "Source"]]
    
    # Capitalize units
    fixed_param_table["Unit"] = fixed_param_table["Unit"].str.capitalize()

    return fixed_param_table


def get_row_col_for_subplots(i_panel, n_cols):
    return int(np.floor(i_panel / n_cols)) + 1, i_panel % n_cols + 1

def get_next_run_number(out_path, num_priors):
    """
    Find the next available run number for a specific number of priors.
    Returns a formatted string like '01', '02', etc.
    """
    # Look for files matching the pattern with the specific number of priors
    pattern = f'calib_full_out_p{num_priors:02d}_*.nc'
    existing_files = list(out_path.glob(pattern))
    
    if not existing_files:
        return '01'
    
    # Extract run numbers from existing files
    numbers = []
    for f in existing_files:
        # Split filename and get the last part after 'p{num_priors}_'
        try:
            run_num = int(f.stem.split('_')[-1])
            numbers.append(run_num)
        except ValueError:
            continue
    
    next_num = max(numbers) + 1 if numbers else 1
    return f'{next_num:02d}'

def get_target_from_name(
    targets: list,
    name: str,
) -> pd.Series:
    """Get the data for a specific target from a set of targets from its name.

    Args:
        targets: All the targets
        name: The name of the desired target

    Returns:
        Single target to identify
    """
    return next((t.data for t in targets if t.name == name), None)

def round_sigfig(value: float, sig_figs: int) -> float:
    """
    Round a number to a certain number of significant figures,
    rather than decimal places.

    Args:
        value: Number to round
        sig_figs: Number of significant figures to round to
    """
    if np.isinf(value):
        return "infinity"
    else:
        return (
            round(value, -int(np.floor(np.log10(value))) + (sig_figs - 1)) if value != 0.0 else 0.0
        )
    
def tanh_based_scaleup(t, shape, inflection_time, start_asymptote, end_asymptote=1.0):
    """
    return the function t: (1 - sigma) / 2 * tanh(b * (a - c)) + (1 + sigma) / 2
    :param shape: shape parameter
    :param inflection_time: inflection point
    :param start_asymptote: lowest asymptotic value
    :param end_asymptote: highest asymptotic value
    :return: a function
    """
    rng = end_asymptote - start_asymptote
    return (jnp.tanh(shape * (t - inflection_time)) / 2.0 + 0.5) * rng + start_asymptote


def triangle_wave_func(
    time: float,
    start: float,
    duration: float,
    peak: float,
) -> float:
    """Generate a peaked triangular wave function
    that starts from and returns to zero.

    Args:
        time: Model time
        start: Time at which wave starts
        duration: Duration of wave
        peak: Peak flow rate for wave

    Returns:
        The wave function
    """
    gradient = peak / (duration * 0.5)
    peak_time = start + duration * 0.5
    time_from_peak = jnp.abs(peak_time - time)
    return jnp.where(time_from_peak < duration * 0.5, peak - time_from_peak * gradient, 0.0)


def get_average_sigmoid(
    low_val, upper_val, inflection
):  # Long's code # Ragonnet, R., et al. (2019)
    """
    A sigmoidal function (x -> 1 / (1 + exp(-(x-alpha)))) is used to model a progressive increase with age.
    """
    return (log(1.0 + exp(upper_val - inflection)) - log(1.0 + exp(low_val - inflection))) / (
        upper_val - low_val
    )


