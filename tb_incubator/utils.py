from jax import numpy as jnp
from math import log, exp
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from tb_incubator.constants import project_path, PARAMETER_GROUPS, PARAMETER_SECTIONS, ImplementCDR
import yaml as yml

def calculate_treatment_outcomes(
    duration: float, prop_death_among_non_success: float, natural_death_rate: float, tsr
) -> Tuple:
    """
    Computes adjusted treatment outcome proportions over a given duration.

    Args:
        duration (float): Treatment duration in the same time units as natural_death_rate.
        prop_death_among_non_success (float): Proportion of non-successful cases resulting in death (excluding natural deaths).
        natural_death_rate (float): Annual natural death rate for those under treatment.
        tsr (float): Treatment success rate.

    Returns:
        tuple of float: Adjusted proportions of treatment success, deaths from treatment (≥0), and relapse.

    Notes:
        - Accounts for natural deaths using exponential decay.
        - Ensures treatment-related deaths are non-negative.
    """
    # Calculate the proportion of people dying from natural causes while on treatment
    prop_natural_death_while_on_treatment = 1.0 - jnp.exp(
        -duration * natural_death_rate
    )

    # Calculate the target proportion of treatment outcomes resulting in death based on requests
    requested_prop_death_on_treatment = (1.0 - tsr) * prop_death_among_non_success

    # Calculate the actual rate of deaths on treatment, with floor of zero
    prop_death_from_treatment = jnp.max(
        jnp.array(
            (
                requested_prop_death_on_treatment
                - prop_natural_death_while_on_treatment,
                0.0,
            )
        )
    )

    # Calculate the proportion of treatment episodes resulting in relapse
    relapse_prop = (
        1.0 - tsr - prop_death_from_treatment - prop_natural_death_while_on_treatment
    )

    return tuple(
        [param * duration for param in [tsr, prop_death_from_treatment, relapse_prop]]
    )

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

def load_full_param_info() -> pd.DataFrame:
    """
    Load specific parameter information from a ridigly formatted yaml file, and crash otherwise.

    Returns:
        The parameters info DataFrame contains the following field:
            value: Enough parameter values to ensure model runs, may be over-written in calibration
    """
    with open(project_path / "parameters.yaml", "r") as param_file:
        param_info = yml.safe_load(param_file)

    return param_info

def create_parameter_table(
    priors, 
    fixed_params_info, 
    sections=["Demographics", "TB transmission and disease natural history", "TB control (detection and treatment)"],
    drop_columns=['Group', 'Unit'],
    additional_params=None
):
    """
    Create a formatted parameter table with sections.
    
    Args:
        priors: List of prior objects from get_all_priors()
        fixed_params_info: Dictionary with parameter information from load_full_param_info()
        sections: List of section names to include (default: all available sections)
                 e.g., ["Demographics", "TB transmission and disease natural history"]
        drop_columns: List of columns to drop (default: ['Group', 'Unit'])
        additional_params: List of dicts for manually added parameters
                          e.g., [{"Parameter": "Birth rate", "Value": "Time-variant", ...}]
    
    Returns:
        pandas DataFrame with formatted parameter table
    """
    import pandas as pd
    
    # Default values
    if drop_columns is None:
        drop_columns = ['Group', 'Unit']
    
    # Extract prior names
    priors_str = [str(prior) for prior in priors]
    prior_names = [priors[1] for priors in map(str.split, priors_str)]
    
    # Build parameter table
    param_table = []
    processed_keys = set()
    
    # Process grouped parameters first
    for group_name, group_info in PARAMETER_GROUPS.items():
        group_keys = group_info["keys"]
        existing_keys = [k for k in group_keys if k in fixed_params_info["value"]]
        if not existing_keys:
            continue
        
        values, units, sources, groups = [], [], [], []
        all_calibrated = True
        
        for key in existing_keys:
            processed_keys.add(key)
            
            if key in prior_names:
                values.append("Calibrated")
            else:
                value = fixed_params_info["value"][key]
                values.append(str(round(value, 3) if value != 0.0 else 0.0))
                all_calibrated = False
            
            units.append(fixed_params_info["unit"][key])
            sources.append(fixed_params_info["sources"][key])
            groups.append(fixed_params_info["group"][key])
        
        value_str = "Calibrated" if all_calibrated else " / ".join(values)
        
        param_table.append({
            "Parameter": group_info["shared_description"],
            "Value": value_str,
            "Unit": units[0] if len(set(units)) == 1 else " / ".join(units),
            "Source": sources[0] if len(set(sources)) == 1 else " / ".join(sources),
            "Group": groups[0] if len(set(groups)) == 1 else " / ".join(groups),
        })
    
    # Process non-grouped parameters
    for key in fixed_params_info["value"]:
        if key in processed_keys:
            continue
            
        if isinstance(fixed_params_info["value"][key], dict):
            if key == "time_variant_tsr" or "time_variant" in key:
                value_str = "Calibrated" if key in prior_names else "time-variant"
                
                param_table.append({
                    "Parameter": fixed_params_info["descriptions"][key],
                    "Value": value_str,
                    "Unit": fixed_params_info["unit"][key],
                    "Source": fixed_params_info["sources"][key],
                    "Group": fixed_params_info["group"][key],
                })
            elif key in fixed_params_info["unit"]:
                for subkey, value in fixed_params_info["value"][key].items():
                    param_name = f"{key}.{subkey}"
                    
                    if param_name in prior_names:
                        value_str = "Calibrated"
                    else:
                        if isinstance(value, dict):
                            value_str = "/".join(str(v) for v in value.values())
                        else:
                            value_str = str(round(value, 3) if value != 0.0 else 0.0)
                    
                    param_table.append({
                        "Parameter": f"{fixed_params_info['descriptions'][key][subkey]}",
                        "Value": value_str,
                        "Unit": fixed_params_info["unit"][key][subkey],
                        "Source": fixed_params_info["sources"][key][subkey],
                        "Group": fixed_params_info["group"][key][subkey],
                    })
        else:
            value_str = "Calibrated" if key in prior_names else str(
                round(fixed_params_info["value"][key], 3) 
                if fixed_params_info["value"][key] != 0.0 else 0.0
            )
            
            param_table.append({
                "Parameter": fixed_params_info["descriptions"][key],
                "Value": value_str,
                "Unit": fixed_params_info["unit"][key],
                "Source": fixed_params_info["sources"][key],
                "Group": fixed_params_info["group"][key],
            })
    
    # Add calibrated-only parameters
    processed_params = set(fixed_params_info["value"].keys())
    for key in fixed_params_info["value"]:
        if isinstance(fixed_params_info["value"][key], dict):
            for subkey in fixed_params_info["value"][key].keys():
                processed_params.add(f"{key}.{subkey}")
    
    for param_name in prior_names:
        if param_name not in processed_params:
            param_table.append({
                "Parameter": fixed_params_info["descriptions"].get(param_name, param_name),
                "Value": "Calibrated",
                "Unit": fixed_params_info["unit"].get(param_name, ""),
                "Source": fixed_params_info["sources"].get(param_name, "Calibrated"),
                "Group": fixed_params_info["group"].get(param_name, ""),
            })
    
    # Add additional parameters if provided
    if additional_params:
        param_table.extend(additional_params)
    
    # Determine which sections to include
    if sections is None:
        sections = list(PARAMETER_SECTIONS.keys())
    
    # Create section tables with headers
    section_dfs = []
    for section_name in sections:
        section_params = get_rows_by_section(param_table, section_name, fixed_params_info)
        if not section_params:
            continue
        
        # Create header
        header = pd.DataFrame([{
            "Parameter": f"\\textbf{{{section_name}}}",
            "Value": "",
            "Unit": "",
            "Source": "",
            "Group": "",
        }])
        
        # Create section table
        section_df = pd.DataFrame(section_params)
        section_dfs.append(pd.concat([header, section_df], ignore_index=True))
    
    # Combine all sections
    if not section_dfs:
        return pd.DataFrame()
    
    combined_df = pd.concat(section_dfs, ignore_index=True)
    
    # Drop specified columns
    if drop_columns:
        combined_df = combined_df.drop(columns=drop_columns, errors='ignore')
    
    # Capitalize column names
    combined_df.columns = combined_df.columns.str.capitalize()
    
    return combined_df

def get_section_for_param(param_description, fixed_params_info):
    """Match parameter description to its section"""
    # First, check for direct description match (for manually added rows)
    for section_name, param_keys in PARAMETER_SECTIONS.items():
        if param_description in param_keys:
            return section_name
    
    # Then check grouped descriptions
    for section_name, param_keys in PARAMETER_SECTIONS.items():
        for key in param_keys:
            # Check if it matches a group
            if key in PARAMETER_GROUPS:
                if param_description == PARAMETER_GROUPS[key]["shared_description"]:
                    return section_name
            # Check if it matches individual parameters from YAML
            elif key in fixed_params_info.get("descriptions", {}):
                if isinstance(fixed_params_info["descriptions"][key], dict):
                    for subkey_desc in fixed_params_info["descriptions"][key].values():
                        if param_description == subkey_desc:
                            return section_name
                else:
                    if param_description == fixed_params_info["descriptions"][key]:
                        return section_name
    return "Other parameters"

def get_rows_by_section(param_table, section_name, fixed_params_info):
    """Get all parameter rows from a specific section"""
    rows = []
    for row in param_table:
        section = get_section_for_param(row["Parameter"], fixed_params_info)
        if section == section_name:
            rows.append(row)
    return rows

def get_row_col_for_subplots(i_panel, n_cols):
    return int(np.floor(i_panel / n_cols)) + 1, i_panel % n_cols + 1

def get_next_run_number_for_config(out_path, draws, tune, config_name):
    """
    Find the next available run number for a specific number of priors.
    Returns a formatted string like '01', '02', etc.
    """
    pattern = f'calib_full_out_{draws}d{tune}t_{config_name}*.nc'
    existing_files = list(out_path.glob(pattern))
    
    if not existing_files:
        return '01'
    
    numbers = []
    for f in existing_files:
        try:
            run_num = int(f.stem.rsplit('_', 1)[-1])
            numbers.append(run_num)
        except ValueError:
            continue

    next_num = max(numbers) + 1 if numbers else 1
    return f'{next_num:02d}'

def get_next_run_number(out_path, draws, tune):
    """
    Find the next available run number for a specific number of priors.
    Returns a formatted string like '01', '02', etc.
    """
    pattern = f'calib_full_out_{draws}d{tune}t_*.nc'
    existing_files = list(out_path.glob(pattern))
    
    if not existing_files:
        return '01'
    
    numbers = []
    for f in existing_files:
        try:
            run_num = int(f.stem.rsplit('_', 1)[-1])
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


