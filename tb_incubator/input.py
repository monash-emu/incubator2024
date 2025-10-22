import pandas as pd
from typing import List, Dict
from tb_incubator.constants import project_path, data_path
from summer2.functions.time import get_sigmoidal_interpolation_function
import yaml as yml
from IPython.display import Markdown, display
from pathlib import Path

def load_report_section(file_path, section):
    # Load the report sections from the YAML file
    with open(project_path / file_path, 'r') as file:
        data = yml.safe_load(file)

    # Retrieve the specified section
    section_text = data.get(section, "Section not found.")
    
    # Display the section text as markdown in a Jupyter notebook
    display(Markdown(section_text))

def load_genexpert_util():
    targets = load_targets()

    return targets["genexpert_utilisation"]

def load_targets():
    with open(project_path / "targets.yaml", "r") as file:
        data = yml.safe_load(file)

    processed_targets = {}

    for key, value in data.items():
        if isinstance(value, dict):
            # Check if the value for each key is a list of three items
            if all(isinstance(v, list) and len(v) == 3 for v in value.values()):
                # Handle as [target, lower_bound, upper_bound]
                target = pd.Series({k: v[0] for k, v in value.items()})
                lower_bound = pd.Series({k: v[1] for k, v in value.items()})
                upper_bound = pd.Series({k: v[2] for k, v in value.items()})

                processed_targets[f'{key}_target'] = target
                processed_targets[f'{key}_lower_bound'] = lower_bound
                processed_targets[f'{key}_upper_bound'] = upper_bound
            else:
                # Handle as single values
                processed_targets[key] = pd.Series(value)
        else:
            # Handle cases where value is not a dictionary
            processed_targets[key] = pd.Series(value)

    return processed_targets

def load_param_info() -> pd.DataFrame:
    """
    Load specific parameter information from a ridigly formatted yaml file, and crash otherwise.

    Returns:
        The parameters info DataFrame contains the following field:
            value: Enough parameter values to ensure model runs, may be over-written in calibration
    """
    with open(project_path / "parameters.yaml", "r") as param_file:
        param_info = yml.safe_load(param_file)["value"]

    return param_info


def get_age_groups_in_range(age_groups, lower_limit, upper_limit):
    """
    Retrieves age groups within a specified age range from a list of age groups.

    Parameters:
        age_groups (list): A list of age group strings (e.g., ['0-4', '5-14', '15-34']).
        lower_limit (int): The lower bound of the age range to filter.
        upper_limit (int): The upper bound of the age range to filter.

    Returns:
        list: A list of age group strings that fall within the specified age range.

    Notes:
        The function filters out age groups that contain '+' and selects only those with a starting age within the specified range.
    """
    return [
        i for i in age_groups if "+" not in i and lower_limit <= int(i.split("-")[0]) <= upper_limit
    ]

def get_birth_rate():
    """
    Load and return the national birth rate data for Vietnam.

    Reads a CSV file named 'vn_birth.csv' from the INPUT_PATH directory and returns
    the 'value' column as a pandas Series, indexed by year.

    Returns
    -------
    pandas.Series
        A Series of birth rates indexed by year.
    """
    return pd.read_csv(Path(data_path / "id_birth.csv"), index_col=0)["value"]

def get_death_rate():
    """
    Load and return age-specific death count data for Vietnam.

    Reads a CSV file named 'vn_cdr.csv' from the INPUT_PATH directory and returns
    a DataFrame containing age- and time-specific deaths and population, indexed
    by (Time, Age).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns ['Population', 'Deaths'], indexed by ['Time', 'Age'].
    """
    return pd.read_csv(
        Path(data_path / "id_cdr.csv"), usecols=["Age", "Time", "Population", "Deaths"]
    ).set_index(["Time", "Age"])

def process_death_rate(data: pd.DataFrame, age_strata: List[int], year_indices: List[float]):
    """
    Processes mortality data to compute age-stratified death rates for specific years.

    This function takes a dataset containing mortality and population data, along with
    definitions for age strata and specific years of interest, to compute the death rate
    within each age stratum for those years. The death rates are calculated as the total
    deaths divided by the total population within each age stratum for each year. The
    function also adjusts age groups to include an "100+" category and handles the mapping
    of raw age groups to the defined age strata.

    Parameters:
    - data: A pandas DataFrame indexed by (year, age_group) with at least
      two columns: 'Deaths' and 'Population', representing the total deaths and total
      population for each age group in each year, respectively.
    - age_strata: A list of integers representing the starting age of each
      age stratum to be considered. The list must be sorted in ascending order.
    - year_indices: A list of float representing the years of interest
      for which the death rates are to be calculated.

    Returns:
    - pd.DataFrame: A pandas DataFrame indexed by the mid-point year values (year + 0.5)
      with columns for each age stratum defined in `age_strata`. Each cell contains the
      death rate for that age stratum and year.
    """
    years = set(data.index.get_level_values(0))
    age_groups = set(data.index.get_level_values(1))

    # Creating the new list
    agegroup_request = [
        [start, end - 1] for start, end in zip(age_strata, age_strata[1:] + [201])
    ]
    agegroup_map = {
        low: get_age_groups_in_range(age_groups, low, up)
        for low, up in agegroup_request
    }
    agegroup_map[agegroup_request[-1][0]].append("100+")
    mapped_rates = pd.DataFrame()
    for year in years:
        for agegroup in agegroup_map:
            age_mask = [
                i in agegroup_map[agegroup] for i in data.index.get_level_values(1)
            ]
            age_year_data = data.loc[age_mask].loc[year, :]
            total = age_year_data.sum()
            mapped_rates.loc[year, agegroup] = total["Deaths"] / total["Population"]
    mapped_rates.index += 0.5
    death_df = mapped_rates.loc[year_indices]

    return death_df


def get_population_entry_rate(model_times):
    """
    Calculates the population entry rates based on total population data over the years.

    Parameters:
        pop_data (DataFrame): A DataFrame containing population data with 'year' and 'population' columns.
        model_start_period (int): The year from which the model starts.

    Returns:
        entry_rate (function): A function that provides sigmoidal interpolation of the calculated population entry rates.

    Notes:
        This will only work for annual data.
    """
    # Population by year and get the duration of the run-in period
    pop_data = get_death_rate()
    
    # Reset the index to access the 'Time' column for grouping
    pop_data_reset = pop_data.reset_index()

    total_pop_by_year = pop_data_reset.groupby("Time")["Population"].sum()

    pop_start_year = total_pop_by_year.index[0]
    start_period = pop_start_year - model_times[0]

    # Calculate population entry rates and convert to function
    pop_entry = total_pop_by_year.diff().dropna()  # Note this will only work if data are annual
    pop_entry.loc[pop_start_year] = total_pop_by_year[pop_start_year] / start_period
    pop_entry = pop_entry.sort_index()
    entry_rate = get_sigmoidal_interpolation_function(pop_entry.index, pop_entry)

    return entry_rate
