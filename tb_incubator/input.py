import pandas as pd
from tb_incubator.constants import set_project_base_path
from summer2.functions.time import get_sigmoidal_interpolation_function
from summer2 import Overwrite


project_paths = set_project_base_path("../tb_incubator")
data_path = project_paths["DATA_PATH"]


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
    return [i for i in age_groups if '+' not in i and lower_limit <= int(i.split('-')[0]) <= upper_limit]


def load_and_process_birth_data():
    """
    Loads and processes birth and population data for Indonesia from a CSV file.

    This function performs the following steps:
    1. Reads the data from a CSV file named 'un_demographics.csv'.
    2. Filters the data to include only entries for Indonesia.
    3. Converts the 'Time' column to a datetime format, extracting the year.
    4. Sets the 'Time' column as the index.
    5. Selects data from 1950 to 2023 and retains relevant columns.
    6. Calculates the birth rate as births divided by the total population.
    7. Renames columns for clarity.

    Returns:
        pd.DataFrame: A DataFrame with columns for population, births, and birth rate, indexed by year.
    """
    demographics = pd.read_csv(data_path / 'un_demographics.csv', low_memory=False)
    countries = ["Indonesia"]  # Select Indonesia's data
    id_demographics = demographics[demographics["Location"].isin(countries)].reset_index(drop=True)
    id_demographics['Time'] = pd.to_datetime(id_demographics['Time'].astype(str).str.extract('(\d+)', expand=False), format='%Y').dt.year
    id_demographics = id_demographics.set_index('Time')  # Set `Time` column as index

    id_births = id_demographics.loc[1950:2023]  # Select 1950-2023 data
    id_births = id_births[["TPopulation1July", "Births"]]
    id_births['Birth_rate'] = id_births['Births'] / id_births['TPopulation1July']  # Calculate birth rate
    id_births = id_births.rename(columns={'TPopulation1July': 'population', "Births": "births", "Birth_rate": "birth_rate"})  # Rename columns

    return id_births

def get_birth_rate():
    """
    Retrieves the birth rate data from the processed birth dataset.

    Returns:
        pd.Series: A Series containing the birth rate data, indexed by year.
    """
    id_births = load_and_process_birth_data()
    return id_births["birth_rate"]


def load_death_data():
    """
    Loads and processes death data for Indonesia from a CSV file.

    This function performs the following steps:
    1. Reads the death data from 'un_deaths_single_age.csv'.
    2. Filters the data to include only records for Indonesia.
    3. Removes unnecessary columns from the dataset.
    4. Extracts and converts the year from the 'Year' column.
    5. Reshapes the data from wide to long format using `pd.melt`.
    6. Renames columns for clarity and sets a MultiIndex based on year and age group.
    7. Converts the 'deaths' column to numeric data type.

    Returns:
        pd.DataFrame: A DataFrame with a MultiIndex (year and age group) and columns for death counts.
    """
    un_deaths = pd.read_csv(data_path / 'un_deaths_single_age.csv', low_memory=False)
    countries = ["Indonesia"]  # Select Indonesia's data
    deaths = un_deaths[un_deaths["Region, subregion, country or area *"].isin(countries)].reset_index(drop=True)
    deaths = deaths.drop(columns=deaths.columns[0:10])

    deaths['Year'] = pd.to_datetime(deaths['Year'].astype(str).str.extract('(\d+)', expand=False), format='%Y').dt.year
    deaths_melt = pd.melt(deaths, id_vars="Year")
    deaths_melt = deaths_melt.rename(columns={'Year': 'year', 'variable': 'age_group', 'value': 'deaths'})

    id_deaths = pd.DataFrame(deaths_melt)
    id_deaths.index = pd.MultiIndex.from_frame(deaths_melt[["year", "age_group"]])
    id_deaths["deaths"] = pd.to_numeric(id_deaths["deaths"])

    return id_deaths


def process_death_data(id_deaths, agegroup_request):
    """
    Processes death data to aggregate deaths into specified age groups.
    
    Args:
        id_deaths (pd.DataFrame): DataFrame with a MultiIndex (year, age group) and a 'deaths' column.
        agegroup_request (list of lists): List of age ranges (e.g., [[0, 4], [5, 14], [15, 34], [35, 49], [50, 100]]).

    Returns:
        pd.DataFrame: A DataFrame with a MultiIndex (year, age group) and a 'deaths' column, aggregated by the specified age groups.
    """
    age_groups = set(id_deaths.index.get_level_values(1))
    agegroup_map = {f"{low}-{up}": get_age_groups_in_range(age_groups, low, up) for low, up in agegroup_request}
    if '100+' in age_groups:
        agegroup_map['50-100'].append('100+')

    index = pd.MultiIndex.from_product([id_deaths.index.get_level_values(0).unique(), agegroup_map.keys()], names=['year', 'agegroup'])
    id_deaths_agegroups = pd.DataFrame(index=index, columns=["deaths"])

    for year in id_deaths.index.get_level_values(0).unique():
        for agegroup, agegroup_list in agegroup_map.items():
            age_mask = id_deaths.index.get_level_values(1).isin(agegroup_list)
            age_year_data = id_deaths.loc[(year, age_mask), :]
            total = age_year_data["deaths"].sum()
            id_deaths_agegroups.loc[(year, agegroup), "deaths"] = total

    return id_deaths_agegroups


def load_population_data():
    """
    Loads and processes population data from a CSV file.

    This function performs the following steps:
    1. Reads the population data from a CSV file.
    2. Filters data for Indonesia.
    3. Drops unnecessary columns.
    4. Extracts and formats the year from the data.
    5. Melts the DataFrame to convert it into a long format.
    6. Renames columns for consistency.
    7. Creates a MultiIndex DataFrame with year and age group.
    8. Converts population values to numeric and scales them.

    Returns:
        pd.DataFrame: A DataFrame with a MultiIndex (year, age_group) and a 'population' column.
    """
    un_pop = pd.read_csv(data_path / 'un_population.csv', low_memory=False)
    countries = ["Indonesia"]  # Select Indonesia's data
    pop = un_pop[un_pop["Region, subregion, country or area *"].isin(countries)].reset_index(drop=True)
    pop = pop.drop(columns=pop.columns[0:10])

    pop['Year'] = pd.to_datetime(pop['Year'].astype(str).str.extract('(\d+)', expand=False), format='%Y').dt.year
    pop_melt = pd.melt(pop, id_vars="Year")
    pop_melt = pop_melt.rename(columns={'Year': 'year', 'variable': 'age_group', 'value': 'population'})

    id_pop = pd.DataFrame({'population': pop_melt["population"]})
    id_pop.index = pd.MultiIndex.from_frame(pop_melt[["year", "age_group"]])
    id_pop["population"] = pd.to_numeric(id_pop["population"], errors='coerce') * 1000

    return id_pop


def process_population_data(id_pop, agegroup_request):
    """
    Processes population data by aggregating it into specified age groups.

    Args:
        id_pop (pd.DataFrame): DataFrame with MultiIndex (year, age_group) and a 'population' column.
        agegroup_request (list of tuples): List of age group ranges as (lower_limit, upper_limit).

    Returns:
        pd.DataFrame: A DataFrame with MultiIndex (year, agegroup) and a 'population' column aggregated by age groups.
    """
    age_groups = set(id_pop.index.get_level_values(1))
    agegroup_map = {f"{low}-{up}": get_age_groups_in_range(age_groups, low, up) for low, up in agegroup_request}
    if '100+' in age_groups:
        agegroup_map['50-100'].append('100+')

    index = pd.MultiIndex.from_product([id_pop.index.get_level_values(0).unique(), agegroup_map.keys()], names=['year', 'agegroup'])
    id_pop_agegroups = pd.DataFrame(index=index, columns=["population"])

    for year in id_pop.index.get_level_values(0).unique():
        for agegroup, agegroup_list in agegroup_map.items():
            age_mask = id_pop.index.get_level_values(1).isin(agegroup_list)
            age_year_data = id_pop.loc[(year, age_mask), :]
            total = age_year_data["population"].sum()
            id_pop_agegroups.loc[(year, agegroup), "population"] = total

    return id_pop_agegroups


def merge_population_death_data(id_pop_agegroups, id_deaths_agegroups):
    """
    Merges population and death data by age group and year.

    This function performs the following steps:
    1. Merges the population data (`id_pop_agegroups`) and death data (`id_deaths_agegroups`) based on their MultiIndex (year, agegroup).
    2. Converts the 'population' and 'deaths' columns to numeric types, handling any conversion errors.
    3. Ensures the 'population' column is of integer type after conversion.

    Args:
        id_pop_agegroups (pd.DataFrame): DataFrame with MultiIndex (year, agegroup) and a 'population' column.
        id_deaths_agegroups (pd.DataFrame): DataFrame with MultiIndex (year, agegroup) and a 'deaths' column.

    Returns:
        pd.DataFrame: A DataFrame with MultiIndex (year, agegroup), including 'population' and 'deaths' columns.
    """
    id_pop_deaths = pd.merge(id_pop_agegroups, id_deaths_agegroups, left_index=True, right_index=True)
    id_pop_deaths['population'] = pd.to_numeric(id_pop_deaths['population'], errors='coerce').astype(int)
    id_pop_deaths['deaths'] = pd.to_numeric(id_pop_deaths['deaths'], errors='coerce')

    return id_pop_deaths

def get_pop_death_data():
    """
    Retrieves and processes population and death data for specified age groups.

    This function performs the following steps:
    1. Defines the age groups for stratification.
    2. Loads death data and processes it into age-specific groups.
    3. Loads population data and processes it into age-specific groups.
    4. Merges the processed population and death data.

    Returns:
        pd.DataFrame: A DataFrame with MultiIndex (year, agegroup) including both 'population' and 'deaths' columns.
    """
    agegroup_request = [[0, 4], [5, 14], [15, 34], [35, 49], [50, 100]]

    id_deaths = load_death_data()
    id_deaths_agegroups = process_death_data(id_deaths, agegroup_request)

    id_pop = load_population_data()
    id_pop_agegroups = process_population_data(id_pop, agegroup_request)

    return merge_population_death_data(id_pop_agegroups, id_deaths_agegroups)


def calculate_death_rates(id_pop_deaths):
    """
    Calculates age-specific death rates from the merged population and death data.

    This function performs the following steps:
    1. Defines age groups and maps them to specific ranges.
    2. Iterates through each year and age group to calculate death rates.
    3. Returns a DataFrame with calculated death rates.

    Args:
        id_pop_deaths (pd.DataFrame): A DataFrame with MultiIndex (year, agegroup) including 'population' and 'deaths' columns.

    Returns:
        pd.DataFrame: A DataFrame with age-specific death rates, indexed by year and age group.
    """
    age_groups = set(id_pop_deaths.index.get_level_values(1))
    years = set(id_pop_deaths.index.get_level_values(0).unique())    

    agegroup_request2 = [[0, 4], [5, 14], [15, 34], [35, 49], [50, 100]]
    agegroup_map2 = {low: get_age_groups_in_range(age_groups, low, up) for low, up in agegroup_request2}
    agegroup_map2[agegroup_request2[-1][0]].append('100+')
    agegroup_map2

    mapped_rates = pd.DataFrame()
    for year in years:
        for agegroup in agegroup_map2:
            age_mask = [i in agegroup_map2[agegroup] for i in id_pop_deaths.index.get_level_values(1)]
            age_year_data = id_pop_deaths.loc[age_mask].loc[year, :]
            total = age_year_data.sum()
            mapped_rates.loc[year, agegroup] = total['deaths'] / total['population']
    
    mapped_rates = mapped_rates.loc[id_pop_deaths.index.get_level_values(0).unique()]
           
    return mapped_rates

def get_death_rates():
    """
    Retrieves and calculates age-specific death rates from population and death data.

    This function performs the following steps:
    1. Loads and processes population and death data.
    2. Calculates age-specific death rates based on the processed data.
    3. Returns a DataFrame with age-specific death rates.

    Returns:
        pd.DataFrame: A DataFrame with calculated death rates, indexed by year and age group.
    """
    id_pop_deaths = get_pop_death_data()
    death_rates = calculate_death_rates(id_pop_deaths)
    return death_rates

def get_population_entry_rate(pop_data, model_start_period):
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
    total_pop_by_year = pop_data.groupby("year")["population"].sum()
    pop_start_year = total_pop_by_year.index[0]
    start_period = pop_start_year - model_start_period

    # Calculate population entry rates and convert to function
    pop_entry = total_pop_by_year.diff().dropna()  # Note this will only work if data are annual
    pop_entry[pop_start_year] = total_pop_by_year[pop_start_year] / start_period
    pop_entry = pop_entry.sort_index()
    entry_rate = get_sigmoidal_interpolation_function(pop_entry.index, pop_entry)

    return entry_rate


def get_death_adjs(deathrate_df, age_strata):
    """
    Generate age-specific death adjustment functions based on age strata and death rate data.
    
    This function calculates and returns a dictionary of functions for adjusting death rates by age. 
    It creates a sigmoidal interpolation function based on the provided death rate data, for each age 
    group specified in `age_strata`.

    Parameters:
    - deathrate_df (pd.DataFrame): A DataFrame containing death rates with years as the index 
      and age groups as columns.
    - age_strata (list): A list of age groups.

    Returns:
    - dict: A dictionary where each key is an age group (as a string) and each value is an 
      `Overwrite` object containing the sigmoidal interpolation function for that age group's 
      death rates.
    """
    death_adjs = {}
    for age in age_strata:
        years = deathrate_df.index
        rates = deathrate_df[age]
        pop_death_func = get_sigmoidal_interpolation_function(years, rates)
        death_adjs[str(age)] = Overwrite(pop_death_func)
    return death_adjs

def display_summary(birth_data, pop_death_data):
    """
    Generates a summary table of key demographic metrics from the provided birth and population-death data.

    Parameters:
    - birth_data (DataFrame): DataFrame containing birth rates with a datetime index.
    - pop_death_data (DataFrame): DataFrame containing population and death data with a multi-level index (year, strata).

    Returns:
    - DataFrame: A table summarizing the following metrics for the start, middle, and end years of the dataset:
        - **Total Population**: Aggregated and formatted with thousands separators.
        - **Total Deaths**: Aggregated and formatted with thousands separators.
        - **Birth Rate**: Extracted and formatted to four decimal places.

    The table provides a concise overview of demographic changes at three distinct points in time, offering insights into population and mortality trends along with birth rates.
    
    """
    start_year = birth_data.index[0]
    middle_year = (birth_data.index[0] + birth_data.index[-1]) // 2
    end_year = birth_data.index[-1]

    birth_rate_start = birth_data.loc[start_year].item()
    birth_rate_middle = birth_data.loc[middle_year].item()
    birth_rate_end = birth_data.loc[end_year].item()

    total_pop_start = pop_death_data.loc[(start_year, slice(None)), 'population'].sum().item()
    total_pop_middle = pop_death_data.loc[(middle_year, slice(None)), 'population'].sum().item()
    total_pop_end = pop_death_data.loc[(end_year, slice(None)), 'population'].sum().item()

    total_death_start = pop_death_data.loc[(start_year, slice(None)), 'deaths'].sum().item()
    total_death_middle = pop_death_data.loc[(middle_year, slice(None)), 'deaths'].sum().item()
    total_death_end = pop_death_data.loc[(end_year, slice(None)), 'deaths'].sum().item()

    total_pop = {
    start_year: f"{total_pop_start:,}",
    middle_year: f"{total_pop_middle:,}",
    end_year: f"{total_pop_end:,}"
    }

    total_death = {
    start_year: f"{total_death_start:,}",
    middle_year: f"{total_death_middle:,}",
    end_year: f"{total_death_end:,}"
    }

    birthrate = {
    start_year: f"{birth_rate_start:,.4f}",  # Adjust to 2 decimal places if needed
    middle_year: f"{birth_rate_middle:,.4f}",
    end_year: f"{birth_rate_end:,.4f}"
    }

    summary_table = pd.DataFrame({"Total Population": total_pop, "Total Death": total_death, "Birth rate": birthrate})

    return summary_table
