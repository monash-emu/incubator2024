import pandas as pd
from tb_incubator.constants import set_project_base_path

project_paths = set_project_base_path("../tb_incubator")
data_path = project_paths["DATA_PATH"]

# Get age groups in a specified range
def get_age_groups_in_range(age_groups, lower_limit, upper_limit):
    return [i for i in age_groups if '+' not in i and lower_limit <= int(i.split('-')[0]) <= upper_limit]

# Birth rate
def load_and_process_birth_data():
    demographics = pd.read_csv(data_path / 'un_demographics.csv')
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
    id_births = load_and_process_birth_data()
    return id_births["birth_rate"]

# Load Indonesia's mortality data
def load_death_data():
    un_deaths = pd.read_csv(data_path / 'un_deaths_single_age.csv')
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

# Age-stratified mortality data
def process_death_data(id_deaths, agegroup_request):
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

# Get Indonesia's population data
def load_population_data():
    un_pop = pd.read_csv(data_path / 'un_population.csv')
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

# Age-stratified population data
def process_population_data(id_pop, agegroup_request):
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

# Merging population and death data
def merge_population_death_data(id_pop_agegroups, id_deaths_agegroups):
    id_pop_deaths = pd.merge(id_pop_agegroups, id_deaths_agegroups, left_index=True, right_index=True)
    id_pop_deaths['population'] = pd.to_numeric(id_pop_deaths['population'], errors='coerce').astype(int)
    id_pop_deaths['deaths'] = pd.to_numeric(id_pop_deaths['deaths'], errors='coerce')

    return id_pop_deaths

def get_pop_death_data():
    agegroup_request = [[0, 4], [5, 14], [15, 34], [35, 49], [50, 100]]

    id_deaths = load_death_data()
    id_deaths_agegroups = process_death_data(id_deaths, agegroup_request)

    id_pop = load_population_data()
    id_pop_agegroups = process_population_data(id_pop, agegroup_request)

    return merge_population_death_data(id_pop_agegroups, id_deaths_agegroups)

# Calculate death rates
def calculate_death_rates(id_pop_deaths):
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
    id_pop_deaths = get_pop_death_data()
    death_rates = calculate_death_rates(id_pop_deaths)
    return death_rates


# Main Blocks
if __name__ == "__main__":
    data_path = project_paths["DATA_PATH"]

    # Calculate and print birth rate
    birth_rate = get_birth_rate()
    print("Birth Rate:")
    print(birth_rate)

    # Get and print population and death data
    pop_death_data = get_pop_death_data()
    print("Population and Death Data:")
    print(pop_death_data)

    # Calculate and print death rates
    death_rates = get_death_rates()
    print("Death Rates:")
    print(death_rates)
    


