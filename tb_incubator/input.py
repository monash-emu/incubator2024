import pandas as pd
from tb_incubator.constants import set_project_base_path

project_paths = set_project_base_path("../tb_incubator")

# get age groups in range
def get_age_groups_in_range(age_groups, lower_limit, upper_limit):
    return [i for i in age_groups if '+' not in i and lower_limit <= int(i.split('-')[0]) <= upper_limit]

# load UN birth data
data_path = project_paths["DATA_PATH"]
demographics = pd.read_csv(data_path / 'un_demographics.csv')

countries = ["Indonesia"] # select Indonesia's data
id_demographics = demographics[demographics["Location"].isin(countries)].reset_index(drop=True) 
id_demographics['Time'] = id_demographics['Time'].astype(str).str.extract('(\d+)', expand=False)
id_demographics['Time'] = pd.to_datetime(id_demographics['Time'].astype(str), format='%Y')
id_demographics['Time'] = id_demographics['Time'].dt.year
id_demographics = id_demographics.set_index('Time') # set `Time` column as index

id_births = id_demographics.loc[1950:2023] # select 1950-2023 data
id_births = id_births[["TPopulation1July", "Births"]]
id_births['Birth_rate'] = id_births['Births'] / id_births['TPopulation1July'] # calculate birth rate
id_births = id_births.rename(columns={'TPopulation1July': 'population',
                                      "Births" : "births",
                                      "Birth_rate" : "birth_rate"}) # rename columns name
id_births

def get_birth_rate():
    return id_births["birth_rate"]

get_birth_rate()

#load UN death and population data

## load death data

un_deaths = pd.read_csv(data_path / 'un_deaths_single_age.csv') 
countries = ["Indonesia"] # select Indonesia's data
deaths = un_deaths[un_deaths["Region, subregion, country or area *"].isin(countries)].reset_index(drop=True)
columns_to_drop = deaths.columns[0:10]
deaths = deaths.drop(columns=columns_to_drop)

deaths['Year'] = deaths['Year'].astype(str).str.extract('(\d+)', expand=False)
deaths['Year'] = pd.to_datetime(deaths['Year'].astype(str), format='%Y')
deaths['Year'] = deaths['Year'].dt.year # show the year only

deaths_melt = pd.melt(deaths, id_vars="Year")
deaths_melt = deaths_melt.rename(columns={'Year': 'year','variable': 'age_group', 'value': 'deaths'})

id_deaths = pd.DataFrame(deaths_melt)
id_deaths.index = pd.MultiIndex.from_frame(deaths_melt[["year", "age_group"]])

id_deaths["deaths"]=pd.to_numeric(id_deaths["deaths"])

age_groups = set(id_deaths.index.get_level_values(1))
years = set(id_deaths.index.get_level_values(0))

agegroup_request = [[0, 4], [5, 14], [15, 34], [35, 49], [50, 100]]
agegroup_map = {f"{low}-{up}": get_age_groups_in_range(age_groups, low, up) for low, up in agegroup_request}
if '100+' in age_groups:
    agegroup_map['50-100'].append('100+')
agegroup_map

index = pd.MultiIndex.from_product([id_deaths.index.get_level_values(0).unique(), agegroup_map.keys()], names=['year', 'agegroup'])
id_deaths_agegroups = pd.DataFrame(index=index, columns=["deaths"])

for year in id_deaths.index.get_level_values(0).unique():
    for agegroup, agegroup_list in agegroup_map.items():
        age_mask = id_deaths.index.get_level_values(1).isin(agegroup_list)
        
        # Filter the data for the specific year and age group
        age_year_data = id_deaths.loc[(year, age_mask), :]
        
        total = age_year_data["deaths"].sum()
        
        # Assign the summed total to the new DataFrame
        id_deaths_agegroups.loc[(year, agegroup), "deaths"] = total


## population data
un_pop = pd.read_csv(data_path / 'un_population.csv')
countries = ["Indonesia"] # select Indonesia's data
pop = un_pop[un_pop["Region, subregion, country or area *"].isin(countries)].reset_index(drop=True)
columns_to_drop = pop.columns[0:10]
pop = pop.drop(columns=columns_to_drop)

pop['Year'] = pop['Year'].astype(str).str.extract('(\d+)', expand=False)
pop['Year'] = pd.to_datetime(pop['Year'].astype(str), format='%Y')
pop['Year'] = pop['Year'].dt.year # show the year only

pop_melt = pd.melt(pop, id_vars="Year")
pop_melt = pop_melt.rename(columns={'Year': 'year','variable': 'age_group', 'value': 'population'})

id_pop = pd.DataFrame({
    'population': pop_melt["population"]
})
id_pop.index = pd.MultiIndex.from_frame(pop_melt[["year", "age_group"]])
id_pop["population"] = pd.to_numeric(id_pop["population"], errors='coerce')
id_pop["population"] = id_pop["population"] * 1000

age_groups = set(id_pop.index.get_level_values(1))
years = set(id_pop.index.get_level_values(0))

agegroup_request = [[0, 4], [5, 14], [15, 34], [35, 49], [50, 100]]
agegroup_map = {f"{low}-{up}": get_age_groups_in_range(age_groups, low, up) for low, up in agegroup_request}
if '100+' in age_groups:
    agegroup_map['50-100'].append('100+')
agegroup_map

index = pd.MultiIndex.from_product([id_pop.index.get_level_values(0).unique(), agegroup_map.keys()], names=['year', 'agegroup'])
id_pop_agegroups = pd.DataFrame(index=index, columns=["population"])

for year in id_pop.index.get_level_values(0).unique():
    for agegroup, agegroup_list in agegroup_map.items():
        age_mask = id_pop.index.get_level_values(1).isin(agegroup_list)
        
        # Filter the data for the specific year and age group
        age_year_data = id_pop.loc[(year, age_mask), :]
        
        total = age_year_data["population"].sum()
        
        # Assign the summed total to the new DataFrame
        id_pop_agegroups.loc[(year, agegroup), "population"] = total

## merge population and death data based on the indices
id_pop_deaths = pd.merge(id_pop_agegroups, id_deaths_agegroups, left_index=True, right_index=True)
id_pop_deaths['population'] = pd.to_numeric(id_pop_deaths['population'], errors='coerce')
id_pop_deaths['deaths'] = pd.to_numeric(id_pop_deaths['deaths'], errors='coerce')
id_pop_deaths["population"] = id_pop_deaths["population"].astype(int)


def get_pop_death_data():
    return id_pop_deaths

get_pop_death_data()


# load death rates
age_groups = set(id_pop_deaths.index.get_level_values(1))
years = set(id_pop_deaths.index.get_level_values(0))

agegroup_request = [[0, 4], [5, 14], [15, 34], [35, 49], [50, 100]]
agegroup_map = {low: get_age_groups_in_range(age_groups, low, up) for low, up in agegroup_request}
agegroup_map[agegroup_request[-1][0]].append('100+')
agegroup_map

mapped_rates = pd.DataFrame()
for year in years:
    for agegroup in agegroup_map:
        age_mask = [i in agegroup_map[agegroup] for i in id_pop_deaths.index.get_level_values(1)]
        age_year_data = id_pop_deaths.loc[age_mask].loc[year, :]
        total = age_year_data.sum()
        mapped_rates.loc[year, agegroup] = total['deaths'] / total['population']
death_rates = mapped_rates.loc[id_pop_deaths.index.get_level_values(0)]

def get_death_rates():
    return death_rates

get_death_rates()
