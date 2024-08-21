import numpy as np
import pandas as pd
from plotly import graph_objects as go
import nevergrad as ng

from tb_incubator.constants import set_project_base_path

project_paths = set_project_base_path("../incubator2024/tb_incubator")

# load UN birth data
data_path = project_paths["DATA_PATH"]
demographics = pd.read_csv(data_path / 'un_demographics.csv')

countries = ["Indonesia"] # select Indonesia's data
id_demographics = demographics[demographics["Location"].isin(countries)].reset_index(drop=True) 
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
    return id_births

get_birth_rate()