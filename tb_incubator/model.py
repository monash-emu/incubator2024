import pandas as pd
from summer2 import CompartmentalModel
from tb_incubator.constants import set_project_base_path
from summer2.parameters import Parameter


project_paths = set_project_base_path("../tb_incubator")
data_path = project_paths["DATA_PATH"]

# Add latency structures

def add_latency_flow(model):
    latency_flows = [
        ["stabilisation", "early latent", "late latent"],
        ["early activation", "early latent", "infectious"],
        ["late activation", "late latent", "infectious"],
    ]
    
    for flow, source, dest in latency_flows:
        print(f"Adding flow: {flow}, from {source} to {dest}")
        model.add_transition_flow(flow, Parameter(f"{flow} rate"), source, dest)


def add_infection_flow(model):
    infection_flows = [
        ["susceptible", None],
        ["late latent", "rr infection latent"],
        ["recovered", "rr infection recovered"],
    ]

    for origin, modifier in infection_flows:
        modifier = Parameter(modifier) if modifier else 1.0
        rate = Parameter("contact rate") * modifier
        name = f"infection from {origin}"
        model.add_infection_frequency_flow(name, rate, origin, "early latent")

    description= "We added infection flows to the model."

    return description
    