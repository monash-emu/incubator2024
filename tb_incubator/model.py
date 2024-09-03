import pandas as pd
from typing import List, Dict
from summer2 import Overwrite, AgeStratification, Multiply, CompartmentalModel
from summer2.parameters import Parameter, Time, Function
from summer2.functions.time import get_sigmoidal_interpolation_function
from tb_incubator.input import get_death_rates
from tb_incubator.constants import set_project_base_path
from tb_incubator.utils import get_average_sigmoid, triangle_wave_func

project_paths = set_project_base_path("../tb_incubator")
data_path = project_paths["DATA_PATH"]

# Add latency structures


def add_latency_flow(model):
    latency_flows = [
        ["stabilisation", "early latent", "late latent"],
        ["early activation", "early latent", "infectious"],
        ["late activation", "late latent", "infectious"],
    ]

    descriptions = []

    for flow, source, dest in latency_flows:
        description = f"Adding {flow} flow, from {source} to {dest}"
        descriptions.append(description)
        model.add_transition_flow(flow, 1.0, source, dest)

    return descriptions


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

    description = f"- Adding infection flows from {', '.join([flow[0] for flow in infection_flows])} to early latent compartment."
    return description


def set_popdeath_adjs(age_strata: List[int], strat: AgeStratification):
    deathrate_df, description = get_death_rates()
    death_adjs = {}
    for age in age_strata:
        years = deathrate_df.index
        rates = deathrate_df[age]
        pop_death_func = get_sigmoidal_interpolation_function(years, rates)
        death_adjs[str(age)] = Overwrite(pop_death_func)

    strat.set_flow_adjustments("population death", death_adjs)

    desc = f"- Adding age-specific adjustment for population death flows"

    return desc


def set_latency_adjs(params: Dict[str, any], age_strata: List[int], strat: AgeStratification):
    for flow_name, latency_params in params["age latency"].items():
        adjs = {}
        for age in age_strata:
            param_age_bracket = max([k for k in latency_params if k <= age])
            age_val = latency_params[param_age_bracket]

            adj = (
                Parameter("progression multiplier") * age_val
                if "late activation" in flow_name
                else age_val
            )
            adjs[str(age)] = adj

    adjs = {k: Overwrite(v) for k, v in adjs.items()}
    strat.set_flow_adjustments(flow_name, adjs)

    desc = f"- Adding age-specific adjustment for latency flows"

    return desc


def add_infectiousness_adjs(
    infectious_comp: List[str],
    params: Dict[str, any],
    age_strata: List[int],
    strat: AgeStratification,
):
    inf_switch_age = params["age_infectiousness_switch"]
    for comp in infectious_comp:
        inf_adjs = {}
        for i, age_low in enumerate(age_strata):
            if age_low == age_strata[-1]:
                average_infectiousness = 1.0
            else:
                age_high = age_strata[i + 1]
                average_infectiousness = get_average_sigmoid(age_low, age_high, inf_switch_age)
            # Update the adjustments dictionary for the current age group.
            inf_adjs[str(age_low)] = Multiply(average_infectiousness)

    strat.add_infectiousness_adjustments(comp, inf_adjs)


def seed_infectious(model: CompartmentalModel):
    """
    Adds an importation flow to the model to simulate the initial seeding of infectious individuals.
    This is used to introduce the disease into the population at any time of the simulation.

    Args:
        model: The compartmental model to which the infectious seed is to be added.
    """
    seed_args = [
        Time,
        Parameter("seed_time"),
        Parameter("seed_duration"),
        Parameter("seed_num"),
    ]
    voc_seed_func = Function(triangle_wave_func, seed_args)
    model.add_importation_flow(
        "seed_infectious",
        voc_seed_func,
        "infectious",
        split_imports=True,
    )
