from typing import List, Dict
from summer2 import Overwrite, AgeStratification, Multiply
from summer2.parameters import Parameter
from summer2.functions.time import get_sigmoidal_interpolation_function
from .input import get_death_rates
import tb_incubator.constants as const
from .utils import get_average_sigmoid

compartments = const.compartments
infectious_compartments = const.infectious_compartments
age_strata = const.age_strata
agegroup_request = const.agegroup_request

# Age stratification
def get_age_strat(params: Dict[str, any]) -> AgeStratification:
    """
    Creates and configures an age stratification for a compartmental model.

    Args:
        params: A dictionary of fixed parameters for the model, which includes
                keys for age-specific latency adjustments, infectiousness switch ages,
                and parameters for BCG effects and treatment outcomes.

    Returns:
        AgeStratification: An object representing the configured age stratification for the model.
    """
    strat = AgeStratification("age", age_strata, compartments)

    # Set universal death rates
    set_popdeath_adjs(age_strata, strat)

    # Set age-specific latency rate
    set_latency_adjs(params, age_strata, strat)

    # Set age-adjusted infectiousness
    add_infectiousness_adjs(infectious_compartments, params, age_strata, strat)

    return strat

def set_popdeath_adjs(age_strata: List[int], strat: AgeStratification):
    deathrate_df, description = get_death_rates()
    death_adjs = {}
    for age in age_strata:
        years = deathrate_df.index
        rates = deathrate_df[age]
        pop_death_func = get_sigmoidal_interpolation_function(years, rates)
        death_adjs[str(age)] = Overwrite(pop_death_func)

    strat.set_flow_adjustments("population_death", death_adjs)

def set_latency_adjs(params: Dict[str, any], age_strata: List[int], strat: AgeStratification):
    for flow_name, latency_params in params["age_latency"].items():
        adjs = {}
        for age in age_strata:
            param_age_bracket = max([k for k in latency_params if k <= age])
            age_val = latency_params[param_age_bracket]

            adj = (
                1.0 * age_val
                if "_activation" in flow_name
                else age_val
            )
            adjs[str(age)] = adj

        adjs = {k: Overwrite(v) for k, v in adjs.items()}
        strat.set_flow_adjustments(flow_name, adjs)

def add_infectiousness_adjs(
    infectious_compartments: List[str],
    params: Dict[str, any],
    age_strata: List[int],
    strat: AgeStratification,
):
    inf_switch_age = params["age_infectiousness_switch"]
    for comp in infectious_compartments:
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