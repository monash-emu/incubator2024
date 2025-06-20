from typing import List, Dict
from summer2 import Overwrite, AgeStratification, Multiply
from summer2.parameters import Function
from summer2.functions.time import get_sigmoidal_interpolation_function
from .input import get_death_rates
import tb_incubator.constants as const
from .utils import get_average_sigmoid, calculate_treatment_outcomes

compartments = const.COMPARTMENTS
infectious_compartments = const.INFECTIOUS_COMPARTMENTS
age_strata = const.AGE_STRATA
agegroup_request = const.AGEGROUP_REQUEST

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
    universal_death_funcs, death_adjs = get_universal_death_adjs(age_strata)

    # Set universal death rates
    strat.set_flow_adjustments("universal_death", death_adjs)

    # Set age-specific latency rate
    set_latency_adjs(params, age_strata, strat)

    # Set age-adjusted infectiousness
    set_infectiousness_adjs(infectious_compartments, params, age_strata, strat)

    # Set age-adjusted treatment outcomes
    time_variant_tsr = get_sigmoidal_interpolation_function(
        list(params["time_variant_tsr"].keys()),
        list(params["time_variant_tsr"].values()),
    )

    treatment_recovery_funcs, treatment_death_funcs, treatment_relapse_funcs = (
        {},
        {},
        {},
    )

    for age in age_strata:
        natural_death_rate = universal_death_funcs[age]
        treatment_outcomes = Function(
            calculate_treatment_outcomes,
            [
                params["treatment_duration"],
                params["prop_death_among_negative_tx_outcome"],
                natural_death_rate,
                time_variant_tsr,
            ],
        )
        treatment_recovery_funcs[str(age)] = Multiply(treatment_outcomes[0])
        treatment_death_funcs[str(age)] = Multiply(treatment_outcomes[1])
        treatment_relapse_funcs[str(age)] = Multiply(treatment_outcomes[2])
        
    strat.set_flow_adjustments("treatment_recovery", treatment_recovery_funcs)
    strat.set_flow_adjustments("treatment_death", treatment_death_funcs)
    strat.set_flow_adjustments("relapse", treatment_relapse_funcs)

    return strat

def get_universal_death_adjs(age_strata: List[int]):
    deathrate_df, description = get_death_rates()
    universal_death_funcs, death_adjs = {}, {}
    for age in age_strata:
        years = deathrate_df.index
        rates = deathrate_df[age]
        universal_death_funcs[age] = get_sigmoidal_interpolation_function(years, rates)
        death_adjs[str(age)] = Overwrite(universal_death_funcs[age])
    
    return universal_death_funcs, death_adjs

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

def set_infectiousness_adjs(
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
            
            if comp == "on_treatment":
                average_infectiousness *= params["on_treatment_infect_multiplier"]
            # Update the adjustments dictionary for the current age group.
            inf_adjs[str(age_low)] = Multiply(average_infectiousness)

        strat.add_infectiousness_adjustments(comp, inf_adjs)
