from typing import List, Dict
from pandas import DataFrame
from summer2 import Overwrite, AgeStratification, Multiply
from summer2.parameters import Function, Parameter
from summer2.functions.time import get_sigmoidal_interpolation_function, get_time_callable, get_linear_interpolation_function
from .constants import COMPARTMENTS, INFECTIOUS_COMPARTMENTS, AGE_STRATA
from .utils import get_average_sigmoid, calculate_treatment_outcomes
from .input import get_birth_rate, get_death_rate, process_death_rate

# Age stratification
def get_age_strat(fixed_params: Dict[str, any], tsr_target: float = None) -> AgeStratification:
    """
    Creates and configures an age stratification for a compartmental model.

    Args:
        fixed_params: A dictionary of fixed parameters for the model, which includes
                keys for age-specific latency adjustments, infectiousness switch ages,
                and parameters for treatment outcomes.
        tsr_target: Optional target TSR to reach by 2030 (e.g., 0.90 for 90%, 0.95 for 95%).
                   If None, uses baseline TSR trajectory. Only scales up if target > current.

    Returns:
        AgeStratification: An object representing the configured age stratification for the model.
    """
    strat = AgeStratification("age", AGE_STRATA, COMPARTMENTS)

    # Set universal death rates
    births = get_birth_rate()
    deaths = get_death_rate()
    death_df = process_death_rate(deaths, AGE_STRATA, births.index)

    universal_death_funcs, death_adjs = {}, {}
    for age in AGE_STRATA:
        universal_death_funcs[age] = get_sigmoidal_interpolation_function(
            death_df.index, death_df[age]
        )
        death_adjs[str(age)] = Overwrite(universal_death_funcs[age])
    strat.set_flow_adjustments("universal_death", death_adjs)

    # Set age-specific latency rate
    set_latency_adjs(fixed_params, strat)

    # Set age-adjusted infectiousness
    set_infectiousness_adjs(fixed_params, strat)

    # Set age-adjusted treatment outcomes
    time_variant_tsr = get_sigmoidal_interpolation_function(
        list(fixed_params["time_variant_tsr"].keys()),
        list(fixed_params["time_variant_tsr"].values()),
    )

    # Scenario intervention: gradual improvement to target by 2030
    if tsr_target is not None:
        tsr_callable = get_time_callable(time_variant_tsr)
        current_tsr = float(tsr_callable(2025.0))

        if tsr_target > current_tsr:
            # Create scaling factor (same pattern as Xpert)
            tsr_scale_factor = get_linear_interpolation_function(
                [2025.0, 2027.0],
                [1.0, tsr_target / current_tsr]
            )
            time_variant_tsr *= tsr_scale_factor  # Apply scaling


    treatment_recovery_funcs, treatment_death_funcs, treatment_relapse_funcs = (
        {},
        {},
        {},
    )

    for age in AGE_STRATA:
        natural_death_rate = universal_death_funcs[age]
        treatment_outcomes = Function(
            calculate_treatment_outcomes,
            [
                fixed_params["treatment_duration"],
                fixed_params["prop_death_among_negative_tx_outcome"],
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

def set_latency_adjs(fixed_params: Dict[str, any], strat: AgeStratification):
    for flow_name, latency_fixed_params in fixed_params["age_latency"].items():
        adjs = {}
        for age in AGE_STRATA:
            param_age_bracket = max([k for k in latency_fixed_params if k <= age])
            age_val = latency_fixed_params[param_age_bracket]

            adj = (
                Parameter("progression_multiplier") * age_val
                if "late_activation" in flow_name
                else age_val
            )
            adjs[str(age)] = adj

        adjs = {k: Overwrite(v) for k, v in adjs.items()}
        strat.set_flow_adjustments(flow_name, adjs)

def set_infectiousness_adjs(
    fixed_params: Dict[str, any],
    strat: AgeStratification,
):
    inf_switch_age = fixed_params["age_infectiousness_switch"]
    for comp in INFECTIOUS_COMPARTMENTS:
        inf_adjs = {}
        for i, age_low in enumerate(AGE_STRATA):
            if age_low == AGE_STRATA[-1]:
                average_infectiousness = 1.0
            else:
                age_high = AGE_STRATA[i + 1]
                average_infectiousness = get_average_sigmoid(age_low, age_high, inf_switch_age)
            
            if comp == "on_treatment":
                average_infectiousness *= fixed_params["on_treatment_infect_multiplier"]
            # Update the adjustments dictionary for the current age group.
            inf_adjs[str(age_low)] = Multiply(average_infectiousness)

        strat.add_infectiousness_adjustments(comp, inf_adjs)
