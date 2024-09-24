from typing import List, Dict
from summer2 import Overwrite, AgeStratification, Multiply, CompartmentalModel
from summer2.parameters import Parameter, Time, Function
from summer2.functions.time import get_sigmoidal_interpolation_function
from tb_incubator.input import get_death_rates, get_population_entry_rate
from tb_incubator.constants import set_project_base_path, agegroup_request
from tb_incubator.utils import get_average_sigmoid, tanh_based_scaleup, triangle_wave_func
from tb_incubator.outputs import request_model_outputs


project_paths = set_project_base_path("../tb_incubator")
data_path = project_paths["DATA_PATH"]

def build_model(
    compartments: List[str],
    #latent_compartments: List[str],
    infectious_compartments: List[str],
    age_strata: List[int],
    params: Dict[str, any],
    model_times: List[int],
) -> CompartmentalModel:
    """
    Builds and returns a compartmental model for epidemiological studies, incorporating
    various flows and stratifications based on age.

    Args:
        compartments: List of compartment names in the model.
        infectious_compartments: List of infectious compartment names.
        age_strata: List of age groups for stratification.
        params: Dictionary of parameters with fixed values.
        model_times: List of start and end periods of the model

    Returns:
        A configured CompartmentalModel object.
    """
    model = CompartmentalModel(
        times=model_times,
        compartments=compartments,
        infectious_compartments=infectious_compartments,
    )

    # Set initial population
    model.set_initial_population({"susceptible": Parameter("start_population_size")})

    # Seed infectious individuals
    seed_infectious(model)

    # Demographic transitions
    model.add_universal_death_flows(
        "population_death", Parameter("universal_death"))  # Placeholder to overwrite later
    model.add_replacement_birth_flow("replacement_birth", "susceptible")

    # Detection
    detection_func = Function(tanh_based_scaleup,
                        [
                            Time,
                            Parameter("screening_scaleup_shape"),
                            Parameter("screening_inflection_time"),
                            0.0,
                            1.0 / Parameter("time_to_screening_end_asymp")
                        ])
    
    model.add_transition_flow("detection", Parameter("algorithm_sensitivity") * detection_func, "infectious", "recovered")

    model.add_transition_flow("missing", (1.0-Parameter("algorithm_sensitivity")) * detection_func, "infectious", "missed")

    # TB natural history
    
    for source in infectious_compartments:
        model.add_death_flow("TB_death", Parameter("death_rate"), source)
    
    for source in infectious_compartments:
        model.add_transition_flow("self_recovery", Parameter("self_recovery_rate"), source, "recovered")



    # Infection 
    add_infection_flow(model)

    # Latency
    add_latency_flow(model)

    # Age-stratification
    strat = get_age_strat(
        compartments,
        infectious_compartments,
        age_strata,
        params,
    )
    model.stratify_with(strat)

    # Calculate population entry rates
    entry_rate, description = get_population_entry_rate(1850)

    # Add births as additional entry rate
    # (split imports in case the susceptible compartments are further stratified later)
    model.add_importation_flow(
        "births", entry_rate, dest="susceptible", split_imports=True, dest_strata={"age": "0"}
    )

    # Request model output
    request_model_outputs(model)

    desc = (
        "We used the [summer framework](https://summer2.readthedocs.io/en/latest/) "
        "to construct a compartmental model of tuberculosis (TB) dynamics. "
        f"The base model consists of {len(compartments)} compartments: {', '.join([comp.replace('_', ' ') for comp in compartments])}--"
        "with flows added to represent the transitions and interactions between compartments. "
        f"We stratified the model based on {len(age_strata)} age groups: {', '.join(f'{start}-{end}' for start, end in agegroup_request)}. "
        "Age group-specific adjustments were applied for population death flows, latency flows, and infectiousness."
    )

    return model, desc


# Age stratification
def get_age_strat(
    compartments: List[str],
    infectious_compartments: List[str],
    age_strata: List[int],
    params: Dict[str, any],
) -> AgeStratification:
    """
    Creates and configures an age stratification for a compartmental model.

    Args:
        compartments: A list of the names of all compartments in the model.
        infectious: A list of the names of infectious compartments in the model.
        age_strata: A list of age strata (as integers) for the model.
        death_df: A DataFrame containing death rates by age.
        fixed_params: A dictionary of fixed parameters for the model, which includes
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


# Add latency structures


def add_latency_flow(model):
    latency_flows = [
        ["stabilisation", "early_latent", "late_latent"],
        ["early_activation", "early_latent", "infectious"],
        ["late_activation", "late_latent", "infectious"],
    ]

    for flow, source, dest in latency_flows:
        model.add_transition_flow(flow, 1.0, source, dest)



def add_infection_flow(model):
    infection_flows = [
        ["susceptible", None],
        ["late_latent", "rr_infection_latent"],
        ["recovered", "rr_infection_recovered"],
    ]

    for origin, modifier in infection_flows:
        modifier = Parameter(modifier) if modifier else 1.0
        rate = Parameter("contact_rate") * modifier
        name = f"infection_from_{origin}"
        model.add_infection_frequency_flow(name, rate, origin, "early_latent")


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
                params["progression_multiplier"] * age_val
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
        Parameter("seed_rate"),
    ]
    voc_seed_func = Function(triangle_wave_func, seed_args)
    model.add_importation_flow(
        "seed_infectious",
        voc_seed_func,
        "infectious",
        split_imports=True,
    )
