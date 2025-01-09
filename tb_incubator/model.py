from typing import List, Dict
from summer2 import Overwrite, AgeStratification, Multiply, CompartmentalModel
from summer2.parameters import Parameter, Time, Function
from summer2.functions.time import get_sigmoidal_interpolation_function, get_linear_interpolation_function
from tb_incubator.input import get_death_rates, get_population_entry_rate, load_genexpert_util, load_targets
import tb_incubator.constants as const
from tb_incubator.utils import get_average_sigmoid, tanh_based_scaleup, triangle_wave_func
from tb_incubator.outputs import request_model_outputs
import numpy as np

compartments = const.compartments
infectious_compartments = const.infectious_compartments
age_strata = const.age_strata
model_times = const.model_times
agegroup_request = const.agegroup_request

def build_model(
    params: Dict[str, any],
    improved_detection: bool = True,
    xpert_sensitivity: bool = True,
    covid_effects: bool = True
) -> CompartmentalModel:
    """
    Builds and returns a compartmental model for epidemiological studies, incorporating
    various flows and stratifications based on age.

    Args:
        params: Dictionary of parameters with fixed values
        xpert_sensitivity: Whether to include GeneXpert sensitivity for detection multiplier
        covid_effect: Whether to include COVID-related reduction and post-COVID detection improvement
        improved_detection: Whether to include the improved detection multiplier after COVID-19 pandemic

    Returns:
        A configured CompartmentalModel object.
    """
    desc  = []

    model = CompartmentalModel(
        times=model_times,
        compartments=compartments,
        infectious_compartments=infectious_compartments,
    )

    desc.append(
        "We used the [summer framework](https://summer2.readthedocs.io/en/latest/) "
        "to construct a compartmental model of tuberculosis (TB) dynamics. "
        f"The base model consists of {len(compartments)} compartments: {', '.join([comp.replace('_', ' ') for comp in compartments])}--"
        "with flows added to represent the transitions and interactions between compartments. "
        "The susceptible (S) compartment includes individuals who have never had TB and are at risk "
        "of being infected by Mycobacterium tuberculosis. "
        "Latent TB infection is modelled using two compartments: early latent (E) and late latent (L). "
        "Latently infected individuals who progress to active TB move to the infectious (I) compartment. "
        "Those who recover through self-recovery are transferred to the recovery (R) compartment.\n\n"
    )


    model.set_initial_population({"susceptible": Parameter("start_population_size")}) # set initial population
    seed_infectious(model) # seed infectious individuals

    desc.append(
        f"Our model predicts the TB dynamics from {model_times[0]} to {model_times[1]}. "
        f"We mostly used estimates from previous study [@ragonnet2022] to inform TB progression "
        "and natural history of TB. "
        "We also fitted some parameters to local data on TB notifications [@whotb2023] and prevalence [@indoprevsurv2015], while "
        "considering uncertainty around TB progression parameters (see @tbl-params). "
        "Initially, we introduce a small number of population and seed infectious individuals to the model. "
    )

    # Demographic transitions
    model.add_universal_death_flows("population_death", 1.0)  # later adjusted by age
    model.add_replacement_birth_flow("replacement_birth", "susceptible")
    entry_rate, description = get_population_entry_rate(model_times) # calculate population entry rates

    desc.append(
        "Births are modelled using a time-variant function of the population entry rate. "
        "The entry rate was calculated by dividing the yearly population difference by the duration of the run-in period. "
        "Time-varying and age-specific non-TB-related mortality was applied to all compartments to represent deaths from "
        "non-TB causes. Estimates from the United Nations’ World Population Prospects [@unwpp2024] were used as reference data.\n\n"
    )

    # TB natural history
    
    for source in infectious_compartments:
        model.add_death_flow("TB_death", Parameter("death_rate"), source)
    
    for source in infectious_compartments:
        model.add_transition_flow("self_recovery", Parameter("self_recovery_rate"), source, "recovered")

    add_infection_flow(model) # add infection flow
    add_latency_flow(model) # add latency flow

    desc.append(
        "We use estimates reported in a previous study [@ragonnet2022] for TB-specific mortality, self-recovery rate, and "
        "age-specific infectiousness to inform the TB dynamics. "
        "Reinfection was illustrated in two different ways: "
        "flows from late latent (L) to early latent compartment (E) and "
        "from individuals who have recovered from TB (R) to early latent. "
        "Both pathways can be adjusted to reflect different reinfection risks compared to infection-naïve individuals. "
        "Progression flows from latent compartments to infectious compartment are also implemented to model the progression from individuals "
        "with latent infection to active TB. "
    )

    # Age-stratification
    strat = get_age_strat(params)
    model.stratify_with(strat)
    desc.append(
        f"We stratified the model based on {len(age_strata)} age groups: {', '.join(f'{start}-{end}' for start, end in agegroup_request)}. "
        "Age group-specific adjustments were applied for population death flows, latency flows, and infectiousness."
    )

    model.add_importation_flow( # Add births as additional entry rate, (split imports in case the susceptible compartments are further stratified later)
        "births", entry_rate, dest="susceptible", split_imports=True, dest_strata={"age": "0"}
    )

    # Detection
    detection_func = Function(
        tanh_based_scaleup,
        [
            Time,
            Parameter("screening_scaleup_shape"),
            Parameter("screening_inflection_time"),
            0.0,
            1.0 / Parameter("time_to_screening_end_asymp")
        ]
    )
    
    ## improved detection
    if improved_detection:
        detection_improvement = get_linear_interpolation_function([2017, 2024], [1.0, Parameter("detection_multiplier")])
        detection_func = detection_func * detection_improvement

    ## xpert sensitivity
    sensitivity = Parameter("base_sensitivity")
    if xpert_sensitivity:
        utilisation = load_genexpert_util()
        genexpert_util = get_sigmoidal_interpolation_function(utilisation.index, utilisation)
        genexpert_improvement = (1.0 - Parameter("base_sensitivity")) * Parameter("genexpert_sensitivity") * genexpert_util
        sensitivity += genexpert_improvement

        model.request_track_modelled_value("genexpert_util", genexpert_util)
    
    detection_func = detection_func * sensitivity
    model.request_track_modelled_value("sensitivity", sensitivity)

    
    ## covid effects
    if covid_effects:
        covid_impacts = get_sigmoidal_interpolation_function(
            [2019.0, 2020.0, 2022.0], [1.0, 1.0 - Parameter("detection_reduction"), Parameter("post_covid_improvement")]
        )
        detection_func = detection_func * covid_impacts
        model.request_track_modelled_value("covid_effects", covid_impacts)

        ### post-COVID sustained improvement
        sustained_improvement = get_linear_interpolation_function([2022.0, model_times[-1]], [1.0, Parameter("sustained_improvement")])
        detection_func = detection_func * sustained_improvement
    
    
    model.add_transition_flow("detection", detection_func, "infectious", "recovered")

    desc.append(
        "The detection rate refers to the progression of individuals with active TB (I) "
        "to the recovered (R) compartment, based on the assumption that detected individuals receive "
        "immediate treatment upon diagnosis, leading to their recovery. "
        "Furthermore, we implement changes in the diagnostic algorithm to model the improved "
        "diagnostic test (GeneXpert) utilisation. "
        "We assume that utilisation is proportional to the number of confirmed cases identified by GeneXpert. "
        #"To inform the time-variant proportion of utilisation, we use Indonesia's Ministry of Health GeneXpert utilisation data from 2016 to 2022 [@moh2022]. "
        #"This proportion is multiplied by the diagnostic sensitivity and the potential improvement "
        #"in sensitivity to reflect the enhancements of the diagnostic test. "
        "The calculated improvement in diagnostic "
        "sensitivity is then applied to the following year's data. \n\n"
    )

    # Request model outputs
    request_model_outputs(model)
    model.request_track_modelled_value("detection", detection_func) # Additional output
    
    final_desc = "".join(desc)

    return model, final_desc


# Add latency structures
def add_latency_flow(model):
    latency_flows = [
        ["stabilisation", "early_latent", "late_latent"],
        ["early_activation", "early_latent", "infectious"],
        ["late_activation", "late_latent", "infectious"],
    ]

    for flow, source, dest in latency_flows:
        model.add_transition_flow(flow, 1.0, source, dest)

# Add infection flow
def add_infection_flow(model):
    infection_flows = [
        ["susceptible", None],
        ["late_latent", "rr_infection_latent"],
        ["recovered", "rr_infection_recovered"],
    ]

    contact_rate = Parameter("contact_rate") #* (
        #get_linear_interpolation_function(
            #[2019.0, 2020.0, 2022], [1.0, 1 - Parameter("contact_reduction"), 1.0]
        #)
    #)

    for origin, modifier in infection_flows:
        modifier = Parameter(modifier) if modifier else 1.0
        rate = contact_rate * modifier
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
                Parameter("progression_multiplier") * age_val
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
    seed_func = Function(triangle_wave_func, seed_args)
    model.add_importation_flow(
        "seed_infectious",
        seed_func,
        "infectious",
        split_imports=True,
    )

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
