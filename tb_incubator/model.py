from typing import Dict
from summer2 import CompartmentalModel
from summer2.parameters import Parameter, Time, Function
from .input import get_population_entry_rate, load_param_info
import tb_incubator.constants as const
from .utils import triangle_wave_func
from .outputs import request_model_outputs
from .strat_age import get_age_strat
from .strat_organ import get_organ_strat
from .detection import get_detection_func

compartments = const.COMPARTMENTS
infectious_compartments = const.INFECTIOUS_COMPARTMENTS
age_strata = const.AGE_STRATA
organ_strata = const.ORGAN_STRATA
model_times = const.MODEL_TIMES
agegroup_request = const.AGEGROUP_REQUEST

param_info = load_param_info()
fixed_params = param_info["value"]

def build_model(
    params: Dict[str, any],
    xpert_improvement: bool = True,
    covid_effects: Dict[str, bool] = None,
    xpert_util_target: float = None,
    improved_detection_multiplier: float = None
) -> CompartmentalModel:
    """
    Builds and returns a compartmental model for epidemiological studies, incorporating
    various flows and stratifications based on age.

    Args:
        params: Dictionary of parameters with fixed values
        xpert_improvement: Whether to include improvement of GeneXpert utilisation for detection multiplier
        covid_effect: Whether to include COVID-related reduction and post-COVID detection improvement
    Returns:
        A configured CompartmentalModel object.
    """
    if covid_effects is None:
        covid_effects = {
            "detection_reduction": False,
        }
    
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
        "Those who recover through self-recovery are transferred to the recovery (R) compartment."
        "Individuals who are detected are assumed to undergo treatment and fully recovered, hence they move from I to R.\n\n"
    )


    model.set_initial_population({"susceptible": Parameter("start_population_size")}) # set initial population
    seed_infectious(model) # seed infectious individuals

    desc.append(
        f"The model is run from {model_times[0]} to {model_times[1]}, with an aim to capture the dynamics between the mid-1990s and 2024."
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
        model.add_death_flow("TB_death", 1.0, source) # later adjusted by organ status
    
    for source in infectious_compartments:
        model.add_transition_flow("self_recovery", 1.0, source, "recovered") # later adjusted by organ status

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
    # Detection and treatment commencement
    model.add_transition_flow("treatment_commencement", 1.0, "infectious", "recovered")

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

    # Organ-stratification
    detection_func, base_detection, diagnostic_capacity, diagnostic_improvement = get_detection_func(xpert_improvement, covid_effects, xpert_util_target, improved_detection_multiplier)
    organ_strat= get_organ_strat(fixed_params, detection_func)
    model.stratify_with(organ_strat)

    model.request_track_modelled_value("base_detection", base_detection)
    model.request_track_modelled_value("diagnostic_capacity", diagnostic_capacity)
    model.request_track_modelled_value("diagnostic_improvement", diagnostic_improvement)
    model.request_track_modelled_value("final_detection", detection_func)

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

def add_infection_flow(model):
    """
    Adds infection flows from various compartments to early latent TB.
    
    Args:
        model: The compartmental model
        covid_effects: Dictionary with COVID effect settings
    """        
    infection_flows = [
        ["susceptible", None],
        ["late_latent", "rr_infection_latent"],
        ["recovered", "rr_infection_recovered"],
    ]
    
    contact_rate = Parameter("contact_rate")
    #if covid_effects["contact_reduction"]:
    #    contact_rate *= get_linear_interpolation_function(
    #        [2019.0, 2020.0, 2022.0], [1.0, 1.0 - Parameter("contact_reduction"), 1.0]
    #    )
        
    for origin, modifier in infection_flows:
        modifier = Parameter(modifier) if modifier else 1.0
        rate = contact_rate * modifier
        name = f"infection_from_{origin}"
        model.add_infection_frequency_flow(name, rate, origin, "early_latent")


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
