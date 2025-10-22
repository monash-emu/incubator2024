from typing import Dict, Optional, Any
from summer2 import CompartmentalModel
from summer2.parameters import Parameter, Time, Function
from summer2.functions.time import get_linear_interpolation_function
from .input import get_population_entry_rate
from .constants import COMPARTMENTS, INFECTIOUS_COMPARTMENTS, MODEL_TIMES, ImplementCDR, CUMULATIVE_OUTPUT_START_TIME
from .utils import triangle_wave_func
from .outputs import request_model_outputs
from .strat_age import get_age_strat
from .strat_organ import get_organ_strat
from .detection import get_detection_func

PLACEHOLDER_PARAM = 1.0

def build_model(
    fixed_params: Dict[str, Any],
    covid_effects: bool = True,
    apply_diagnostic_capacity: bool = True,
    xpert_improvement: bool = True,
    apply_cdr: ImplementCDR = ImplementCDR.NONE,
    improved_detection_multiplier: Optional[float] = None,
    xpert_util_target: Optional[float] = None,
    acf_screening_rate: Optional[Dict[float, float]] = None,
    acf_sensitivity: Optional[float] = None,
    tsr_target: Optional[float] = None,
) -> CompartmentalModel:
    """
    Builds and returns a compartmental model for epidemiological studies, incorporating
    various flows and stratifications based on age.

    Args:
        fixed_params: Dictionary of parameters with fixed values
        covid_effects: Whether to include COVID-related reduction and post-COVID detection improvement
        apply_diagnostic_capacity: Whether to apply diagnostic capacity scaling
        xpert_improvement: Whether to include improvement of GeneXpert utilisation for detection multiplier
        apply_cdr: Case detection rate implementation strategy
        acf_screening_rate: Active case finding screening rates by time
        acf_sensitivity: Sensitivity for active case finding
        improved_detection_multiplier: Future detection improvement multiplier by 2030
        xpert_util_target: Target Xpert utilization rate for 2030 (0-1)
        tsr_target: Treatment success rate target
        
    Returns:
        A configured CompartmentalModel object.
    """
    model = CompartmentalModel(
        times=MODEL_TIMES,
        compartments=COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPARTMENTS,
    )

    model.set_initial_population({"susceptible": Parameter("start_population_size")}) # set initial population
    seed_infectious(model) # seed infectious individuals

    # Demographic transitions
    model.add_universal_death_flows("universal_death", PLACEHOLDER_PARAM)  # later adjusted by age
    model.add_replacement_birth_flow("replacement_birth", "susceptible")
    entry_rate = get_population_entry_rate(MODEL_TIMES) # calculate population entry rates

    # TB natural history
    model.add_death_flow("infect_death", PLACEHOLDER_PARAM, "infectious") # later adjusted by organ status
    model.add_transition_flow("self_recovery", PLACEHOLDER_PARAM, "infectious", "recovered") # later adjusted by organ status

    add_infection_flow(model) # add infection flow
    add_latency_flow(model) # add latency flow

    # Detection and treatment commencement
    model.add_transition_flow("detection", PLACEHOLDER_PARAM, "infectious", "on_treatment")
    add_treatment_related_outcomes(model)

    # Add ACF detection flow 
    if acf_screening_rate is not None:
        acf_detection_rate = calculate_acf_detection_rate(acf_screening_rate, acf_sensitivity)
        model.add_transition_flow("acf_detection", acf_detection_rate, "infectious", "on_treatment")

    # Age-stratification
    strat = get_age_strat(fixed_params, tsr_target)
    model.stratify_with(strat)

    model.add_importation_flow( # Add births as additional entry rate, (split imports in case the susceptible compartments are further stratified later)
        "births", entry_rate, dest="susceptible", split_imports=True, dest_strata={"age": "0"}
    )
    
    detection_func = get_detection_func(
        covid_effects=covid_effects,
        improved_detection_multiplier=improved_detection_multiplier,
        apply_diagnostic_capacity=apply_diagnostic_capacity,
        xpert_improvement=xpert_improvement,
        apply_cdr=apply_cdr,
        xpert_util_target=xpert_util_target
    )

    # Organ-stratification
    organ_strat = get_organ_strat(fixed_params, detection_func)
    model.stratify_with(organ_strat)
    model.request_track_modelled_value("final_detection", detection_func)

    # Request model outputs
    request_model_outputs(model, detection_func, acf_screening_rate, apply_cdr=apply_cdr, 
                          cumulative_output_start_time=CUMULATIVE_OUTPUT_START_TIME)

    return model

def add_treatment_related_outcomes(model: CompartmentalModel) -> None:
    treatment_outcomes_flows = [
        ("treatment_recovery", 1.0, "recovered"),
        ("relapse", 1.0, "infectious"),
    ]

    for flow_name, rate, destination in treatment_outcomes_flows:
        model.add_transition_flow(flow_name, rate, "on_treatment", destination)

    # Add death flow
    model.add_death_flow("treatment_death", PLACEHOLDER_PARAM, "on_treatment") 

def calculate_acf_detection_rate(
    acf_screening_rate: Dict[float, float],
    acf_sensitivity: Optional[float] = None, 
) -> Function:  
    """Calculate active case finding detection rate based on screening rate and sensitivity."""
    times = list(acf_screening_rate.keys())
    
    if acf_sensitivity is not None:
        sensitivity = acf_sensitivity
    else:
        sensitivity = Parameter("acf_sensitivity")
    
    acf_rate_vals = [
        screening_rate * sensitivity
        for screening_rate in acf_screening_rate.values()
    ]
    
    return get_linear_interpolation_function(times, acf_rate_vals)

# Add latency structures
def add_latency_flow(model: CompartmentalModel) -> None:
    latency_flows = [
        ["stabilisation", "early_latent", "late_latent"],
        ["early_activation", "early_latent", "infectious"],
        ["late_activation", "late_latent", "infectious"],
    ]

    for flow, source, dest in latency_flows:
        model.add_transition_flow(flow, 1.0, source, dest)

def add_infection_flow(model: CompartmentalModel) -> None:
    """
    Adds infection flows from various compartments to early latent TB.
    
    Args:
        model: The compartmental model
    """        
    infection_flows = [
        ["susceptible", None],
        ["late_latent", "rr_infection_latent"],
        ["recovered", "rr_infection_recovered"],
    ]
    
    contact_rate = Parameter("contact_rate")
        
    for origin, modifier in infection_flows:
        modifier = Parameter(modifier) if modifier else 1.0
        rate = contact_rate * modifier
        name = f"infection_from_{origin}"
        model.add_infection_frequency_flow(name, rate, origin, "early_latent")


def seed_infectious(model: CompartmentalModel) -> None:
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
    seed_func = Function(triangle_wave_func, seed_args)
    model.add_importation_flow(
        "seed_infectious",
        seed_func,
        "infectious",
        split_imports=True,
    )

