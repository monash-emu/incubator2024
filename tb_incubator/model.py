from typing import List, Dict
from summer2 import Overwrite, AgeStratification, Multiply, CompartmentalModel
from summer2.parameters import Parameter, Time, Function
from summer2.functions.time import get_sigmoidal_interpolation_function
from tb_incubator.input import get_death_rates, get_population_entry_rate, load_genexpert_util
import tb_incubator.constants as const
from tb_incubator.utils import get_average_sigmoid, tanh_based_scaleup, triangle_wave_func
from tb_incubator.outputs import request_model_outputs

compartments = const.compartments
infectious_compartments = const.infectious_compartments
age_strata = const.age_strata
model_times = const.model_times
agegroup_request = const.agegroup_request

def build_model(params: Dict[str, any]) -> CompartmentalModel:
    """
    Builds and returns a compartmental model for epidemiological studies, incorporating
    various flows and stratifications based on age.

    Args:
        params: Dictionary of parameters with fixed values.

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
        "Those who recover through self-recovery are transferred to the recovery (R) compartment. "
        "We also account for under-reporting by modelling transitions of some individuals from the infectious (I)"
        " compartment to the missed (M) compartment.\n\n"
    )

    # Set initial population
    model.set_initial_population({"susceptible": Parameter("start_population_size")})

    # Seed infectious individuals
    seed_infectious(model)

    desc.append(
        f"Our model predicts the TB dynamics from {model_times[0]} to {model_times[1]}. "
        f"We mostly used estimates from previous study [@ragonnet2022] to inform TB progression "
        "and natural history of TB. "
        "We also fitted some parameters to local data on TB notifications [@whotb2023] and prevalence [@indoprevsurv2015], while "
        "considering uncertainty around TB progression parameters (see @tbl-params). "
        "Initially, we introduce a small number of population and seed infectious individuals to the model. "
    )

    # Demographic transitions
    model.add_universal_death_flows(
        "population_death", Parameter("universal_death"))  # Placeholder to overwrite later
    model.add_replacement_birth_flow("replacement_birth", "susceptible")

    desc.append(
        "Births are modelled using a time-variant function of the population entry rate. "
        "The entry rate was calculated by dividing the yearly population difference by the duration of the run-in period. "
        "Time-varying and age-specific non-TB-related mortality was applied to all compartments to represent deaths from "
        "non-TB causes. Estimates from the United Nations’ World Population Prospects [@unwpp2024] were used as reference data. "
        "We also assume that the population is closed, where the number of births replaces "
        "the total number of deaths each year. \n\n"
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

    utilisation = load_genexpert_util()
    genexpert_util = get_sigmoidal_interpolation_function(utilisation.index, utilisation)
    genexpert_improvement = (1.0 - Parameter("base_sensitivity")) * Parameter("genexpert_sensitivity") * genexpert_util
    sensitivity = Parameter("base_sensitivity") + genexpert_improvement
    
    model.add_transition_flow("detection", sensitivity * detection_func, "infectious", "recovered")
    model.add_transition_flow("missing", (1.0-sensitivity) * detection_func, "infectious", "missed")

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
        "sensitivity is then applied to the following year's data. On the other hand, the flow of under-reporting, "
        "or “missing” TB cases, refers to the potential sensitivity bias multiplied by the detection function.  \n\n"
    )

    # TB natural history
    
    for source in infectious_compartments:
        model.add_death_flow("TB_death", Parameter("death_rate"), source)
    
    for source in infectious_compartments:
        model.add_transition_flow("self_recovery", Parameter("self_recovery_rate"), source, "recovered")

    # Infection 
    add_infection_flow(model)

    desc.append(
        "We use estimates reported in a previous study [@ragonnet2022] for TB-specific mortality, self-recovery rate, and "
        "age-specific infectiousness to inform the TB dynamics. "
        "Reinfection was illustrated in two different ways: "
        "flows from late latent (L) to early latent compartment (E) and "
        "from individuals who have recovered from TB (R) to early latent. "
        "Both pathways can be adjusted to reflect different reinfection risks compared to infection-naïve individuals. "
    )

    # Latency
    add_latency_flow(model)

    desc.append(
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

    # Calculate population entry rates
    entry_rate, description = get_population_entry_rate(model_times)

    # Add births as additional entry rate
    # (split imports in case the susceptible compartments are further stratified later)
    model.add_importation_flow(
        "births", entry_rate, dest="susceptible", split_imports=True, dest_strata={"age": "0"}
    )

    # Request model output
    request_model_outputs(model)

    final_desc = "".join(desc)

    return model, final_desc


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
    voc_seed_func = Function(triangle_wave_func, seed_args)
    model.add_importation_flow(
        "seed_infectious",
        voc_seed_func,
        "infectious",
        split_imports=True,
    )


# organ stratification
def get_organ_strat(
    fixed_params: Dict[str, any],
    detection_reduction,
    improved_detection_multiplier = None,
) -> Stratification:
    """
    Creates and configures an organ stratification for the model. This includes defining
    adjustments for infectiousness, infection death rates, and self-recovery rates based
    on organ involvement, as well as adjusting progression rates by organ using requested
    incidence proportions.

    Args:
        infectious_compartments: A list of names of compartments that can transmit infection.
        organ_strata: A list of organ strata names for stratification (e.g., 'lung', 'extrapulmonary').
        fixed_params: A dictionary containing fixed parameters for the model, including
                      multipliers for infectiousness by organ, death rates by organ, and
                      incidence proportions for different organ involvements.

    Returns:
        A Stratification object configured with organ-specific adjustments.
    """
    strat = Stratification("organ", organ_strata, infectious_compartments)

    # Define different detection rates by organ status
    detection_adjs = {}
    detection_func = Function(
        tanh_based_scaleup,
        [
            Time,
            Parameter("screening_scaleup_shape"),
            Parameter("screening_inflection_time"),
            0.0,
            1.0 / Parameter("time_to_screening_end_asymp"),
        ],
    )
    detection_func*= (get_sigmoidal_interpolation_function([2020.0, 2021.0, 2022.0], [1.0, 1.0 - Parameter("detection_reduction"), 1.0], curvature=8) if detection_reduction else 1.0)
    if improved_detection_multiplier is not None:
        assert isinstance(improved_detection_multiplier, float) and improved_detection_multiplier > 0, "improved_detection_multiplier must be a positive float."
        detection_func *= get_linear_interpolation_function([2025.0, 2035.0], [1.0, improved_detection_multiplier])

    # Detection, self-recovery and infect death
    inf_adj, detection_adjs, infect_death_adjs, self_recovery_adjustments = {}, {}, {}, {}
    for organ_stratum in organ_strata:
        # Define infectiousness adjustment by organ status
        inf_adj_param = fixed_params[f"{organ_stratum}_infect_multiplier"]
        inf_adj[organ_stratum] = Multiply(inf_adj_param)

        # Define different natural history (self-recovery) by organ status
        param_strat = "smear_negative" if organ_stratum == "extrapulmonary" else organ_stratum
        self_recovery_adjustments[organ_stratum] = Overwrite(Parameter(f"{param_strat}_self_recovery"))

        # Adjust detection by organ status
        param_name = f"passive_screening_sensitivity_{organ_stratum}"
        detection_adjs[organ_stratum] = fixed_params[param_name] * detection_func

        # Calculate infection death adjustment using detection adjustments
        infect_death_adjs[organ_stratum] = Parameter(f"{param_strat}_death_rate")
       

    # Apply the Multiply function to the detection adjustments
    detection_adjs = {k: Multiply(v) for k, v in detection_adjs.items()}
    infect_death_adjs = {k: Overwrite(v) for k, v in infect_death_adjs.items()}

    # Set flow and infectiousness adjustments
    strat.set_flow_adjustments("detection", detection_adjs)
    strat.set_flow_adjustments("self_recovery", self_recovery_adjustments)
    strat.set_flow_adjustments("infect_death", infect_death_adjs)
    for comp in infectious_compartments:
        strat.add_infectiousness_adjustments(comp, inf_adj)

    splitting_proportions = {
        "smear_positive": fixed_params["incidence_props_pulmonary"]
        * fixed_params["incidence_props_smear_positive_among_pulmonary"],
        "smear_negative": fixed_params["incidence_props_pulmonary"]
        * (1.0 - fixed_params["incidence_props_smear_positive_among_pulmonary"]),
        "extrapulmonary": 1.0 - fixed_params["incidence_props_pulmonary"],
    }
    for flow_name in ["early_activation", "late_activation"]:
        flow_adjs = {k: Multiply(v) for k, v in splitting_proportions.items()}
        strat.set_flow_adjustments(flow_name, flow_adjs)

    return strat