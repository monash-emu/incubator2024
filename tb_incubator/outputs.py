from summer2 import CompartmentalModel
from typing import List
from summer2.parameters import DerivedOutput, Time, Function
import tb_incubator.constants as const
import numpy as np
from summer2.functions.time import get_linear_interpolation_function
from summer2.parameters import Parameter

compartments = const.COMPARTMENTS
infectious_compartments = const.INFECTIOUS_COMPARTMENTS
organ_strata = const.ORGAN_STRATA
age_strata = const.AGE_STRATA
latent_compartments = const.LATENT_COMPARTMENTS


def request_model_outputs(
    model: CompartmentalModel,
):
    compartments = [c.name for c in model._original_compartment_names]

    # compartment sizes
    for c in compartments:
        model.request_output_for_compartments(f"comp_size_{c}", c)
    tot_pop = model.request_output_for_compartments("total_population", compartments)
    model.request_function_output("population_log", np.log(tot_pop))

    # latent percentage
    latent_pop = model.request_output_for_compartments(
        "latent_pop", latent_compartments
    )
    model.request_function_output("percentage_latent", latent_pop / tot_pop * 100.0)

    # death
    deaths = model.request_output_for_flow("mortality_infectious", "TB_death")
    model.request_function_output("mortality", 1e5 * deaths / tot_pop)

    # prevalence
    # Calculate and request prevalence of pulmonary
    for organ_stratum in organ_strata:
        model.request_output_for_compartments(
            f"infectious_sizeXorgan_{organ_stratum}",
            infectious_compartments,
            strata={"organ": organ_stratum},
            save_results=False,
        )

    pulmonary_outputs = [
        f"infectious_sizeXorgan_{organ_stratum}"
        for organ_stratum in ["smear_positive", "smear_negative"]
    ]

    pulmonary_pop_size = model.request_aggregate_output("pulmonary_pop_size", pulmonary_outputs)
    model.request_function_output("prevalence_pulmonary", 1e5 * pulmonary_pop_size / tot_pop)

    # total prevalence
    infectious_population_size = model.request_output_for_compartments(
        "infectious_population_size", infectious_compartments
    )
    prevalence = model.request_function_output("prevalence", 1e5 * infectious_population_size / tot_pop)
    model.request_function_output("prevalence_log", np.log(prevalence))

    # incidence
    timings = ["early_activation", "late_activation"]
    for t in timings:
        model.request_output_for_flow(t, t, save_results=False)
    inc_raw = model.request_aggregate_output("incidence_raw", timings)
    model.request_function_output("incidence", 1e5 * inc_raw / tot_pop)

    # notification
    # Get the treatment flow
    model.request_output_for_flow("treatment_commencement", "treatment_commencement")
    model.request_output_for_flow("acf_notification", "acf_detection")

    prop_reported_case = get_linear_interpolation_function(
        [2017.0,2023.0],
        [Parameter("initial_notif_rate"), Parameter("latest_notif_rate")]
    )

    tracked_prop_reported_case = model.request_track_modelled_value("notif_ratio", prop_reported_case)

    notifs = model.request_function_output("notification", (DerivedOutput("treatment_commencement") * tracked_prop_reported_case) + DerivedOutput("acf_notification"))
    #notifs = model.request_output_for_flow("notification", "treatment_commencement")

    model.request_function_output("notification_log", np.log(notifs))

    # notification per incidence:
    model.request_function_output(
        "notification_per_incidence", notifs / DerivedOutput("incidence_raw") * 100.0
    )

    # adults (age >15) smear positive
    
    # Request proportion of each compartment in the total population
    for compartment in compartments:
        model.request_output_for_compartments(f"number_{compartment}", compartment)
        model.request_function_output(
            f"prop_{compartment}",
            DerivedOutput(f"number_{compartment}") / tot_pop,
        )

    # Request total population by age stratum
    for age_stratum in age_strata:
        model.request_output_for_compartments(
            f"total_populationXage_{age_stratum}",
            compartments,
            strata={"age": str(age_stratum)},
        )
        
    # request adults population
    adults_pop = [
        f"total_populationXage_{adults_stratum}" for adults_stratum in age_strata[2:]
    ]

    model.request_aggregate_output("adults_pop", adults_pop)

    for organ_stratum in organ_strata:
        model.request_output_for_compartments(
            f"total_infectiousXorgan_{organ_stratum}",
            infectious_compartments,
            strata={"organ": str(organ_stratum)},
        )
        for age_stratum in age_strata:
            model.request_output_for_compartments(
                f"total_infectiousXorgan_{organ_stratum}Xage_{age_stratum}",
                infectious_compartments,
                strata={"organ": str(organ_stratum), "age": str(age_stratum)},
            )
        model.request_function_output(
            f"prop_{organ_stratum}",
            DerivedOutput(f"total_infectiousXorgan_{organ_stratum}")
            / DerivedOutput("infectious_population_size"),
        )

    # Request adults smear_positive
    adults_smear_positive = [
        f"total_infectiousXorgan_smear_positiveXage_{adults_stratum}" for adults_stratum in age_strata[2:]
    ]

    model.request_aggregate_output("adults_smear_positive", adults_smear_positive)
    model.request_function_output(
        "prevalence_smear_positive",
        1e5 * DerivedOutput("adults_smear_positive") / DerivedOutput("adults_pop"),
    )

    # request adults pulmonary (smear postive + smear neagative)
    adults_pulmonary = [
        f"total_infectiousXorgan_{smear_status}Xage_{adults_stratum}"
        for adults_stratum in age_strata[2:]
        for smear_status in organ_strata[:2]
    ]
    model.request_aggregate_output("adults_pulmonary", adults_pulmonary)
    adults_prevalence_pulmonary = model.request_function_output("adults_prevalence_pulmonary", 1e5 * DerivedOutput("adults_pulmonary") / DerivedOutput("adults_pop"),)
    model.request_function_output("adults_prevalence_pulmonary_log", np.log(adults_prevalence_pulmonary))



