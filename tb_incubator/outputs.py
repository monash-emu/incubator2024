from summer2 import CompartmentalModel
from typing import List, Dict
from summer2.parameters import DerivedOutput
from tb_incubator.constants import COMPARTMENTS, INFECTIOUS_COMPARTMENTS, ORGAN_STRATA, AGE_STRATA, LATENT_COMPARTMENTS, ImplementCDR
import numpy as np
from summer2.parameters import Function, Parameter
from summer2.functions.time import get_linear_interpolation_function

def request_model_outputs(
    model: CompartmentalModel, 
    detection_func: Function,
    acf_screening_rate: Dict[float, float] = None,
    apply_cdr: ImplementCDR = ImplementCDR.NONE,
    cumulative_output_start_time: float = 2026.0
):
    # compartment sizes
    for c in COMPARTMENTS:
        model.request_output_for_compartments(f"comp_size_{c}", c)
    total_population = model.request_output_for_compartments("total_population", COMPARTMENTS)
    model.request_function_output("population_log", np.log(total_population))

    # latent percentage
    latent_pop = model.request_output_for_compartments(
        "latent_pop", LATENT_COMPARTMENTS
    )
    model.request_function_output("percentage_latent", latent_pop / total_population * 100.0)

    # death
    model.request_output_for_flow("mortality_infectious_raw", "infect_death")
    model.request_output_for_flow("mortality_on_treatment_raw", "treatment_death")
    mortality_raw = model.request_aggregate_output(
        "mortality_raw", ["mortality_infectious_raw", "mortality_on_treatment_raw"]
    )
    model.request_cumulative_output(
        "cumulative_deaths",
        "mortality_raw",
        start_time=cumulative_output_start_time,
    )
    model.request_function_output("mortality", 1e5 * mortality_raw / total_population)

    # prevalence
    # Calculate and request prevalence of pulmonary
    for organ_stratum in ORGAN_STRATA:
        model.request_output_for_compartments(
            f"infectious_sizeXorgan_{organ_stratum}",
            INFECTIOUS_COMPARTMENTS,
            strata={"organ": organ_stratum},
            save_results=False,
        )

    pulmonary_outputs = [
        f"infectious_sizeXorgan_{organ_stratum}"
        for organ_stratum in ["smear_positive", "smear_negative"]
    ]

    pulmonary_pop_size = model.request_aggregate_output("pulmonary_pop_size", pulmonary_outputs)
    model.request_function_output("prevalence_pulmonary", 1e5 * pulmonary_pop_size / total_population)

    # total prevalence
    infectious_population_size = model.request_output_for_compartments(
        "infectious_population_size", INFECTIOUS_COMPARTMENTS
    )
    prevalence = model.request_function_output("prevalence", 1e5 * infectious_population_size / total_population)
    model.request_function_output("prevalence_log", np.log(prevalence))

    # incidence
    incidence_early_raw = model.request_output_for_flow("incidence_early_raw", "early_activation")
    model.request_output_for_flow("incidence_late_raw", "late_activation")

    incidence_raw = model.request_aggregate_output("incidence_raw", ["incidence_early_raw", "incidence_late_raw"])
    incidence_early_prop = model.request_function_output(
        "incidence_early_prop", incidence_early_raw / incidence_raw * 100)
    
    model.request_function_output("incidence_late_prop", 100 - incidence_early_prop)


    model.request_cumulative_output(
        "cumulative_diseased",
        "incidence_raw",
        start_time=cumulative_output_start_time,
    )

    model.request_function_output("incidence", 1e5 * incidence_raw / total_population)

    # notification
    model.request_output_for_flow("passive_notification_raw", "detection")

    if apply_cdr == ImplementCDR.ON_NOTIFICATION:
        national_cdr = get_linear_interpolation_function(
            [2017.0, 2023.0],
            [Parameter("initial_notif_rate"), Parameter("latest_notif_rate")]
        )
        tracked_prop_reported_case = model.request_track_modelled_value("notif_ratio", national_cdr)
    else:
        tracked_prop_reported_case = Function(lambda: 1.0)
    
    passive_notif_raw = DerivedOutput("passive_notification_raw") * tracked_prop_reported_case

    if acf_screening_rate is not None:
        model.request_output_for_flow("active_notification_raw", "acf_detection")
        notifs = model.request_function_output("notification", passive_notif_raw + DerivedOutput("active_notification_raw"))
    else:
        notifs = model.request_function_output("notification", passive_notif_raw)
        #notifs = model.request_output_for_flow("notification", "detection")
    
    model.request_function_output("notif_100k", notifs / total_population * 1e5)

    model.request_function_output("notification_log", np.log(notifs))

    # notification per incidence:
    model.request_function_output(
        "notification_per_incidence", notifs / DerivedOutput("incidence_raw") * 100.0
    )

    extra_notif = model.request_output_for_flow(
        name="extra_notification",
        flow_name="detection",
        source_strata={"organ": "extrapulmonary"},
    )
    pul_notif = model.request_function_output("pulmonary_notif", notifs - extra_notif)
    model.request_function_output("pulmonary_prop",  pul_notif / notifs * 100)


    # adults (age >15) smear positive
    
    # Request proportion of each compartment in the total population
    for compartment in COMPARTMENTS:
        model.request_output_for_compartments(f"number_{compartment}", compartment)
        model.request_function_output(
            f"prop_{compartment}",
            DerivedOutput(f"number_{compartment}") / total_population,
        )

    # Request total population by age stratum
    for age_stratum in AGE_STRATA:
        model.request_output_for_compartments(
            f"total_populationXage_{age_stratum}",
            COMPARTMENTS,
            strata={"age": str(age_stratum)},
        )
        model.request_output_for_flow(
                f"early_activationXage_{age_stratum}_raw",
                "early_activation",
                source_strata={"age": str(age_stratum)},
            )
        model.request_output_for_flow(
                f"late_activationXage_{age_stratum}_raw",
                "late_activation",
                source_strata={"age": str(age_stratum)},
            )
        
    # request adults population
    adults_pop_list = [
        f"total_populationXage_{adults_stratum}" for adults_stratum in AGE_STRATA[2:]
    ]

    adults_pop = model.request_aggregate_output("adults_pop", adults_pop_list)
    children_pop = model.request_function_output("children_pop", total_population - adults_pop)

    for organ_stratum in ORGAN_STRATA:
        model.request_output_for_compartments(
            f"total_infectiousXorgan_{organ_stratum}",
            INFECTIOUS_COMPARTMENTS,
            strata={"organ": str(organ_stratum)},
        )
        for age_stratum in AGE_STRATA:
            model.request_output_for_compartments(
                f"total_infectiousXorgan_{organ_stratum}Xage_{age_stratum}",
                INFECTIOUS_COMPARTMENTS,
                strata={"organ": str(organ_stratum), "age": str(age_stratum)},
            )
        model.request_function_output(
            f"prop_{organ_stratum}",
            DerivedOutput(f"total_infectiousXorgan_{organ_stratum}")
            / DerivedOutput("infectious_population_size"),
        )

    # Request adults smear_positive
    adults_smear_positive = [
        f"total_infectiousXorgan_smear_positiveXage_{adults_stratum}" for adults_stratum in AGE_STRATA[2:]
    ]

    model.request_aggregate_output("adults_smear_positive", adults_smear_positive)
    model.request_function_output(
        "prevalence_smear_positive",
        1e5 * DerivedOutput("adults_smear_positive") / DerivedOutput("adults_pop"),
    )

    # request adults pulmonary (smear postive + smear neagative)
    adults_pulmonary = [
        f"total_infectiousXorgan_{smear_status}Xage_{adults_stratum}"
        for adults_stratum in AGE_STRATA[2:]
        for smear_status in ORGAN_STRATA[:2]
    ]
    model.request_aggregate_output("adults_pulmonary", adults_pulmonary)
    adults_prevalence_pulmonary = model.request_function_output("adults_prevalence_pulmonary", 1e5 * DerivedOutput("adults_pulmonary") / DerivedOutput("adults_pop"),)
    model.request_function_output("adults_prevalence_pulmonary_log", np.log(adults_prevalence_pulmonary))

    # Request output for children
    children_pulmonary = [
        f"total_infectiousXorgan_{smear_status}Xage_{children_stratum}"
        for children_stratum in AGE_STRATA[:2]
        for smear_status in ORGAN_STRATA[:2]
    ]
    model.request_aggregate_output("children_pulmonary", children_pulmonary)
    model.request_function_output(
        "children_prevalence_pulmonary",
        1e5 * DerivedOutput("children_pulmonary") / children_pop
    )
    children_early_activation = [
        f"early_activationXage_{children_stratum}_raw"
        for children_stratum in AGE_STRATA[:2]
    ]
    model.request_aggregate_output("children_early_activation", children_early_activation)

    children_late_activation = [
        f"late_activationXage_{children_stratum}_raw"
        for children_stratum in AGE_STRATA[:2]
    ]
    model.request_aggregate_output("children_late_activation", children_late_activation)
    children_incidence_raw = model.request_aggregate_output("children_incidence_raw", [DerivedOutput("children_early_activation"), DerivedOutput("children_late_activation")])
    model.request_function_output("children_incidence", 1e5 *  children_incidence_raw / children_pop)
    
    # Request detection rate
    detection_out = detection_func
    model.add_computed_value_func("detection_rate", detection_out)
    model.request_computed_value_output("detection_rate")



