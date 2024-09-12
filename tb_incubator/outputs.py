from summer2 import CompartmentalModel
from typing import List
from summer2.parameters import Parameter, DerivedOutput, Function, Time
from tb_incubator.constants import latent_compartments, infectious_compartments, age_strata
from tb_incubator.utils import tanh_based_scaleup
from summer2.functions.time import get_linear_interpolation_function

def request_model_outputs(
    model: CompartmentalModel,
):
    compartments = [c.name for c in model._original_compartment_names]

    # compartment size
    for c in compartments:
        model.request_output_for_compartments(f"comp_size_{c}", c)

    # calculate and request percentage of latent population
    tot_pop = model.request_output_for_compartments("total_population", compartments, save_results=False)
    latent_pop = model.request_output_for_compartments("latent_pop_size", latent_compartments, save_results=False)
    model.request_function_output("percentage_latent", latent_pop / tot_pop * 100.0)

    # death
    deaths = model.request_output_for_flow("mortality_infectious", "TB_death")
    model.request_function_output("mortality", 1e5 * deaths / tot_pop)

    # total prevalence
    infect_pop = model.request_output_for_compartments("infectious_pop_size", infectious_compartments, save_results=False)
    model.request_function_output("prevalence", infect_pop / tot_pop * 1e5)

    # incidence
    for timing in ["early", "late"]:
        model.request_output_for_flow(
            f"{timing}_activation", f"{timing}_activation", save_results=False
        )

    inc_raw = model.request_aggregate_output(
        "incidence_raw", ["early_activation", "late_activation"], save_results=True
    )

    model.request_function_output("incidence", 1e5 * inc_raw / tot_pop)

    # notification
    notifs = model.request_function_output(
        "notification", DerivedOutput("incidence_raw") * Parameter("case_detection_rate")
    )

    # case notification rate:
    model.request_function_output("case_notification_rate", notifs / DerivedOutput("incidence_raw") * 100.0)

    # request total population by age stratum
    for age_stratum in age_strata:
        model.request_output_for_compartments(
            f"total_populationXage_{age_stratum}",
            compartments,
            strata={"age": str(age_stratum)},
        )

    # Detection rate
    detection_func = Function(tanh_based_scaleup,
                          [
                              Time,
                              Parameter("screening_scaleup_shape"),
                              Parameter("screening_inflection_time"),
                              0.0,
                              1.0 / Parameter("time_to_screening_end_asymp")
                          ])

    model.add_computed_value_func("detection_rate", detection_func)
    model.request_computed_value_output("detection_rate")
