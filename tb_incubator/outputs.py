from summer2 import CompartmentalModel
from typing import List
from summer2.parameters import Parameter, DerivedOutput
import numpy as np


def request_model_outputs(
    model: CompartmentalModel,
    compartments: List[str],
    latent_compartments: List[str],
    infectious_compartments: List[str],
    age_strata: List[int],
):
    # compartment size
    for c in compartments:
        model.request_output_for_compartments(f"comp_size_{c}", c)

    # calculate and request percentage of latent population
    model.request_output_for_compartments("total_population", compartments, save_results=False)
    model.request_output_for_compartments(
        "latent_population_size", latent_compartments, save_results=False
    )
    percentage_latent = DerivedOutput("latent_population_size") / DerivedOutput("total_population")
    model.request_function_output("percentage_latent", percentage_latent)

    # death
    model.request_output_for_flow("mortality_infectious", "TB death")
    model.request_function_output(
        "mortality",
        1e5 * DerivedOutput("mortality_infectious") / DerivedOutput("total_population"),
    )

    # total prevalence
    model.request_output_for_compartments(
        "infectious_population_size", infectious_compartments, save_results=False
    )
    prevalence_infectious = (
        DerivedOutput("infectious_population_size") / DerivedOutput("total_population") * 1e5
    )
    model.request_function_output("prevalence", prevalence_infectious)

    # incidence
    for timing in ["early", "late"]:
        model.request_output_for_flow(
            f"{timing}_activation", f"{timing} activation", save_results=False
        )

    model.request_aggregate_output(
        "incidence_raw", ["early_activation", "late_activation"], save_results=True
    )

    incidence = 1e5 * DerivedOutput("incidence_raw") / DerivedOutput("total_population")
    model.request_function_output("incidence", incidence)

    # notification
    model.request_function_output(
        "notification", DerivedOutput("incidence_raw") * Parameter("case_detection_rate")
    )

    # case notification rate:
    model.request_function_output(
        "case_notification_rate",
        DerivedOutput("notification") / DerivedOutput("incidence_raw") * 100,
    )

    # request total population by age stratum
    for age_stratum in age_strata:
        model.request_output_for_compartments(
            f"total_populationXage_{age_stratum}",
            compartments,
            strata={"age": str(age_stratum)},
        )
