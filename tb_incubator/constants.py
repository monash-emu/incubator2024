from pathlib import Path

COMPARTMENTS = ["susceptible", 
                "early_latent", 
                "late_latent", 
                "infectious", 
                "recovered"]

LATENT_COMPARTMENTS = ["early_latent", "late_latent"]
INFECTIOUS_COMPARTMENTS = ["infectious"]


MODEL_TIMES = [1800, 2035]

AGEGROUP_REQUEST = [[0, 4], [5, 14], [15, 34], [35, 49], [50, 100]]
AGE_STRATA = [i[0] for i in AGEGROUP_REQUEST]

ORGAN_STRATA = [
    "smear_positive",
    "smear_negative",
    "extrapulmonary",
]

indicator_names = {
    "comp_size_susceptible": "Susceptible",
    "comp_size_early_latent": "Early latent",
    "comp_size_late_latent": "Late latent",
    "comp_size_infectious": "Infectious",
    "comp_size_recovered": "Recovered",
    "total_population": "Total population",
    "notification": "TB notifications",
    "incidence": "TB incidence (per 100,000/year)",
    "percentage_latent": "Percentage of latent TB infection (%)",
    "prevalence": "TB (all forms) prevalence (per 100,000/year)",
    "adults_prevalence_pulmonary": "Prevalence of pulmonary TB among adults (per 100,000/year)",
    "notif_ratio": "Proportion of reported cases",
    "case_detection_rate": "Case detection rate",
    "case_notification_rate": "Case notification rate",
    "mortality": "TB deaths (per 100,000/year)",
    "detection_rate": "Detection rate",
    "final_detection": "Rate of treatment commencement (/year)",
    "treatment_commencement": "No. of people commenced for treatment",
    "diagnostic_capacity": "Diagnostic algorithm capacity",
}

QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

scenario_names = {
	'base_scenario': 'Baseline scenario',
	'increase_xpert_util_target_by_0_9': 'GeneXpert Utilisation: 90%',
    'increase_xpert_util_target_by_0_8': 'GeneXpert Utilisation: 80%',
    'increase_xpert_util_target_by_0_7': 'GeneXpert Utilisation: 70%',
    'increase_case_detection_by_2_0': 'Detection rate doubled (2×)',
    'increase_case_detection_by_5_0': 'Detection rate five-fold (5×)',
    'increase_case_detection_by_10_0': 'Detection rate ten-fold (10×)',
	'increase_xpert_util_target_by_0_9_and_case_detection_by_2_0': 'Xpert Util: 90%; Detection: 2×',
    'increase_xpert_util_target_by_0_8_and_case_detection_by_2_0': 'Xpert Util: 80%; Detection: 2×',
    'increase_xpert_util_target_by_0_7_and_case_detection_by_2_0': 'Xpert Util: 70%; Detection: 2×',
    'increase_xpert_util_target_by_0_9_and_case_detection_by_5_0': 'Xpert Util 90%; Detection: 5×',
    'increase_xpert_util_target_by_0_8_and_case_detection_by_5_0': 'Xpert Util: 80%; Detection: 5×',
    'increase_xpert_util_target_by_0_7_and_case_detection_by_5_0': 'Xpert Util: 70%; Detection: 5×',
	'increase_xpert_util_target_by_0_9_and_case_detection_by_10_0': 'Xpert Util: 90%; Detection: 10×',
    'increase_xpert_util_target_by_0_8_and_case_detection_by_10_0': 'Xpert Util: 80%; Detection: 10×',
    'increase_xpert_util_target_by_0_7_and_case_detection_by_10_0': 'Xpert Util: 70%; Detection: 10×',
}

covid_configs = {
    'no_covid': {
        "detection_reduction": False
    },  # No COVID effects at all
    
    'case_detection_reduction_only': {
        "detection_reduction": True
    },  # With detection reduction
}


def set_project_base_path(path: Path):
    global _PROJECT_PATH
    _PROJECT_PATH = Path(path).resolve()

    return get_project_paths()


def get_project_paths():
    if _PROJECT_PATH is None:
        raise Exception(
            "set_project_base_path must be called before attempting to use project paths"
        )
    return {
        "PROJECT_PATH": _PROJECT_PATH,
        "DATA_PATH": _PROJECT_PATH / "data",
        "LOCAL_DATA_PATH": _PROJECT_PATH / "data" / "local_data",
        "IMAGE_PATH": _PROJECT_PATH / "imgs",
        "OUT_PATH": _PROJECT_PATH / "runs",
        "OUTPUTS": _PROJECT_PATH / "outputs"
    }


project_paths = set_project_base_path("../tb_incubator")
project_path = project_paths["PROJECT_PATH"]
data_path = project_paths["DATA_PATH"]
image_path = project_paths["IMAGE_PATH"]
out_path = project_paths["OUT_PATH"]

BURN_IN = 12500