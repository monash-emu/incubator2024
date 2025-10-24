from pathlib import Path
import numpy as np
from enum import Enum

COMPARTMENTS = [
    "susceptible", 
    "early_latent", 
    "late_latent", 
    "infectious", 
    "on_treatment",
    "recovered"
]

LATENT_COMPARTMENTS = ["early_latent", "late_latent"]
INFECTIOUS_COMPARTMENTS = ["infectious", "on_treatment"]

MODEL_TIMES = [1800., 2035.]
CUMULATIVE_OUTPUT_START_TIME = 2026.0

AGEGROUP_REQUEST = [[0, 4], [5, 14], [15, 34], [35, 49], [50, 100]]
AGE_STRATA = [group[0] for group in AGEGROUP_REQUEST]

ORGAN_STRATA = [
    "smear_positive",
    "smear_negative",
    "extrapulmonary",
]

INDICATOR_NAMES = {
    "comp_size_susceptible": "Susceptible",
    "comp_size_early_latent": "Early latent",
    "comp_size_late_latent": "Late latent",
    "comp_size_infectious": "Infectious",
    "comp_size_recovered": "Recovered",
    "total_population": "Total population",
    "notification": "No. of case notification",
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

PARAM_DESC = {
    "contact_rate": "Contact rate",
    "rr_infection_latent": "Relative risk of reinfection while latently infected (ref. infection-naive)",
    "rr_infection_recovered": "Relative risk of reinfection after recovery (ref. infection-naive)",
    "progression_multiplier": "TB progression multiplier",
    "screening_scaleup_shape": "Passive screening shape",
    "screening_inflection_time": "Passive screening inflection time",
    "time_to_screening_end_asymp": "Time from active TB to be screened",
    "base_diagnostic_capacity": "Preceding diagnostic algorithm capacity",
    "genexpert_sensitivity": "GeneXpert sensitivity",
    "detection_reduction": "Detection reduction",
    "incidence_props_pulmonary": "Proportion of pulmonary TB cases among all TB cases",
    "incidence_props_smear_positive_among_pulmonary": "Proportion of smear-positive TB among all pulmonary TB cases",
    "smear_positive_death_rate": "Death rate for smear-positive TB",
    "smear_negative_death_rate": "Death rate for smear-negative and extrapulmonary TB",
    "smear_positive_self_recovery": "Self-recovery rate for smear-positive TB",
    "smear_negative_self_recovery": "Self-recovery rate for smear-negative and extrapulmonary TB",
    "initial_notif_rate": "2017 CDR",
    "latest_notif_rate": "2023 CDR",
}
QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

SCENARIO_NAMES = {
    'base_scenario': 'Baseline scenario',
    'increase_xpert_util_target_by_0_95': 'Xpert: 95%',
    'increase_xpert_util_target_by_0_8': 'Xpert: 80%',
    'increase_xpert_util_target_by_0_7': 'Xpert: 70%',
    'increase_tsr_target_by_0_9': 'TSR: 90%',
    'increase_tsr_target_by_0_95': 'TSR: 95%',
    'acf': 'ACF',
    'acf_xpert': 'ACF + Xpert',
    'acf_xpert_tsr': 'ACF + Xpert + TSR',
    'acf_xpert_tsr_detect': 'ACF + Xpert + TSR + Passive detection',
}

# Case Detection Scenarios
for detection in range(2, 11):
    key = f"increase_case_detection_by_{detection}_0"
    label = f"{detection}× Detection"
    SCENARIO_NAMES[key] = label


# ACF Scenarios
for acf in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80]:
    key = f"acf_{acf}"
    label = f"{acf}% population"
    SCENARIO_NAMES[key] = label

for freq in [2, 3, 4]:
    key = f"implement_acf_every_{freq}_years"
    label = f"ACF imp. every {freq} years"
    SCENARIO_NAMES[key] = label

# Combined Scenarios: Xpert + ACF
xpert_targets = [0.9, 0.8, 0.7]
acf_scenarios = [5, 10, 20, 30, 40, 50, 60, 70, 80]
for x in xpert_targets:
    for acf in acf_scenarios:
        key = f"increase_xpert_util_target_by_{str(x).replace('.', '_')}_and_implement_acf_{str(acf)}"
        label = f"Xpert Util: {int(x*100)}%; ACF: {int(acf)}%"
        SCENARIO_NAMES[key] = label

SCENARIO_DASH_STYLES = {
    "_case_detection_by_2_0": "solid",
    "_case_detection_by_5_0": "dash",
    "_case_detection_by_10_0": "dot",
    "_and_implement_acf_5": "solid",
    "_and_implement_acf_10": "dot",
    "_and_implement_acf_20": "dash",
    "_and_implement_acf_30": "dashdot",
    "_and_implement_acf_40": "longdash",
    "_and_implement_acf_50": "longdashdot",
    "_and_implement_acf_60": "dash",
    "_and_implement_acf_70": "dot",
    "_and_implement_acf_80": "solid",
}

SCENARIO_MARKER_STYLES = {
    "_case_detection_by_2_0": "cross",
    "_case_detection_by_5_0": "square",
    "_case_detection_by_10_0": "diamond",
    "_and_implement_acf_5": "circle",
    "_and_implement_acf_10": "square",
    "_and_implement_acf_20": "diamond",
    "_and_implement_acf_30": "cross",
    "_and_implement_acf_40": "x",
    "_and_implement_acf_50": "triangle-up",
    "_and_implement_acf_60": "triangle-down",
    "_and_implement_acf_70": "triangle-left",
    "_and_implement_acf_80": "triangle-right",
}

SCENARIO_COLOURS = {
    "xpert_util_target_by_0_9": 'rgb(68, 1, 84)', #violet
    "xpert_util_target_by_0_8": 'rgb(255, 127, 0)', #orange
    "xpert_util_target_by_0_7": 'rgb(31, 120, 180)', #blue
}

COVID_CONFIGS = {
    'no_covid': {"detection_reduction": False},
    'case_detection_reduction_only': {"detection_reduction": True},
}

PARAMETER_SECTIONS = {
    "Demographics": [
        "start_population_size",
        "Birth rate",
    ],
    "TB transmission and disease natural history": [
        "contact_rate",
        "age_infectiousness_switch",
        "rr_infection_latent",
        "rr_infection_recovered",
        "smear_positive_infect_multiplier",
        "smear_negative_infect_multiplier",
        "extrapulmonary_infect_multiplier",
        "on_treatment_infect_multiplier",
        "progression_multiplier",
        "age_latency",
        "incidence_props_pulmonary",
        "incidence_props_smear_positive_among_pulmonary",
        "smear_positive_death_rate",
        "smear_negative_death_rate",
        "smear_positive_self_recovery",
        "smear_negative_self_recovery",
    ],
    "TB control (detection and treatment)": [
        "screening_scaleup_shape",
        "screening_inflection_time",
        "time_to_screening_end_asymp",
        "base_diagnostic_capacity",
        "genexpert_sensitivity",
        "passive_screening_sensitivity_extrapulmonary",
        "passive_screening_sensitivity_smear_negative",
        "passive_screening_sensitivity_smear_positive",
        "acf_sensitivity",
        "initial_notif_rate",
        "latest_notif_rate",
        "detection_reduction",
        "treatment_duration",
        "time_variant_tsr",
        "prop_death_among_negative_tx_outcome",
    ]
}

PARAMETER_GROUPS = {
    "infect_multiplier": {
        "keys": [
            "smear_positive_infect_multiplier",
            "smear_negative_infect_multiplier", 
            "extrapulmonary_infect_multiplier"
        ],
        "shared_description": "Relative infectiousness of smear-positive/smear-negative/extrapulmonary TB"
    },
    "passive_screening_sensitivity": {
        "keys": [
            "passive_screening_sensitivity_smear_positive",
            "passive_screening_sensitivity_smear_negative",
            "passive_screening_sensitivity_extrapulmonary"
        ],
        "shared_description": "Passive TB screening sensitivity (smear-positive/smear-negative/extrapulmonary TB)"
    },
    "tb_death_rate": {
        "keys": [
            "smear_positive_death_rate",
            "smear_negative_death_rate"
        ],
        "shared_description": "Rate of TB-specific mortality (smear-positive/other forms of TB)"
    },
    "tb_recovery_rate": {
        "keys": [
            "smear_positive_self_recovery",
            "smear_negative_self_recovery"
        ],
        "shared_description": "Rate of self-recovery (smear-positive/other forms of TB)"
    }
}

class ImplementCDR(Enum):
    ON_DETECTION = "on_detection_func"  
    ON_NOTIFICATION = "on_notif"
    NONE = None

_PROJECT_PATH = None  # module-level global

def set_project_base_path(path: Path):
    global _PROJECT_PATH
    _PROJECT_PATH = Path(path).resolve()
    return get_project_paths()

def get_project_paths():
    if _PROJECT_PATH is None:
        raise Exception("set_project_base_path must be called before attempting to use project paths")
    return {
        "PROJECT_PATH": _PROJECT_PATH,
        "DATA_PATH": _PROJECT_PATH / "data",
        "LOCAL_DATA_PATH": _PROJECT_PATH / "data" / "local_data",
        "IMAGE_PATH": _PROJECT_PATH / "imgs",
        "OUT_PATH": _PROJECT_PATH / "runs",
        "OUTPUTS": _PROJECT_PATH / "outputs"
    }

# Initialize paths (can be removed from here if set externally)
project_paths = set_project_base_path("../tb_incubator")
project_path = project_paths["PROJECT_PATH"]
data_path = project_paths["DATA_PATH"]
image_path = project_paths["IMAGE_PATH"]
out_path = project_paths["OUT_PATH"]

BURN_IN = 2500
