from pathlib import Path
import numpy as np

# ========================================
# Compartments & Model Structure
# ========================================
COMPARTMENTS = [
    "susceptible", 
    "early_latent", 
    "late_latent", 
    "infectious", 
    "recovered"
]

LATENT_COMPARTMENTS = ["early_latent", "late_latent"]
INFECTIOUS_COMPARTMENTS = ["infectious"]

# ========================================
# Time and Stratifications
# ========================================
MODEL_TIMES = [1800, 2035]

AGEGROUP_REQUEST = [[0, 4], [5, 14], [15, 34], [35, 49], [50, 100]]
AGE_STRATA = [group[0] for group in AGEGROUP_REQUEST]

ORGAN_STRATA = [
    "smear_positive",
    "smear_negative",
    "extrapulmonary",
]

# ========================================
# Indicator Names (for Plotting / Output)
# ========================================
INDICATOR_NAMES = {
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

# ========================================
# Scenario Labels
# ========================================
SCENARIO_NAMES = {
    'base_scenario': 'Baseline scenario',
    'increase_xpert_util_target_by_0_9': 'Xpert Utilisation: 90%',
    'increase_xpert_util_target_by_0_8': 'Xpert Utilisation: 80%',
    'increase_xpert_util_target_by_0_7': 'Xpert Utilisation: 70%',
}

# Case Detection Scenarios
for detection in range(2, 11):
    key = f"increase_case_detection_by_{detection}_0"
    label = f"{detection}× Detection"
    SCENARIO_NAMES[key] = label

# Combined Scenarios: Xpert + Detection
xpert_targets = [0.9, 0.8, 0.7]
detection_multipliers = [2.0, 5.0, 10.0]
for x in xpert_targets:
    for d in detection_multipliers:
        key = f"increase_xpert_util_target_by_{str(x).replace('.', '_')}_and_case_detection_by_{str(d).replace('.', '_')}"
        label = f"Xpert Util: {int(x*100)}%; Detection: {int(d)}×"
        SCENARIO_NAMES[key] = label

# ACF Scenarios
for acf in [5, 10, 20, 30, 40, 50, 60, 70, 80]:
    key = f"acf_{acf}"
    label = f"ACF in {acf}% population"
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

# ========================================
# Scenario Visual Styles
# ========================================
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
    "xpert_util_target_by_0_9": 'rgb(68, 1, 84)',
    "xpert_util_target_by_0_8": 'rgb(49, 104, 142)',
    "xpert_util_target_by_0_7": 'rgb(53, 183, 121)', 
}

# ========================================
# COVID Configs
# ========================================
COVID_CONFIGS = {
    'no_covid': {"detection_reduction": False},
    'case_detection_reduction_only': {"detection_reduction": True},
}

# ========================================
# Project Paths
# ========================================
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

# ========================================
# Sampler
# ========================================
BURN_IN = 12500
