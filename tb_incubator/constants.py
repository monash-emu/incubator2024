from pathlib import Path

compartments = ["susceptible", 
                "early_latent", 
                "late_latent", 
                "infectious", 
                "recovered",
                "missed"]

latent_compartments = ["early_latent", "late_latent"]
infectious_compartments = ["infectious", "missed"]

model_times = [1820, 2024]

agegroup_request = [[0, 4], [5, 14], [15, 34], [35, 49], [50, 100]]
age_strata = [i[0] for i in agegroup_request]

indicator_names = {
    "comp_size_susceptible": "Susceptible",
    "comp_size_early_latent": "Early latent",
    "comp_size_late_latent": "Late latent",
    "comp_size_infectious": "Infectious",
    "comp_size_recovered": "Recovered",
    "comp_size_missed": "Missed",
    "total_population": "Total population",
    "notification": "TB notifications",
    "incidence": "TB incidence (per 100,000)",
    "percentage_latent": "Percentage of latent TB infection (%)",
    "prevalence": "TB prevalence (per 100,000)",
    "case_detection_rate": "Case detection rate",
    "case_notification_rate": "Case notification rate",
    "mortality": "TB deaths (per 100,000)",
    "detection_rate": "Detection rate",
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
        "IMAGE_PATH": _PROJECT_PATH / "supplementary",
    }


project_paths = set_project_base_path("../tb_incubator")
project_path = project_paths["PROJECT_PATH"]
data_path = project_paths["DATA_PATH"]
image_path = project_paths["IMAGE_PATH"]
