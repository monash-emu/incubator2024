from pathlib import Path


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
    }
