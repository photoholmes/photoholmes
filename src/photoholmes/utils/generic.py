from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(yaml_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        yaml_path (str | Path): The path to the YAML file.

    Returns:
        Dict[str, Any]: The contents of the YAML file.
    """
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return data
