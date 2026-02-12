import yaml
from types import SimpleNamespace
from typing import Union
from pathlib import Path


def load_args_from_yaml(yaml_file_path: Union[str, Path]) -> SimpleNamespace:
    """ Little helper function to load the arguments from a yaml file """
    with open(yaml_file_path, 'r') as file:
        args = yaml.safe_load(file)
    return SimpleNamespace(**args)