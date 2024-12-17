import os
import yaml
from types import SimpleNamespace

def yaml_file_loader(file_path: str) -> SimpleNamespace:
    assert os.path.exists(file_path), f"The file should exist, cannot read {file_path}"
    assert os.path.isfile(file_path), f"Should specify a valid file, got {file_path}"
    assert file_path.lower().endswith(".yaml"), f"Shouldbe a yaml file, got {file_path}"
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return SimpleNamespace(**data)



    