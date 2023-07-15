# ! ./venv/bin/python3.8

"""
Script destinated to create import and output functions to read jsons, csv
and excel files and export the final product into
"""

import config
import json
from logging import debug, info
import os
from pathlib import Path
import pickle
from zipfile import ZipFile

PATHS = config.config_path()


def extract_zip_content_files(
    zip_files_list: list, destiny_path: str = PATHS["SOT"]
):
    info("Extracting files from a zip files list")
    info(f"Directory: {Path(zip_files_list[0]).parent.absolute()}")

    for zip_file in zip_files_list:
        debug(f"File - {Path(zip_file).name}")

        if "zip" in zip_file:
            with ZipFile(zip_file, "r") as zp:
                zp.extractall(destiny_path)
                debug("files extracted")


def json_to_dict(path: str) -> dict:
    """Function used to import json archive into python dictionary"""

    with open(path, "r+", "utf8") as json_file:
        return json.load(json_file)


def list_dir_files(path: str) -> list:
    """Function which returns a list of filepaths in a directory"""

    all_files = []

    for root, dirs, files in os.walk(path):
        # Reading only subdirectories
        if len(dirs) == 0:
            subdir_files = [f"{root}/{file}" for file in files]
            all_files += subdir_files

    return all_files


def load_artifact(filepath):
    assert (
        os.path.isfile(filepath) is True
    ), f"Artfact not founded in {filepath}"

    with open(filepath, "rb") as input_model:
        return pickle.load(input_model)


def save_artifact(model, model_name: str, path: str):
    _filepath = path / f"{model_name}.pickle"

    with open(_filepath, "wb") as output_model:
        pickle.dump(model, output_model)

    assert (
        os.path.isfile(_filepath) is True
    ), f"Impossible to save {model_name} artfact not founded in {_filepath}"
