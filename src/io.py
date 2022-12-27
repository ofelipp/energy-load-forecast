# ! ./venv/bin/python3.8

"""
Script destinated to create import and output functions to read jsons, csv
and excel files and export the final product into
"""

import json
from os import walk


def json_to_dict(path: str) -> dict:

    """ Function used to import json archive into python dictionary """

    with open(path, "r+", "utf8") as json_file:
        return json.load(json_file)


def list_dir_files(path: str) -> list:

    """ Function which returns a list of filepaths in a directory """

    all_files = []

    for root, dirs, files in walk(path):

        # Reading only subdirectories
        if len(dirs) == 0:
            subdir_files = [root + "/" + file for file in files]
            all_files += subdir_files

    return all_files


def leitura_database():
    ...


def exporta_database():
    ...
