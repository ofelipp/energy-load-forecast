# ! ./venv/bin/python3.10

"""
Script destinated to prepare the dataset before input in the NN model.
"""

from logging import debug, info
import numpy as np
import pandas as pd
from pathlib import Path
import re


TRIMESTER_NAME_DATE = {
    "1": "01-01",
    "2": "04-01",
    "3": "07-01",
    "4": "10-01",
}

MONTH_NAME_NUMBER = {
    "janeiro": "01",
    "fevereiro": "02",
    "março": "03",
    "abril": "04",
    "maio": "05",
    "junho": "06",
    "julho": "07",
    "agosto": "08",
    "setembro": "09",
    "outubro": "10",
    "novembro": "11",
    "dezembro": "12",
}


class DataprepPipeline:

    """
    Class created to structure the pipeline of dataprep for model training
    """


def consolidate_files(
    files_path: str, name_condition: str = None
) -> pd.DataFrame:
    """
    Consolidate every predictions realized by ONS Dessem
    to create a baseline to compare with the other models.
    """

    info("Consolidating files from a csv files list")
    info(f"Directory: {Path(files_path).parent.absolute()}")

    _concat_df = pd.DataFrame()

    for file in files_path.iterdir():
        if name_condition is not None:
            if name_condition not in str(file):
                continue

        debug(f"file - {Path(file).name}")
        _tmp_df = pd.read_csv(file, sep=";", decimal=",")

        # file_date = re.search(r"(?<=\w_)\d.*(?=\.csv)", str(file)).group()
        file_date = re.search(r"\d{4}\-\d{2}\-\d{2}", str(file)).group()

        _tmp_df["file_date"] = pd.to_datetime(file_date)
        _tmp_df["file_name"] = Path(file).name

        _concat_df = pd.concat([_concat_df, _tmp_df])

    return _concat_df.reset_index(drop=True)


def create_df_resol_horaria(df: pd.DataFrame, _num_cols: list) -> pd.DataFrame:
    """Dataset creation with horary resolution"""

    print("Shape original:", df.shape, "\n")

    print(" === Periodo === ")
    _min = df.Datetime.min().strftime("%Y-%m-%d")
    _max = df.Datetime.max().strftime("%Y-%m-%d")
    print(f"{_min} - {_max}\n")

    # Criação do Dataset em resolução horária
    df_horario = pd.DataFrame(
        {"Datetime": pd.date_range(start=_min, end=_max, freq="H")}
    )

    # Preenchimento dos valores existentes
    df_horario = pd.merge(df_horario, df, on="Datetime", how="left")

    # Interpolação nas colunas numéricas
    df_horario[_num_cols] = df_horario[_num_cols].interpolate(method="linear")

    _cat_cols = [
        col for col in df_horario.columns if col not in _num_cols + ["Datetime"]
    ]
    df_horario[_cat_cols] = df_horario[_cat_cols].ffill()

    print("Shape final:", df_horario.shape, "\n")

    print(df_horario.head(), "\n")

    return df_horario


def ibge_datetime(serie: pd.Series, input_format: "str") -> pd.Series:
    """
    Function used to create datetime column in dataframes provided
    by IBGE (Brazilian Geograph and Statistics Institute).

    Args:
        input_format: allowed ['trimester', 'month']
    """

    # Copying
    date_serie = serie.copy()

    # Spliting
    date_df = date_serie.str.split(expand=True)

    # Renaming
    if input_format == "trimester":
        _dict_names = {0: "Trimester", 1: "Name", 2: "Year"}
    elif input_format == "month":
        _dict_names = {0: "Month", 1: "Year"}
    else:
        # TODO: change this to a raise exception
        print("Error")

    date_df.rename(columns=_dict_names, inplace=True)

    # Datetime creation
    if input_format == "trimester":
        return pd.to_datetime(
            date_df["Year"]
            + "-"
            + date_df["Trimester"].str.slice(0, 1).map(TRIMESTER_NAME_DATE)
        )
    elif input_format == "month":
        return pd.to_datetime(
            date_df["Year"] + "-" + date_df["Month"].map(MONTH_NAME_NUMBER)
        )
    else:
        # TODO: change this to a raise exception
        print("Error")


def interpolate_final(serie: pd.Series) -> pd.Series:
    """Fill the last values from Series not covered by intepolation method"""

    df = pd.DataFrame({"to_fill": serie})

    # Nulls
    df["null"] = np.where(df["to_fill"].isnull(), 1, 0).cumsum()

    ar = np.array(df["to_fill"].dropna().iloc[-3:-1])
    linear_step = ar[-1] - ar[-2]

    return df["to_fill"].ffill() + df["null"] * linear_step


def truncate_date_range(date_range: pd.date_range) -> pd.date_range:
    ideal_date_range = pd.date_range(
        start=date_range.min().normalize(),
        end=date_range.max().normalize(),
        freq="H",
        inclusive="left",
    )

    if not date_range.equals(ideal_date_range):
        diff_begin = date_range[0] != ideal_date_range[0]
        if diff_begin:
            _rm_incomplete_day = (
                date_range >= date_range.normalize().shift(1, "D")[0]
            )
            print(f"Removing {(~_rm_incomplete_day).sum()} from beggining")

            date_range = date_range[_rm_incomplete_day]

        diff_end = date_range[-1] != ideal_date_range[-1]
        if diff_end:
            _rm_incomplete_day = date_range < date_range.normalize()[-1]
            print(f"Removing {(~_rm_incomplete_day).sum()} from ending")

            date_range = date_range[_rm_incomplete_day]

        print(f"Final Date Range: {date_range[0]} - {date_range[-1]}")
        return date_range
