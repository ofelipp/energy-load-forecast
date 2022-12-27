# ! ./venv/bin/python3.8

"""
"""

import pandas as pd
from sklearn.model_selection import train_test_split

PRC_DATA = "data/prc/"

electric_demand = pd.read_parquet(f"{PRC_DATA}electricity_demand.parquet")


def split_train_cv_test(
    df: pd.DataFrame, target: str, features: list = None,
    perc: dict = {"train": 0.6, "cv": 0.2, "test": 0.2}
) -> tuple:

    """
    Function used to split dataset into train, cross-validation and test \n
    datasets to training and evaluate the model.

    Can be used in all types of ML models (neural networking, linear, and so on)
    """

    # Periods percentage needs sum 100%
    assert perc['train'] + perc['cv'] + perc['test'] == 1, \
        "Periods percentage needs sum 100%"

    # Target and Features
    Y = df[target].copy()
    X = df[[col for col in df.columns if target != col]].copy()

    if features is not None:
        X = df[features].copy()

    # Train - Cross Validation
    X_train, X_cv = train_test_split(
        X, train_size=perc["train"],
        test_size=perc["cv"] + perc["test"],
        random_state=42, shuffle=False
    )

    Y_train, Y_cv = train_test_split(
        Y, train_size=perc["train"],
        test_size=perc["cv"] + perc["test"],
        random_state=42, shuffle=False
    )

    # Cross Validation - Test
    X_cv, X_test = train_test_split(
        X_cv, train_size=perc["cv"] / (perc["cv"] + perc["test"]),
        test_size=perc["test"] / (perc["cv"] + perc["test"]),
        random_state=42, shuffle=False
    )

    Y_cv, Y_test = train_test_split(
        Y_cv, train_size=perc["cv"] / (perc["cv"] + perc["test"]),
        test_size=perc["test"] / (perc["cv"] + perc["test"]),
        random_state=42, shuffle=False
    )

    return X_train, X_cv, X_test, Y_train, Y_cv, Y_test


def time_serie_window():

    """ """
    # Requirements:

    #   input_width = how many steps you will use to train
    #   predict_width = how many steps you want to predict
    #   predict_begin = when the predictions will begin


    return ...






def calcula_erro_treino():
    ...
