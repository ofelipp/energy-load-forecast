# ! ./venv/bin/python3.8

"""
Script containing class of models used to predict electric energy demand and
measure its accuracy compared to actual DESSEM model used by ONS.
"""

import pandas as pd

class Model():

    """
    Class to create the model used on prediction.

    Records:
        - Original Variables Dataset
        - Train, CV, Test Splits Datasets
        - Model Construction
        - Model Training
        - Model Prediction
        - Model Evalutating
        - Model save method

    """

    def __init__(self, type: str, description: str, variables: pd.DataFrame = None):

        self.type = type
        self.description = description
        self.variables = variables

    def predict_dessem(self):

        """ Dessem prediction """

        self.dessem = None




    def
