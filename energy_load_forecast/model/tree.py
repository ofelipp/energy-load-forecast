# ! ./venv/bin/python3.10

"""
Script containing trees-based models construction, training and predict
functions.
"""

from data.io import save_artifact
from logging import debug, info
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Algorithms ===============


def decision_tree(params: dict, search: str = None, space: dict = None):
    """
    Creates a decision tree model:

        params : dictionary containing max_depth, min_samples_leaf and
        min_samples_split parameters to create a decision tree instance;

            params = {
                "max_depth": 30,
                "min_samples_leaf": 10,
                "min_samples_split": 15,
            }

        search: "grid" or "random" are avaiable options. "grid" performs
        GridSearchCV and "random" performs a RandomSearchCV. None is default.

        space: dictionary containg parameters to be seeked in search type.
        space supports only 'params' avaiable parameters.

            space = {
                "max_depth": [num for num in range(5, 60, 5)],
                "min_samples_leaf": [num for num in range(5, 60, 5)],
                "min_samples_split": [num for num in range(5, 60, 5)],
            }
    """

    info("===== Creating Decision Tree ===== ")
    debug(f"Params:\n{params}")

    if search == "grid":
        info("GridSearch")
        return GridSearchCV(
            DecisionTreeRegressor(), space, n_jobs=-1, verbose=4,
            error_score='raise'
        )

    elif search == "random":
        info("RandomSearch")
        return RandomizedSearchCV(
            DecisionTreeRegressor(), space, n_jobs=-1, verbose=4,
            error_score='raise'
        )

    else:
        info("Proceding with no search for hyperparameters")
        return DecisionTreeRegressor(
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            min_samples_split=params["min_samples_split"],
            random_state=22,
        )


def random_forest(params: dict, search: str = None, space: dict = None):
    """
    Creates a ensemble random forest model:

        params : dictionary containing max_depth, min_samples_leaf and
        min_samples_split parameters to create a random forest instance;

            params = {
                "max_depth": 30,
                "min_samples_leaf": 10,
                "min_samples_split": 15,
            }

        search: "grid" or "random" are avaiable options. "grid" performs
        GridSearchCV and "random" performs a RandomSearchCV. None is default.

        space: dictionary containg parameters to be seeked in search type.
        space supports only 'params' avaiable parameters.

            space = {
                "max_depth": [num for num in range(5, 60, 5)],
                "min_samples_leaf": [num for num in range(5, 60, 5)],
                "min_samples_split": [num for num in range(5, 60, 5)],
            }
    """

    info("===== Creating Random Forest ===== ")
    debug(f"Params:\n{params}")

    if search == "grid":
        info("GridSearch")
        return GridSearchCV(
            RandomForestRegressor(), space, n_jobs=-1, verbose=4
        )

    elif search == "random":
        info("RandomSearch")
        return RandomizedSearchCV(
            RandomForestRegressor(), space, n_jobs=-1, verbose=4
        )

    else:
        info("Proceding with no search for hyperparameters")
        return RandomForestRegressor(
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            min_samples_split=params["min_samples_split"],
            random_state=22,
        )


# Training Process ===============


def train_tree(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    save_path: str,
):
    """Train Tree Regressor model to perform predictions"""

    info("===== Training Tree Regressor ===== ")

    debug(f"Shape of features:\t{features.shape}")
    debug(f"Shape of labels:\t{labels.shape}")

    trained_tree = model.fit(features, labels.reshape(-1))

    info("Training complete...")

    if "best_params_" in dir(trained_tree):
        debug(f"Best Hyperparameters:\t{trained_tree.best_params_}")

        info("Saving Model pickle...")
        save_artifact(trained_tree.best_estimator_, model_name, save_path)

        return trained_tree.best_estimator_

    else:
        info("Saving Model pickle...")
        save_artifact(trained_tree, model_name, save_path)

        return trained_tree
