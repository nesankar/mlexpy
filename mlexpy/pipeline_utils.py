import numpy as np
import pandas as pd
import logging
from typing import Any, Dict
from collections import namedtuple

from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


"""Some ML utility tools."""
MLSetup = namedtuple("MLSetup", ["obs", "labels"])
ExperimentSetup = namedtuple("ExperimentSetup", ["train_data", "test_data"])


def get_stratified_train_test_data(
    train_data: pd.DataFrame,
    label_data: pd.Series,
    random_state: np.random.RandomState,
    test_frac: float = 0.3,
    stratify: bool = True,
) -> ExperimentSetup:
    """Perform a structured training and testing split of a dataset.

    Parameters
    ----------
    train_data : pd.DataFrame
        The dataframe containing our feature data.

    label_data : pd.Series
        The labels or targets associated with the respective training dataframe.

    random_state : np.random.RandomState
        The random state to use to perform the splitting.

    test_frac : float
        The fraction of data that you want to use for testing. NOTE: if 1, then only test data will be returned (at it will be the entire dataset).

    stratify : bool
        A boolean to designate if the splitting should be stratified.

    Returns
    -------
    ExperimentSetup
    """
    # First, test to see if the test frac is 1. Essentially this is to initialize a dataset ONLY for testing.
    if test_frac == 1:
        return ExperimentSetup(
            MLSetup(pd.DataFrame(), pd.Series()), MLSetup(train_data, label_data)
        )

    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            train_data,
            label_data,
            test_size=test_frac,
            stratify=label_data,
            random_state=random_state,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            train_data,
            label_data,
            test_size=test_frac,
            random_state=random_state,
        )

    return ExperimentSetup(MLSetup(X_train, y_train), MLSetup(X_test, y_test))


def cv_report(results: Dict[str, Any], n_top: int = 5) -> None:

    """Print out a cross validation result following sklearn cross validation.

    Parameters
    ----------
    results : Dict[str, Any]
        The dictionary storing cross validation result models, and performance statistics.

    n_top : int
        The defined number of n top performing results to print out.

    Returns
    -------
    None
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print(f"Model with rank: {i}")
            print(
                f"""Mean validation score: {results["mean_test_score"][candidate]} (std: {results["std_test_score"][candidate]})"""
            )
            print(f"""Parameters: {results["params"][candidate]}\n""")
