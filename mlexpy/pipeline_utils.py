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
    stratify: bool = False,
) -> ExperimentSetup:
    """Perform some structured training and testing splitting. Default to stratified splitting."""

    # First, test to see if the test frac is 1. Essentially this is to initilize a dataset ONLY for testing.
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
    # Utility to print CV results from sklearn.

    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print(f"Model with rank: {i}")
            print(
                f"""Mean validation score: {results["mean_test_score"][candidate]} (std: {results["std_test_score"][candidate]})"""
            )
            print(f"""Parameters: {results["params"][candidate]}\n""")
