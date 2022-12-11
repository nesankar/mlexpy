import pytest
from mlexpy import experiment, processor, pipeline_utils
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Callable
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import sys
from numpy.testing import assert_array_equal
from fixtures import simple_dataframe, simple_binary_dataframe, rs_10, rs_20
from test_ml_examples import (
    basic_processor,
)


@pytest.fixture
def class_labels():
    return pd.Series([0, 1, 1, 1, 0, 1], dtype=int)


@pytest.fixture
def class_predictions():
    return pd.Series([1, 1, 1, 1, 0, 0], dtype=int)


@pytest.fixture
def class_probabilities():
    return np.array([[0.05, 0.95], [0, 1], [0, 1], [0.5, 0.5], [1, 0], [1, 0]])


@pytest.fixture
def value_labels():
    return pd.Series([0, 1, 1, 1, 0, 1], dtype=int)


@pytest.fixture
def value_predictions():
    return pd.Series([1, 1, 1, 1, 0, 0], dtype=int)


def test_pct_correct_metric(class_labels, class_predictions):
    """Test that we can correctly add a test metric for the predictions."""

    def test_metric(true_labels: pd.Series, predictions: pd.Series) -> float:

        return sum(
            [
                true_labels.iloc[i] == predictions.iloc[i]
                for i in range(len(true_labels))
            ]
        ) / len(true_labels)

    exp_obj = experiment.ClassifierExperiment(
        train_setup=None,
        test_setup=None,
    )

    exp_obj.add_metric(test_metric, "pct_correct")

    results = exp_obj.evaluate_predictions(class_labels, class_predictions)

    # Assert that the prediction we make comes out to the value we desire
    assert results["pct_correct"] == 4 / 6


def test_weighted_prob_metric(
    class_labels,
    class_probabilities,
    class_predictions,
):
    """Test that we can correctly add a test metric for the predictions using class probabilities."""

    def test_metric(true_labels: pd.Series, predictions: np.ndarray) -> float:

        # We need to check that we are using class probabilities.
        try:
            predictions.shape[1]
        except IndexError:
            raise ValueError(
                "The predictions are not valued for each class, thus not class probabilities."
            )

        print(predictions)
        return sum(
            [predictions[i][true_labels.iloc[i]] for i in range(len(true_labels))]
        ) / len(true_labels)

    exp_obj = experiment.ClassifierExperiment(
        train_setup=None,
        test_setup=None,
    )

    exp_obj.add_metric(test_metric, "weighted_prob_correct")

    results = exp_obj.evaluate_predictions(
        class_labels, class_predictions, class_probabilities
    )

    # Assert that the prediction we make comes out to the value we desire
    assert results["weighted_prob_correct"] == sum([0.05, 1, 1, 0.5, 1, 0]) / len(
        class_labels
    )
