import pytest
from fixtures import simple_dataframe, simple_series
from mlexpy import utils, pipeline_utils
import pandas as pd
import numpy as np


def test_filtering(simple_dataframe):
    """Make sure we are filtering as desired"""

    def filter_fn(value: int) -> bool:
        return value != 4

    filter_dict = {"obs2": [filter_fn]}
    filtered_data = utils.initial_filtering(simple_dataframe, filter_dict)
    assert filtered_data.equals(simple_dataframe[simple_dataframe["obs2"] != 4])

    def filter_fn_2(value: int) -> bool:
        return value == 4

    filter_dict_2 = {"obs2": [filter_fn_2]}
    filtered_data_2 = utils.initial_filtering(simple_dataframe, filter_dict_2)
    assert filtered_data_2.equals(simple_dataframe[simple_dataframe["obs2"] == 4])

    filter_dict_3 = {"obs2": [filter_fn_2, filter_fn]}
    filtered_data_3 = utils.initial_filtering(simple_dataframe, filter_dict_3)
    assert filtered_data_3.empty


def test_assertion_functions(simple_series, simple_dataframe):

    utils.series_assertion(simple_series)
    utils.df_assertion(simple_dataframe)

    with pytest.raises(
        AssertionError,
    ):
        utils.df_assertion(simple_series)
    with pytest.raises(
        AssertionError,
    ):
        utils.series_assertion(simple_dataframe)

    with pytest.raises(
        AssertionError,
    ):
        utils.series_assertion(simple_series.values)


def test_train_split(simple_dataframe):

    rs1 = np.random.RandomState(10)
    first_split = pipeline_utils.get_stratified_train_test_data(
        simple_dataframe[["obs1", "obs2"]], simple_dataframe["target"], random_state=rs1
    )

    rs2 = np.random.RandomState(10)
    second_split = pipeline_utils.get_stratified_train_test_data(
        simple_dataframe[["obs1", "obs2"]], simple_dataframe["target"], random_state=rs2
    )

    assert first_split.train_data.obs.equals(
        second_split.train_data.obs
    ), "When using identical data and random seeds, the train test split data are not identical."

    rs3 = np.random.RandomState(10)
    third_split = pipeline_utils.get_stratified_train_test_data(
        simple_dataframe[["obs1", "obs2"]],
        simple_dataframe["target"],
        random_state=rs1,
        test_frac=0.5,
    )

    assert (
        len(third_split.train_data.obs) == len(simple_dataframe) / 2
    ), "The test frac is passed as 50% but not returning 50% in the training data"

    fourth_split = pipeline_utils.get_stratified_train_test_data(
        simple_dataframe[["obs1", "obs2"]],
        simple_dataframe["target"],
        random_state=rs1,
        test_frac=1,
    )

    assert (
        fourth_split.train_data.obs.empty
    ), "A test frac of 1 is not returning an empty train set."
    assert fourth_split.test_data.obs.equals(
        simple_dataframe[["obs1", "obs2"]]
    ), "A test frac of 1 is not returning the passed data as the train set."
