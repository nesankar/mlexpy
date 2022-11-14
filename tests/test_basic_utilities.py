import pytest
from fixtures import simple_dataframe, simple_series
from mlexpy import utils, pipeline_utils
from pandas.testing import assert_frame_equal
import numpy as np


def test_filtering(simple_dataframe):
    """Make sure we are filtering as desired"""

    def filter_fn(value: int) -> bool:
        return value != 4

    filter_dict = {"obs2": [filter_fn]}
    filtered_data = utils.initial_filtering(simple_dataframe, filter_dict)
    # Assert that the filtered df returned is exactly what we wanted
    assert_frame_equal(filtered_data, simple_dataframe[simple_dataframe["obs2"] != 4])

    def filter_fn_2(value: int) -> bool:
        return value == 4

    filter_dict_2 = {"obs2": [filter_fn_2]}
    filtered_data_2 = utils.initial_filtering(simple_dataframe, filter_dict_2)
    # Assert that the filtered df returned is exactly what we wanted
    assert_frame_equal(filtered_data_2, simple_dataframe[simple_dataframe["obs2"] == 4])

    filter_dict_3 = {"obs2": [filter_fn_2, filter_fn]}
    filtered_data_3 = utils.initial_filtering(simple_dataframe, filter_dict_3)
    # Assert that 2 symmetrical conditions would result in an empty DF (something we don't want)
    assert filtered_data_3.empty


def test_assertion_functions(simple_series, simple_dataframe):

    utils.series_assertion(simple_series)
    utils.df_assertion(simple_dataframe)

    with pytest.raises(
        AssertionError,
    ):
        # Make sure we raise an exception if a series is passed to the df_assertion function
        utils.df_assertion(simple_series)
    with pytest.raises(
        AssertionError,
    ):
        # Make sure we raise an exception if a dataframe is passed to the series_assertion function
        utils.series_assertion(simple_dataframe)

    with pytest.raises(
        AssertionError,
    ):
        # Make sure we raise an exception if an ndarray is passed to the series_assertion function
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

    # Assert that we create 2 identical training sets when using the same data, and the same random seed.
    assert_frame_equal(
        first_split.train_data.obs, second_split.train_data.obs
    ), "When using identical data and random seeds, the train test split data are not identical."

    rs3 = np.random.RandomState(10)
    third_split = pipeline_utils.get_stratified_train_test_data(
        simple_dataframe[["obs1", "obs2"]],
        simple_dataframe["target"],
        random_state=rs1,
        test_frac=0.5,
    )

    # Assert that if we use a test frac of 50%, we get a dataframe of exactly half the size passed for the training set
    assert (
        len(third_split.train_data.obs) == len(simple_dataframe) / 2
    ), "The test frac is passed as 50% but not returning 50% in the training data"

    fourth_split = pipeline_utils.get_stratified_train_test_data(
        simple_dataframe[["obs1", "obs2"]],
        simple_dataframe["target"],
        random_state=rs1,
        test_frac=1,
    )

    # Assert that the special case of test_frac=1 results in no training data...
    assert (
        fourth_split.train_data.obs.empty
    ), "A test frac of 1 is not returning an empty train set."
    # ... and assert that when the test_frac=1 the testing data is the same as what was passed to the function.
    assert_frame_equal(
        fourth_split.test_data.obs, simple_dataframe[["obs1", "obs2"]]
    ), "A test frac of 1 is not returning the passed data as the train set."
