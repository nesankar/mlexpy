import pytest
from fixtures import (
    simple_dataframe,
    base_processor,
    to_scale_dataframe,
    one_hot_dataframe,
)
from mlexpy import processor
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
from pathlib import Path
import sys


def test_basic_processor_exceptions(base_processor, simple_dataframe):
    """Test that things don't work without defining the base processor as expected."""

    # We can't call a processor...
    with pytest.raises(
        NotImplementedError, match="This needs to be implemented in the child class."
    ):
        # Assert that we raise a NotImplementedError here b/c this needs to be done in the class inheriting this class.
        base_processor.process_data(simple_dataframe)

    # ... and we cant fit model based features b/c we don't know the features.
    with pytest.raises(
        NotImplementedError, match="This needs to be implemented in the child class."
    ):
        # Assert that we raise a NotImplementedError here b/c this needs to be done in the class inheriting this class.
        base_processor.fit_model_based_features(simple_dataframe)


def test_basic_capabilities(base_processor, to_scale_dataframe):

    # We can do some basic processing, ex. label encoding...
    labels = ["a", "b", "c", "c", "c", "b"]
    series = pd.Series(labels, index=list(range(len(labels))), name="labels")
    base_processor.fit_label_encoder(series)
    encoded_labels = base_processor.encode_labels(series)

    # Assert that the label encoded encodes string labels as we would expect.
    assert_series_equal(
        encoded_labels,
        pd.Series([0, 1, 2, 2, 2, 1], index=list(range(len(labels))), name="labels"),
    ), "Failure to correctly encode labels."

    # ... ex. a min-max scaler
    base_processor.fit_scaler(to_scale_dataframe["obs3"], standard_scaling=False)
    transformed_df = base_processor.transform_model_based_features(to_scale_dataframe)
    equivalent = to_scale_dataframe["obs3"] / to_scale_dataframe["obs3"].max()
    equivalent.name = "obs3_minmaxscaler"

    # Assert that the min-max encoder both (1) operates as expected, and (2) returns the values we expect.
    assert_series_equal(
        transformed_df["obs3_minmaxscaler"], equivalent, check_less_precise=True
    ), "Failure to correctly apply a min-max scaler"


def test_no_transform_to_unnamed(base_processor, to_scale_dataframe):
    """Make sure we don't process any unnamed columns"""

    for col in to_scale_dataframe.columns:
        base_processor.fit_scaler(to_scale_dataframe[col], standard_scaling=False)

    transformed_df = base_processor.transform_model_based_features(to_scale_dataframe)
    scaled_columns = [col for col in transformed_df if "minmax" in col]

    # Assert that there are NO columns generated with models applied to the Unnamed: columns.
    assert all(
        ["Unnamed:" not in col for col in scaled_columns]
    ), "Failure to NOT transform an 'Unnamed: ' column."


def test_column_logic(to_scale_dataframe):
    """Test that we can keep and drop columns as expected"""

    pcr = processor.ProcessPipelineBase()

    # Test we drop this colum
    assert_frame_equal(
        pcr.drop_columns(to_scale_dataframe.copy(), ["obs1"]),
        to_scale_dataframe[["obs2", "obs3", "Unnamed: 0"]],
    )

    # Test that we keep the columns we want
    assert_frame_equal(
        pcr.keep_columns(
            to_scale_dataframe.copy(),
            keep_cols=[col for col in to_scale_dataframe.columns if "obs" in col],
        ),
        to_scale_dataframe[["obs1", "obs2", "obs3"]],
    )


def test_directory_functions():
    """Test that we can successfully define our file structure"""

    pcr = processor.ProcessPipelineBase(
        process_tag="test_example", model_dir=Path(__file__)
    )

    # Assert that we define the path as expected
    assert pcr.model_dir == Path(__file__) / "test_example"

    pcr_string = processor.ProcessPipelineBase(
        process_tag="test_example", model_dir=str(Path(__file__))
    )

    # Assert that we create the correct path even if passing a string as the model directory
    assert pcr_string.model_dir == Path(__file__) / "test_example"

    pcr_none = processor.ProcessPipelineBase(
        process_tag="test_example",
    )

    # Assert that we create the correct path even if not passing a model directory
    assert pcr_none.model_dir == Path(sys.path[-1]) / ".models" / "test_example"


def test_onehot_encoding(one_hot_dataframe, base_processor):

    base_processor.fit_one_hot_encoding(one_hot_dataframe["obs4"])
    encoded_df = base_processor.transform_model_based_features(one_hot_dataframe)
    print(encoded_df.columns)

    result_df = pd.concat(
        [
            one_hot_dataframe,
            pd.DataFrame(
                zip([1.0, 0.0, 0.0], [0.0, 1.0, 1.0]),
                columns=["obs4_onehotencoder_x0_a", "obs4_onehotencoder_x0_b"],
                index=[0, 1, 2],
            ),
        ],
        axis=1,
    )
    print(result_df.columns)

    # Test that the output of the one hot encoding is exactly how we expect it
    assert_frame_equal(
        encoded_df,
        result_df,
    )
