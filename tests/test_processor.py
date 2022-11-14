import pytest
from fixtures import simple_dataframe, base_processor, to_scale_dataframe
from mlexpy import processor
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
import numpy as np


def test_basic_processor_exceptions(base_processor, simple_dataframe):
    """Test that things don't work without defining the base processor as expected."""

    # We can't call a processor...
    with pytest.raises(
        NotImplementedError, match="This needs to be implemented in the child class."
    ):
        base_processor.process_data(simple_dataframe)

    # ... and we cant fit model based features b/c we don't know the features.
    with pytest.raises(
        NotImplementedError, match="This needs to be implemented in the child class."
    ):
        base_processor.fit_model_based_features(simple_dataframe)


def test_basic_capabilities(base_processor, to_scale_dataframe):

    # We can do some basic processing, ex. label encoding...
    labels = ["a", "b", "c", "c", "c", "b"]
    series = pd.Series(labels, index=list(range(len(labels))), name="labels")
    base_processor.fit_label_encoder(series)
    encoded_labels = base_processor.encode_labels(series)
    assert_series_equal(
        encoded_labels,
        pd.Series([0, 1, 2, 2, 2, 1], index=list(range(len(labels))), name="labels"),
    ), "Failure to correctly encode labels."

    # ... ex. a min-max scaler
    base_processor.fit_scaler(to_scale_dataframe["obs3"], standard_scaling=False)
    transformed_df = base_processor.transform_model_based_features(to_scale_dataframe)
    equivalent = to_scale_dataframe["obs3"] / to_scale_dataframe["obs3"].max()
    equivalent.name = "obs3_minmaxscaler()"

    print(transformed_df["obs3_minmaxscaler()"])
    print(equivalent)

    assert_series_equal(
        transformed_df["obs3_minmaxscaler()"], equivalent, check_less_precise=True
    ), "Failure to correctly apply a min-max scaler"


def test_no_transform_to_unnamed(base_processor, to_scale_dataframe):
    """Make sure we don't process any unnamed columns"""

    for col in to_scale_dataframe.columns:
        base_processor.fit_scaler(to_scale_dataframe[col], standard_scaling=False)

    transformed_df = base_processor.transform_model_based_features(to_scale_dataframe)
    scaled_columns = [col for col in transformed_df if "minmax" in col]

    assert all(
        ["Unnamed:" not in col for col in scaled_columns]
    ), "Failure to NOT transform an 'Unnamed: ' column."
