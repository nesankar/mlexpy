import pytest
import pandas as pd

from mlexpy import processor


@pytest.fixture()
def simple_dataframe() -> pd.DataFrame:
    var_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    var_2 = [0, 2, 1, 2, 3, 1, 2, 3, 4, 1]

    target = [v1 ** var_2[i] for i, v1 in enumerate(var_1)]

    return pd.DataFrame(
        zip(var_1, var_2, target),
        columns=["obs1", "obs2", "target"],
        index=list(range(len(var_1))),
    )


@pytest.fixture()
def simple_series(simple_dataframe) -> pd.Series:
    return simple_dataframe["target"]


@pytest.fixture()
def base_processor() -> processor.ProcessPipelineBase:
    """Create a base data processor class to test."""
    return processor.ProcessPipelineBase()


@pytest.fixture()
def to_scale_dataframe() -> pd.DataFrame:
    var_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    var_2 = [0, 2, 1, 2, 3, 1, 2, 3, 4, 1]
    var_3 = [0, 2, 3, 4, 5, 5, 5, 5, 5, 10]

    target = [v1 ** var_2[i] for i, v1 in enumerate(var_1)]

    return pd.DataFrame(
        zip(var_1, var_2, var_3, var_3),
        columns=["obs1", "obs2", "obs3", "Unnamed: 0"],
        index=list(range(len(var_1))),
    )