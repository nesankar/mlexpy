import pytest
import pandas as pd


@pytest.fixture()
def simple_dataframe():
    var_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    var_2 = [0, 2, 1, 2, 3, 1, 2, 3, 4, 1]

    target = [v1 ** var_2[i] for i, v1 in enumerate(var_1)]

    return pd.DataFrame(
        zip(var_1, var_2, target),
        columns=["obs1", "obs2", "target"],
        index=list(range(len(var_1))),
    )


@pytest.fixture()
def simple_series(simple_dataframe):
    return simple_dataframe["target"]
