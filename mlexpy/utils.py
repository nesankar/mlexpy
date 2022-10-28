import pandas as pd
from typing import Any


def df_assertion(data_structure: Any) -> None:
    assert isinstance(data_structure, pd.DataFrame)
    f"The provided variable is not a pd.DataFrame ({type(data_structure)}). Need to pass a DataFrame."


def series_assertion(data_structure, Any) -> None:
    assert isinstance(data_structure, pd.Series)
    f"The provided variable is not a pd.Series ({type(data_structure)}). Need to pass a Series."
