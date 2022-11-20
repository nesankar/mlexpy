import pandas as pd
from typing import Any, List, Callable, Dict
import logging
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_directory(directory_path: Path) -> None:
    """Create a directory defined by the directory path, including creating all intermediate directories.

    Parameters
    ----------
    directory_path : Path
        The full path to the directory you would like to create.

    Returns
    -------
    None
    """
    logger.info(f"Creating the directory(s) at {directory_path}.")
    logger.info(f"Current directory(s) {Path.cwd()}")
    os.makedirs(directory_path)


def df_assertion(data_structure: Any) -> None:
    """Simply assert that the passed value is a dataframe type.

    Parameters
    ----------
    data_structure : Any
        The value to check the type of.

    Returns
    -------
    None

    Raises
    ------
    AssertionException
        If you didn't pass a dataframe
    """
    assert isinstance(
        data_structure, pd.DataFrame
    ), f"The provided variable is not a pd.DataFrame ({type(data_structure)}). Need to pass a DataFrame."


def series_assertion(data_structure: Any) -> None:
    """Simply assert that the passed value is a Series type.

    Parameters
    ----------
    data_structure : Any
        The value to check the type of.

    Returns
    -------
    None

    Raises
    ------
    AssertionException
        If you didn't pass a Series
    """
    assert isinstance(
        data_structure, pd.Series
    ), f"The provided variable is not a pd.Series ({type(data_structure)}). Need to pass a Series."


def initial_filtering(
    df: pd.DataFrame, column_mask_functions: Dict[str, List[Callable]]
) -> pd.DataFrame:
    """For each column in the initial dataframe, create a masking function to
    define a boolean to keep, or not keep the record.

    In the end, pass over every record and keep only the records that are all True.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe you would like to perform filtering on.
    column_mask_functions : Dict[str, List[Callable]]
        The Dictionary of masking functions to apply. Each function is only applied to the
        respective key in the dictionary. The masking functions are by default a list in
        order to allow multiple functions per column.

    Returns
    -------
    pd.DataFrame
    """

    initial_size = df.shape
    logger.info(
        f"Performing basic filtering on the {column_mask_functions.keys()} columns, of df of size {initial_size}."
    )

    # Now begin the filtering process
    for column_name, mask_list in column_mask_functions.items():
        if column_name not in df.columns:
            continue
        for i, mask_fn in enumerate(mask_list):
            df = df[df[column_name].apply(mask_fn)]

    post_size = df.shape
    drop_pct = round((initial_size[0] - post_size[0]) / initial_size[0] * 100)
    logger.info(
        f"Following filtering the data size is {post_size}. ({drop_pct}% of rows dropped)"
    )
    return df
