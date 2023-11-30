import json
from typing import Tuple

import numpy as np
import pandas as pd
from deephyper.analysis.hpo import filter_failed_objectives

import lcdb.json


def deserialize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the dataframe for analysis. For example, load the arrays/list from json format."""

    df.sort_values("job_id", inplace=True)

    # Convert the string to JSON
    str_to_json = (
        lambda x: x.replace("'", '"').replace("nan", "NaN").replace("inf", "Infinity")
    )
    load_json = lambda x: lcdb.json.loads(str_to_json(x))
    load_array = lambda x: np.array(load_json(x))

    # Load the arrays
    columns = []
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(load_array)

    # Load the dicts
    columns = ["m:json"]
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(load_json)

    return df


def read_csv_results(path_or_buffer) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read the csv file and deserialize the dataframe.

    Args:
        path (str): path to the csv file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: the first dataframe is the one with successful runs, the second one is the one with failed runs.
    """
    df = pd.read_csv(path_or_buffer)
    df, df_failed = filter_failed_objectives(df)
    df = deserialize_dataframe(df)
    return df, df_failed


def hyperparameters_from_row(row) -> dict:
    """Return a dictionary of hyperparameters from a row of the results dataframe.

    Args:
        row (pd.Series): a row of the dataframe.

    Returns:
        dict: a dictionary of hyperparameters where keys are the hyperparameters names and values are the corresponding values.
    """
    return {k[2:]: v for k, v in row.items() if k.startswith("p:")}
