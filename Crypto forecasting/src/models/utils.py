from typing import List, Tuple, Dict, Union

import numpy as np
import pandas as pd
import pandas_ta as ta


def sliding_window(
    time_series: List[List[np.float32]], num_features: int, target_len: int = 1
) -> Tuple[List[List[List[np.float32]]], List[List[List[np.float32]]]]:
    """
    The sliding window for a given time series. Slices the passed
    time_series up into windows with length num_features and target length
    target_len.

    :param time_series: The time series to slice up.
    :param num_features: The length of time windows to use.
    :param target_len: The length of the target, or Y.
    :return: Time windows for X and Y.
    """
    x = [
        time_series[i : i + num_features]
        for i in range(len(time_series) + 1 - num_features - target_len)
    ]
    y = [
        time_series[i : i + target_len]
        for i in range(num_features, len(time_series) + 1 - target_len)
    ]
    return x, y


def get_indicator(
    dataframe: pd.DataFrame, indicator_name: str, parameters: Dict[str, Union[int, str]]
) -> pd.DataFrame:
    """
    Gets a specified indicator with a given indicator name.

    :param dataframe: A pandas dataframe.
    :param indicator_name: The name of the indicator.
    :param parameters: Relevant parameters.
    :return: A new dataframe with the indicator operation performed on it.
    """
    if indicator_name == "ao":
        return dataframe.ta.ao(parameters["fast"], parameters["slow"])
    elif indicator_name == "apo":
        return dataframe.ta.apo(parameters["fast"], parameters["slow"])
    elif indicator_name == "cci":
        return dataframe.ta.cci(parameters["length"]) / 100
    elif indicator_name == "cmo":
        return dataframe.ta.cmo(parameters["length"]) / 100
    elif indicator_name == "mom":
        return dataframe.ta.mom(parameters["length"])
    elif indicator_name == "rsi":
        return dataframe.ta.rsi(parameters["length"]) / 100

    raise Exception("That indicator is not supported")
