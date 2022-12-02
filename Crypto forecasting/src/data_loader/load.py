"""
Author: Nathan Waltz

Implemented loading utilities for the datasets.
"""
from pathlib import Path

import pandas as pd
import numpy as np


class CryptoDataset:
    """
    Dataset class.
    """

    def __init__(self, name: str, dataset_name: str) -> None:
        self.name = name
        self.dataset_name = dataset_name

    def __str__(self) -> str:
        return self.name

    def get_dataset_name(self) -> str:
        return self.dataset_name


class DataLoader:
    """
    Helper class for loading data from a variety of sources.
    """

    def __init__(self):
        """
        Constructs a new instance of the DataLoader class.
        """
        self._data_path = (
            (Path(__file__).parents[2]).joinpath("assets").joinpath("datasets")
        )
        self.data_type_dict = {"Open": np.float32}

    def load_data(self, dataset: CryptoDataset) -> pd.DataFrame:
        """
        Loads a typed cryptocurrency dataset.

        :param dataset: The type of CryptoDataset to load.
        :return: A dataframe that corresponds to that dataset.
        """
        path = self._data_path.joinpath(dataset.get_dataset_name())
        df: pd.DataFrame = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        df.astype({col: np.float32 for col in df.columns})
        return df
