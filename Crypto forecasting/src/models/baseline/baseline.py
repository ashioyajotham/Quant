"""
A baseline model
"""
from torch import Tensor

import torch


def buy_and_hold(price_diff: Tensor) -> Tensor:
    """
    Performs the buy and hold strategy using tomorrow's price difference.
    This basically means that we are aggregating the total results from the price differences
    from each day and seeing how much they have at any given point in time

    Args:

    :param price_diff: The price difference from each day. The first
    element cooresponds to the price difference of the second - first days,
    then the second element cooresponds to the price difference between the
    third - second days + second - first days, and so fourth.
    """
    return [0] + torch.cumsum(price_diff, dim=0).view(-1).tolist()
