"""
An Oracle for the crypto problem. Using a "stock market prediction" sample
found in Ivan Gridin's book "Time Series Forecasting using Deep Learning:
Combining Pytorch, RNN, TCN, and Deep Neural Network Models to Provide
Production-Ready Prediction Solutions." with ISBN-13: 978-9391392574.

The codebase has an MIT license, so I am assuming it is safe to use:
https://github.com/bpbpublications/Time-Series-Forecasting-using-Deep-Learning/tree/main/Chapter%2007/stock
"""
import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

# trust me, we need this for the trading indicators
import torch
import torch.nn as nn

from collections import OrderedDict

from scipy.stats import norm
from torch.optim import Adam
from src.models.baseline.baseline import buy_and_hold
from src.models.loss import NegativeMeanReturnLoss
from src.models.utils import get_indicator, sliding_window

# check whether it can run on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')


class Oracle(nn.Module):
    """
    The Oracle, adapted from the Ivan Gridin's solution in 978-9391392574.
    """

    def __init__(
        self,
        rnn_input_size: int,
        ind_input_size: int,
        rnn_type: str = "gru",
        rnn_hidden_size: int = 16,
        ind_hidden_size: int = 4,
        des_size: int = 4,
    ):
        """
        Constructs a new instance of the Oracle class.
        """
        super(Oracle, self).__init__()
        self.oracle_model = "Ivan's Alg Trader"
        rnn_params = {
            "input_size": rnn_input_size,
            "hidden_size": rnn_hidden_size,
            "batch_first": True,
        }

        if rnn_type == "gru":
            self.rnn = nn.GRU(**rnn_params)
        elif rnn_type == "rnn":
            self.rnn = nn.RNN(**rnn_params)
        else:
            raise Exception(f"This type of RNN is not supported: {rnn_type}")

        self.lin_ind = nn.Linear(ind_input_size, ind_hidden_size)
        self.lin_des = nn.Linear(rnn_hidden_size + ind_hidden_size, des_size)
        self.lin_pos = nn.Linear(des_size, 1)

    def forward(self, raw: torch.Tensor, indicators: torch.Tensor) -> torch.Tensor:
        """
        The forward method, defines how the model
        shall be run.

        :param raw: The raw data.
        :param indicators: Trade indicators
        :return: An output tensor which results after
        the input data goes through the model.
        """
        _, h = self.rnn(raw)
        z = torch.relu(self.lin_ind(indicators))
        x = torch.cat((z, h[0]), dim=1)
        x = torch.relu(self.lin_des(x))
        p = torch.tanh(self.lin_pos(x))
        return p.view(-1)


def train_model(
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
    parameters: Dict,
    window_size: int = 40,
    num_epochs: int = 500,
    train_val_ratio: float = 0.8,
) -> Tuple[float, nn.Module]:
    """
    Trains the model.

    :param data: The data to use.
    :param start_date: The date to start with.
    :param end_date: The date to end with.
    :param window_size: The window size.
    :param num_epochs: The number of epochs to train the model for.
    :param train_val_ratio: How much we want for train and validation.
    :param parameters: Model hyperparameters.
    :return: The minimum validation loss and the model.
    """
    # cut after end date
    data = data[data.index < end_date]
    ts_len = data[data.index > start_date].shape[0]
    train_length = int(ts_len * train_val_ratio)

    # get the hyperparameters
    learning_rate = parameters["lr"]
    rnn_type = parameters["rnn_type"]
    rnn_hidden_size = parameters["rnn_hidden_size"]
    ind_hidden_size = parameters["ind_hidden_size"]
    des_size = parameters["des_size"]
    ind1_name = parameters["ind1"]["_name"]
    ind2_name = parameters["ind2"]["_name"]

    # an ordered dictionary of data
    data_source = OrderedDict()
    data_source["open_diff"] = data["Open"] - data["Open"].shift(1)
    data_source["open_roc"] = data["Open"] / data["Open"].shift(1)
    data_source["ind1"] = get_indicator(data, ind1_name, parameters["ind1"])
    data_source["ind2"] = get_indicator(data, ind2_name, parameters["ind2"])

    # Cut to 'start date'
    for k, v in data_source.items():
        data_source[k] = v[v.index >= start_date].dropna().values

    # aggregate all of that data
    data_aggregation = [[v[i] for _, v in data_source.items()] for i in range(ts_len)]

    x, y = sliding_window(data_aggregation, window_size)
    x_train, y_train = x[:train_length], y[:train_length]
    x_val, y_val = x[train_length:], y[train_length:]

    x_train = torch.tensor(x_train).to(device).float()
    y_train = torch.tensor(y_train).to(device).float()
    x_val = torch.tensor(x_val).to(device).float()
    y_val = torch.tensor(y_val).to(device).float()

    # get separate inputs, outputs
    open_train, open_val = x_train[:, :, :2], x_val[:, :, :2]
    index_train, index_val = x_train[:, -1, 2:], x_val[:, -1, 2:]
    price_train, price_val = y_train[:, :, 0].view(-1), y_val[:, :, 0].view(-1)

    # Initialize model
    model_params = {
        "rnn_input_size": 2,
        "ind_input_size": 2,
        "rnn_type": rnn_type,
        "rnn_hidden_size": rnn_hidden_size,
        "ind_hidden_size": ind_hidden_size,
        "des_size": des_size,
    }
    model = Oracle(**model_params).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = NegativeMeanReturnLoss()

    # train the model
    val_losses = []
    for e in range(1, num_epochs + 1):
        model.train()
        predicted = model(open_train, index_train)
        loss = criterion(predicted, price_train)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_predicted = model(open_val, index_val)
            val_loss = criterion(val_predicted, price_val)
            val_losses.append(val_loss)

        if (e + 1) % 100 == 0:
            print(f"Epoch {e+1} | train: {loss.item()}, " f"val: {val_loss.item()}")

    return torch.mean(torch.tensor(val_losses)).item(), model


def evaluate(
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
    parameters: Dict,
    model: nn.Module = None,
    window_size: int = 40,
) -> None:
    """
    Evaluates the model.

    :param model: The model that is being evaluated.
    :param start_date: The date to start with.
    :param end_date: The date to end with.
    :param data: The data that is being used to train the model.
    :param parameters: Model hyperparameters.
    :param window_size: The window size.
    :return: Nothing.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = data[data.index < end_date]
    ts_len = data[data.index > start_date].shape[0]

    # an ordered dictionary of data
    data_source = OrderedDict()
    data_source["open_diff"] = data["Open"] - data["Open"].shift(1)
    data_source["open_roc"] = data["Open"] / data["Open"].shift(1)
    # data_source['pct_change'] = data['Open'].pct_change()
    data_source["ind1"] = get_indicator(
        data, parameters["ind1"]["_name"], parameters["ind1"]
    )
    data_source["ind2"] = get_indicator(
        data, parameters["ind2"]["_name"], parameters["ind2"]
    )
    data_source["open_diff"][0] = 0
    data_source["open_roc"][0] = 1

    # Cut to 'start date'
    for k, v in data_source.items():
        data_source[k] = v[v.index >= start_date].dropna().values

    # aggregate all of that data
    data_aggregation = [[v[i] for _, v in data_source.items()] for i in range(ts_len)]

    x, y = sliding_window(data_aggregation, window_size)
    x = torch.tensor(x).to(device).float()
    y = torch.tensor(y).to(device).float()

    cost = x[:, :, :2]
    index = x[:, -1, 2:]
    tomorrow_price_diff = y[:, :, 0].view(-1)

    model_params = {
        "rnn_input_size": 2,
        "ind_input_size": 2,
        "rnn_type": parameters["rnn_type"],
        "rnn_hidden_size": parameters["rnn_hidden_size"],
        "ind_hidden_size": parameters["ind_hidden_size"],
        "des_size": parameters["des_size"],
    }

    if model is None:
        model = Oracle(**model_params).to(device)
        model.load_state_dict(torch.load(f"{dir_path}/data/oracle_best.pth"))

    # evaluate the model
    model.eval()
    with torch.no_grad():
        trades = model(cost, index)
        # Rounded Trades
        trades = torch.round(trades)

        # Calculating Absolute Returns
        abs_return = torch.mul(trades, tomorrow_price_diff)
        cumsum_return = [0] + torch.cumsum(abs_return, dim=0).view(-1).tolist()
        # Buy and Hold Strategy Returns
        buy_and_hold_returns = buy_and_hold(tomorrow_price_diff)

        # print(f'Buy and Hold Returns: {round(buy_and_hold_returns[-1], 4)}')
        # print(f'Buy and Hold Mean Returns: {np.mean(buy_and_hold_returns)}')
        # print(f'Oracle Returns: {round(cumsum_return[-1], 4)}')
        # print(f'Oracle Mean returns: {np.mean(cumsum_return)}')

        # plt.title(f'Trading evaluation from {start_date} to {end_date}')
        # plt.plot(cumsum_return, label='Model Returns')
        # plt.plot(buy_and_hold_returns, label='Buy and Hold Returns')
        # plt.axhline(y=0, color='black', linestyle='--')
        # plt.legend()
        # plt.show()

    return cumsum_return, buy_and_hold_returns, trades
