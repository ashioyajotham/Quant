from collections import OrderedDict
from dataclasses import dataclass
import os
import pandas as pd
from scipy.stats import norm
from torch import device, nn, Tensor
from torch.optim import Adam
from typing import Dict, Iterable, List, Tuple

import torch
from src.models.loss import NegativeMeanReturnLoss

from src.models.utils import get_indicator, sliding_window

# run on CUDA iff it is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class CustomFeatureData:
    """
    A class for keeping track of feature-related data.
    """

    feature_name: str
    input_size: int
    hidden_size: int
    batch_first: bool = True


class CustomFeature:
    """
    A class for keeping track of custom features.
    """

    def __init__(
        self, data: Tensor, feature_name: str, input_size: int, hidden_size: int
    ) -> None:
        self.feature_data = CustomFeatureData(
            feature_name=feature_name, input_size=input_size, hidden_size=hidden_size
        )
        self.data = data


class FeatureMissingException(Exception):
    """
    A custom exception for missing features.
    """

    def __init__(self, message="That feature is missing!"):
        super().__init__(message)


class Custom(nn.Module):
    """
    Our custom model for performing buy/sell predictions.
    """

    def __init__(
        self,
        features: Iterable[CustomFeatureData],
        trading_indicator_input_size: int,
        rnn_aggregation_hidden_size: int,
        trading_indicator_hidden_size: int,
        linear_aggregator_hidden_size: int,
    ):
        """
        Constructs a new instance of the Custom class.

        Args:

        :param features: The input features (i.e. closing, and other things like that).
        :param rnn_aggregation_hidden_size: The hidden size of the rnn aggregation linear layer.
        :param trading_indicator_input_size: The number of trading indicators to take into account.
        :param trading_indicator_hidden_size: The hidden size for the linear layer that takes these trading
                                              indicators as input.
        :param linear_aggregator_hidden_size: The hidden size of the aggregator layer that aggregates both the linear layer
                                              for the indicators and linear layer that took the RNN outputs as its input.
        """
        super(Custom, self).__init__()

        # create a list of LSTMs
        rnn_dict: Dict[str, nn.RNN] = {}
        for feature in features:
            rnn = nn.LSTM(
                input_size=feature.input_size,
                hidden_size=feature.hidden_size,
                num_layers=1,
                batch_first=True,
            ).to(device=device)
            rnn.name = feature.feature_name
            rnn_dict[feature.feature_name] = rnn

        # define the model's layers
        self.rnn_dict = rnn_dict
        self.rnn_aggregation = nn.Linear(
            sum(map(lambda x: x.hidden_size * x.num_layers, self.rnn_dict.values())),
            rnn_aggregation_hidden_size,
        )
        self.linear_indicator = nn.Linear(
            trading_indicator_input_size, trading_indicator_hidden_size
        )
        self.linear_aggregator = nn.Linear(
            trading_indicator_hidden_size + rnn_aggregation_hidden_size,
            linear_aggregator_hidden_size,
        )
        self.final_linear_layer = nn.Linear(linear_aggregator_hidden_size, 1)

    def forward(self, features: Iterable[CustomFeature], indicators: Tensor) -> Tensor:
        """
        The forward method, defines how the model
        shall be run.

        Args:

        :param features: The input features (closing, etc...).
        :param indicators: Trade indicators.
        :param value_at_risk: The value at risk.
        :return: An output tensor which results after the input data
                 goes through the model. This is our prediction for
                 the number of shares to buy.
        """
        rnn_outputs = []
        for feature in features:
            feature_data = feature.feature_data
            if feature_data.feature_name not in self.rnn_dict.keys():
                raise FeatureMissingException(
                    f"The feature [<{feature_data.feature_name}>] does not exist!"
                )

            # h has dim = (num_layers, h_out)
            _, (h, _) = self.rnn_dict[feature_data.feature_name](feature.data)
            rnn_outputs.append(h)

        concatenated_outputs = torch.cat(tuple(map(lambda h: h[0], rnn_outputs)), dim=1)
        linear_from_rnn = torch.relu(self.rnn_aggregation(concatenated_outputs))
        linear_indicators = torch.relu(self.linear_indicator(indicators))
        indicators_and_linear_output = torch.cat(
            (linear_from_rnn, linear_indicators), dim=1
        )
        linear_aggregator = torch.relu(
            self.linear_aggregator(indicators_and_linear_output)
        )
        return torch.sigmoid(self.final_linear_layer(linear_aggregator)).view(-1)


def create_feature_list(
    data: Tensor, columns: Iterable[str], hidden_size: int
) -> List[CustomFeature]:
    """
    Creates a list from passed parameters of features. Specific to
    this function.

    Args:

    :param data: The data to create a feature dictionary out of.
    :param columns: The columns (features) to utilize.
    :param hidden_size: The hidden size to use for the RNN.
    """
    return [
        CustomFeature(
            data[:, :, i * 2 : (i + 1) * 2],
            f'{elem.strip().replace(" ", "_").lower()}',
            2,
            hidden_size,
        )
        for i, elem in enumerate(columns)
    ]


def train_model(
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
    parameters: Dict,
    columns: List[str],
    window_size: int = 40,
    num_epochs: int = 500,
    train_val_ratio: float = 0.8,
) -> Tuple[float, nn.Module]:
    """
    Trains the custom model using the provided data, start_date, and end_date.
    This is the "heart" of what we are doing, hyperparameter tuning is very important!

    Args:

    :param data: Columns that might be useful for learning.
    :param start_date: The date that signifies the first possible date that can
                       be included in any time window.
    :param end_date: The last date that can be included in any time window.
    :param parameters: The model's hyperparameters.
    :param window_size: The window size, defaults to 40.
    :param num_epochs: The number of training epochs to use, defaults to 500.
    :param train_val_ratio: The ratio of training data to validation data, defaults
                            to 0.8.
    """
    # cut after end date
    data = data[data.index < end_date]
    ts_len = data[data.index > start_date].shape[0]
    train_length = int(ts_len * train_val_ratio)

    # get the hyperparameters
    learning_rate = parameters["lr"]
    rnn_hidden_size = parameters["rnn_hidden_size"]
    rnn_aggregation_hidden_size = parameters["rnn_agg_hidden_size"]
    linear_aggregator_hidden_size = parameters["linear_agg_hidden_size"]
    trading_indicator_hidden_size = parameters["trading_ind_hidden_size"]
    ind1_name = parameters["ind1"]["_name"]
    ind2_name = parameters["ind2"]["_name"]

    # an ordered dictionary of data
    data_source = OrderedDict()
    for name in columns:
        key_prefix = name.strip().replace(" ", "_").lower()
        data_source[f"{key_prefix}_diff"] = data[name] - data[name].shift(1)
        data_source[f"{key_prefix}_roc"] = data[name] / data[name].shift(1)

    data_source["ind1"] = get_indicator(data, ind1_name, parameters["ind1"])
    data_source["ind2"] = get_indicator(data, ind2_name, parameters["ind2"])

    # add value at risk as an indicator
    pct_changes = data["Close"].pct_change()
    value_at_risk = []
    for i in range(len(pct_changes)):
        mean = pct_changes[:i].mean()
        std = pct_changes[:i].std()
        value_at_risk_pct = abs(norm.ppf(0.01, mean, std))
        value_at_risk.append(value_at_risk_pct)

    data_source["var"] = pd.Series(value_at_risk)
    data_source["var"].index = pct_changes.index

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

    train_list = create_feature_list(x_train, columns, rnn_hidden_size)
    val_list = create_feature_list(x_val, columns, rnn_hidden_size)

    index_train, index_val = (
        x_train[:, -1, 2 * len(columns) :],
        x_val[:, -1, 2 * len(columns) :],
    )
    price_train, price_val = y_train[:, :, 0].view(-1), y_val[:, :, 0].view(-1)

    model_params = {
        "features": map(lambda feature: feature.feature_data, train_list),
        "trading_indicator_input_size": 3,
        "linear_aggregator_hidden_size": linear_aggregator_hidden_size,
        "rnn_aggregation_hidden_size": rnn_aggregation_hidden_size,
        "trading_indicator_hidden_size": trading_indicator_hidden_size,
    }

    model: nn.Module = Custom(**model_params).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = NegativeMeanReturnLoss()

    # train the model
    val_losses = []
    for e in range(1, num_epochs + 1):
        model.train()
        predicted = model(train_list, index_train)
        loss = criterion(predicted, price_train)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_predicted = model(val_list, index_val)
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
    columns: List[str],
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
    # cut after end date
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = data[data.index < end_date]
    ts_len = data[data.index > start_date].shape[0]

    # get the hyperparameters
    rnn_hidden_size = parameters["rnn_hidden_size"]
    rnn_aggregation_hidden_size = parameters["rnn_agg_hidden_size"]
    linear_aggregator_hidden_size = parameters["linear_agg_hidden_size"]
    trading_indicator_hidden_size = parameters["trading_ind_hidden_size"]
    ind1_name = parameters["ind1"]["_name"]
    ind2_name = parameters["ind2"]["_name"]

    # an ordered dictionary of data
    data_source = OrderedDict()
    col_list = columns
    for name in col_list:
        key_prefix = name.strip().replace(" ", "_").lower()
        data_source[f"{key_prefix}_diff"] = data[name] - data[name].shift(1)
        data_source[f"{key_prefix}_roc"] = data[name] / data[name].shift(1)

    data_source["ind1"] = get_indicator(data, ind1_name, parameters["ind1"])
    data_source["ind2"] = get_indicator(data, ind2_name, parameters["ind2"])

    # add value at risk as an indicator
    pct_changes = data["Close"].pct_change()
    value_at_risk = []
    for i in range(len(pct_changes)):
        mean = pct_changes[:i].mean()
        std = pct_changes[:i].std()
        value_at_risk_pct = abs(norm.ppf(0.01, mean, std))
        value_at_risk.append(value_at_risk_pct)

    data_source["var"] = pd.Series(value_at_risk)
    data_source["var"].index = pct_changes.index

    for k, v in data_source.items():
        data_source[k] = v[v.index >= start_date].dropna().values

    # aggregate all of that data
    data_aggregation = [[v[i] for _, v in data_source.items()] for i in range(ts_len)]

    x, y = sliding_window(data_aggregation, window_size)
    x = torch.tensor(x).to(device).float()
    y = torch.tensor(y).to(device).float()

    feature_list = create_feature_list(x, col_list, rnn_hidden_size)
    index = x[:, -1, 2 * len(col_list) :]
    tomorrow_price_diff = y[:, :, 0].view(-1)

    model_params = {
        "features": map(lambda feature: feature.feature_data, feature_list),
        "trading_indicator_input_size": 3,
        "linear_aggregator_hidden_size": linear_aggregator_hidden_size,
        "rnn_aggregation_hidden_size": rnn_aggregation_hidden_size,
        "trading_indicator_hidden_size": trading_indicator_hidden_size,
    }

    if model is None:
        model = Custom(**model_params).to(device)
        model.load_state_dict(torch.load(f"{dir_path}/data/custom_best.pth"))

    # evaluate the model
    model.eval()
    with torch.no_grad():
        trades = model(feature_list, index)
        # Rounded Trades
        trades = torch.round(trades)

        # print(trades)

        # Calculating Absolute Returns
        abs_return = torch.mul(trades, tomorrow_price_diff)
        cumsum_return = [0] + torch.cumsum(abs_return, dim=0).view(-1).tolist()
        # # Buy and Hold Strategy Returns
        # buy_and_hold_returns = buy_and_hold(tomorrow_price_diff)

        # print(f'Model Returns: {round(cumsum_return[-1], 4)}')
        # print(f'Model Mean returns: {np.mean(cumsum_return)}')
        # print(f'Buy and Hold Returns: {round(buy_and_hold_returns[-1], 4)}')
        # print(f'Buy and Hold Mean Returns: {np.mean(buy_and_hold_returns)}')

        # plt.title(f'Trading evaluation from {start_date} to {end_date}')
        # plt.plot(cumsum_return, label='Model Returns')
        # plt.plot(buy_and_hold_returns, label='Buy and Hold Returns')
        # plt.axhline(y=0, color='black', linestyle='--')
        # plt.legend()
        # plt.show()

    return cumsum_return, trades
