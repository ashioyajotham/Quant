from datetime import datetime
import os
import random
import numpy as np
import torch

from src.data_loader.load import DataLoader
from src.data_loader.load import CryptoDataset
from src.models.custom import custom
import src.models.oracle.oracle as oracle
from matplotlib import pyplot as plt
import pandas as pd
from tabulate import tabulate

# define the model parameters
params_oracle = {
    "lr": 0.01,
    "rnn_type": "gru",
    "rnn_hidden_size": 24,
    "ind_hidden_size": 4,
    "des_size": 16,
    "ind1": {"_name": "cmo", "length": 20},
    "ind2": {"_name": "cci", "length": 5},
}

params_custom = {
    "lr": 0.1,
    "rnn_hidden_size": 16,
    "rnn_agg_hidden_size": 12,
    "trading_ind_hidden_size": 16,
    "linear_agg_hidden_size": 4,
    "ind1": {"_name": "cmo", "length": 5},
    "ind2": {"_name": "cmo", "length": 10},
}

# set seed
seed = 80085
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# dates
start_date = "2018-01-01"
end_date = "2021-01-01"
start_pred_date = "2021-01-01"
end_pred_date = "2022-01-01"

data_loader = DataLoader()
crypto_dataset = CryptoDataset("BITCOIN", "BTC-USD.csv")
dataset = data_loader.load_data(crypto_dataset)

oracle_msg = "Learning using Oracle"
pad_len = 4
custom_msg = "Learning using Custom Model"
summary_msg = "Learning using Summary"

print("#" * (len(oracle_msg) + 2 * pad_len + 2))
print("#" + pad_len * " " + oracle_msg + pad_len * " " + "#")
print("#" * (len(oracle_msg) + 2 * pad_len + 2))

_, oracle_model = oracle.train_model(dataset, start_date, end_date, params_oracle)
ora_preds, buy_hold_preds, ora_trades = oracle.evaluate(
    dataset, start_pred_date, end_pred_date, params_oracle, oracle_model
)

d = [
    [f"Day {day}", action, returns]
    for day, (action, returns) in enumerate(zip(ora_trades, ora_preds))
]
df = pd.DataFrame(d, columns=["Day of Trading", "Shares Owned", "Model Profits (USD)"])
print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))

print("#" * (len(custom_msg) + 2 * pad_len + 2))
print("#" + pad_len * " " + custom_msg + pad_len * " " + "#")
print("#" * (len(custom_msg) + 2 * pad_len + 2))
_, custom_model = custom.train_model(
    dataset, start_date, end_date, params_custom, ["Open"]
)
cus_preds, cus_trades = custom.evaluate(
    dataset, start_pred_date, end_pred_date, params_custom, ["Open"], custom_model
)

d = [
    [f"Day {day}", action, returns]
    for day, (action, returns) in enumerate(zip(cus_trades, cus_preds))
]
df = pd.DataFrame(d, columns=["Day of Trading", "Shares Owned", "Model Profits (USD)"])
print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))

print("#" * (len(summary_msg) + 2 * pad_len + 2))
print("#" + pad_len * " " + summary_msg + pad_len * " " + "#")
print("#" * (len(summary_msg) + 2 * pad_len + 2))
d = [
    ["Buy and Hold", round(buy_hold_preds[-1], 4), np.mean(buy_hold_preds)],
    ["Oracle", round(ora_preds[-1], 4), np.mean(ora_preds)],
    ["Custom", round(cus_preds[-1], 4), np.mean(cus_preds)],
]

df = pd.DataFrame(d, columns=["Model", "Final Return", "Mean Return"])
print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))

# sanity check
assert len(ora_preds) == len(buy_hold_preds) == len(cus_preds)

# plotting logic
plt.figure(figsize=(8, 6)) 
plt.title(
    f'Trading evaluation from {(pd.Timestamp(end_pred_date) - pd.Timedelta(days=len(ora_preds))).strftime("%m/%d/%y")} to {pd.Timestamp(end_pred_date).strftime("%m/%d/%y")} on {str(crypto_dataset)}'
)
plt.plot(ora_preds, label="Oracle Returns")
plt.plot(buy_hold_preds, label="Buy and Hold Returns")
plt.plot(cus_preds, label="Model Returns")
plt.xlabel("Day of Trading")
plt.ylabel("Model Returns (USD)")
plt.xticks(rotation=45)
plt.axhline(y=0, color="black", linestyle="--")
plt.legend()
plt.savefig(os.path.join('assets', 'images', 'CryptoModelReturns.png'))
plt.show()
