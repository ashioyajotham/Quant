# FinRL 2.0: Deep Reinforcement Learning in Quantitative Finance
#
# Deep Reinforcement Learning for Stock Trading from Scratch: 
# A Practical Guide with Python Accelerated Finance (2nd ed.)
#
# Code Repository: https://github.com/AI4Finance-LLC/FinRL

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import sys

sys.path.append("C://Users//Administrator//FinRL") #

import itertools
import re
from finrl import config
import itertools

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline

from finrl.plot import backtest_plot
from finrl.plot import backtest_stats
from finrl.plot import get_baseline
from finrl.plot import get_daily_return
from finrl.plot import plot_return
from pprint import pprint

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--stock', default='AAPL', type=str)
    parser.add_argument('--window_size', default=10, type=int)
    parser.add_argument('--start_date', default='2018-01-01', type=str)
    parser.add_argument('--end_date', default='2021-01-01', type=str)
    parser.add_argument('--model_name', default='ppo', type=str)
    parser.add_argument('--model_params', default='default', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--output_dir', default='../output/', type=str)
    parser.add_argument('--baseline', default=False, type=bool)
    parser.add_argument('--baseline_buy_and_hold', default=False, type=bool)
    parser.add_argument('--baseline_random', default=False, type=bool)
    parser.add_argument('--baseline_rl', default=False, type=bool)
    parser.add_argument('--baseline_rl_agent', default='ppo', type=str)
    parser.add_argument('--baseline_rl_params', default='default', type=str)
    parser.add_argument('--baseline_rl_train', default=False, type=bool)
    parser.add_argument('--baseline_rl_train_start_date', default='2018-01-01', type=str)
    parser.add_argument('--baseline_rl_train_end_date', default='2020-01-01', type=str)
    parser.add_argument('--baseline_rl_train_model_name', default='ppo', type=str)
    parser.add_argument('--baseline_rl_train_model_params', default='default', type=str)
    parser.add_argument('--baseline_rl_train_output_dir', default='../output/', type=str)
    parser.add_argument('--baseline_rl_train_gpu', default=0, type=int)
    parser.add_argument('--baseline_rl_train_window_size', default=10, type=int)
    parser.add_argument('--baseline_rl_train_train', default=False, type=bool)
    parser.add_argument('--baseline_rl_train_train_start_date', default='2018-01-01', type=str)
    parser.add_argument('--baseline_rl_train_train_end_date', default='2020-01-01', type=str)
    parser.add_argument('--baseline_rl_train_train_model_name', default='ppo', type=str)
    parser.add_argument('--baseline_rl_train_train_model_params', default
                        ='default', type=str)
    parser.add_argument('--baseline_rl_train_train_output_dir', default='../output/', type=str)
    parser.add_argument('--baseline_rl_train_train_gpu', default=0, type=int)

    args = parser.parse_args()
    

    mode = args.mode
    stock = args.stock
    window_size = args.window_size
    start_date = args.start_date
    end_date = args.end_date
    model_name = args.model_name
    model_params = args.model


    # The above returns a Namespace, which is a simple class that has attributes and methods for us to use:
    # print(args)

    # You can access the attributes of the Namespace by name:
    # print(args.stock)

    # To run in command line:
    # python trade.py --mode train --stock AAPL --window_size 10 --start_date 2018-01-01 --end_date 2021-01-01 --model_name ppo --model_params default --gpu 0 --data_dir ../data/ --output_dir ../output/ --baseline False --baseline_buy_and_hold False --baseline_random False --baseline_rl False --baseline_rl_agent ppo --baseline_rl_params default --baseline_rl_train False --baseline_rl_train_start_date 2018-01-01 --baseline_rl_train_end_date 2020-01-01 --baseline_rl_train_model_name ppo --baseline_rl_train_model_params default --baseline_rl_train_output_dir ../output/ --baseline_rl_train_gpu 0 --baseline_rl_train_window_size 10 --baseline_rl_train_train False --baseline_rl_train_train_start_date 2018-01-01 --baseline_rl_train_train_end_date 2020-01-01 --baseline_rl_train_train_model_name ppo --baseline_rl_train_train_model_params default --baseline_rl_train_train_output_dir ../output/ --baseline_rl_train_train_gpu 0
