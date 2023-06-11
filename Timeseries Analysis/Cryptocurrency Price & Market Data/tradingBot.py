# Creating a Trading Bot based on the Prophet model
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Getting the data
from pycoingecko import CoinGeckoAPI
cg = CoinGeckoAPI()
coin_gecko = cg.get_coin_market_chart_by_id('bitcoin', 'usd', 90) # 90 days
coin_gecko = coin_gecko['prices']
coin_gecko = pd.DataFrame(coin_gecko, columns=['date', 'price'])
coin_gecko = coin_gecko.rename(columns={'date': 'ds', 'price': 'y'})
coin_gecko['ds'] = pd.to_datetime(coin_gecko['ds'])

model = Prophet()
model.fit(coin_gecko)


# Instantiating the bot
class TradingBot:
    def __init__(self, coin, currency, start_date, end_date, model):
        self.coin = coin
        self.currency = currency
        self.start_date = start_date
        self.end_date = end_date
        self.model = model
        self.data = self.get_data()
        self.model.fit(self.data)
        self.future = self.create_future()
        self.forecast = self.predict_future()
        self.forecast = self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        self.forecast = self.forecast.set_index('ds')
        self.forecast = self.forecast.loc[self.start_date:self.end_date]
        self.forecast = self.forecast.rename(columns={'yhat': 'price'})
        self.forecast['price'] = self.forecast['price'].round(2)
        self.forecast['yhat_lower'] = self.forecast['yhat_lower'].round(2)
        self.forecast['yhat_upper'] = self.forecast['yhat_upper'].round(2)
        self.forecast['previous_price'] = self.forecast['price'].shift(1)
        self.forecast['previous_price'] = self.forecast['previous_price'].fillna(self.forecast['price'].iloc[0])
        self.forecast['change'] = self.forecast['price'] - self.forecast['previous_price']
        self.forecast['change'] = self.forecast['change'].round(2)
        self.forecast['signal'] = self.forecast['change'].apply(lambda x: 'buy' if x > 0 else 'sell')
        self.forecast['signal'] = self.forecast['signal'].shift(1)
        self.forecast['signal'] = self.forecast['signal'].fillna('buy')
        self.forecast['trade'] = self.forecast['signal'] != self.forecast['signal'].shift(1)
        self.forecast['trade'] = self.forecast['trade'].astype(int)
        self.forecast['trade'] = self.forecast['trade'].apply(lambda x: 'yes' if x == 1 else 'no')
        self.forecast['trade'] = self.forecast['trade'].shift(-1)
        self.forecast = self.forecast.dropna()
        self.forecast = self.forecast[['price', 'yhat_lower', 'yhat_upper', 'signal', 'trade']]
        self.forecast = self.forecast.loc[self.start_date:self.end_date]

    def get_data(self):
        coin_gecko = cg.get_coin_market_chart_by_id(self.coin, self.currency, 90)
        coin_gecko = coin_gecko['prices']
        coin_gecko = pd.DataFrame(coin_gecko, columns=['date', 'price'])
        coin_gecko['date'] = pd.to_datetime(coin_gecko['date'], unit='ms')
        coin_gecko = coin_gecko.set_index('date')
        coin_gecko = coin_gecko.loc[self.start_date:self.end_date]
        coin_gecko = coin_gecko.reset_index()
        coin_gecko = coin_gecko.rename(columns={'date': 'ds', 'price': 'y'})
        return coin_gecko
    
    def create_future(self):
        future = self.model.make_future_dataframe(periods=30, freq='D')
        return future
    
    def predict_future(self):
        forecast = self.model.predict(self.future)
        return forecast
    
    def plot_forecast(self):
        fig = self.model.plot(self.forecast, xlabel='Date', ylabel='Price')
        return fig
    
    def plot_components(self):
        fig = self.model.plot_components(self.forecast)
        return fig
    
    def get_forecast(self):
        return self.forecast
    
    def get_trade_recommendation(self):
        return self.forecast['trade'].iloc[-1]  
    
    def get_price(self):
        return self.forecast['price'].iloc[-1]
    
    def get_previous_price(self):
        return self.forecast['price'].iloc[-2]
    

# Instantiating the bot
bot = TradingBot('bitcoin', 'usd', '2021-01-01', '2021-04-01', model)

# Plotting the forecast
bot.plot_forecast()

# Plotting the components
bot.plot_components()

# Getting the forecast
bot.get_forecast()

# Getting the trade recommendation
bot.get_trade_recommendation()

# Getting the price
bot.get_price()

# Getting the previous price
bot.get_previous_price()

# to run the bot type in the terminal: python3 bot.py