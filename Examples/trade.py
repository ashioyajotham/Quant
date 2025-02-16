import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MarketInefficencyTrader:
    def __init__(self, model_name="AI4Finance-Foundation/FinGPT"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.positions = {}
        self.capital = 1000000
        self.min_sentiment_threshold = 0.65  # Require strong conviction
        
    def analyze_market_psychology(self, headlines, prices):
        """Model behavioral inefficiencies in market reactions"""
        sentiment_scores = []
        for headline in headlines:
            inputs = self.tokenizer(headline, return_tensors="pt", truncation=True)
            outputs = self.model(**inputs)
            sentiment = outputs.logits.softmax(dim=-1)[0][1].item()
            
            # Check for overreaction patterns
            if sentiment > self.min_sentiment_threshold:
                # Look for price overreaction
                recent_returns = prices.pct_change().tail(3)
                if abs(recent_returns.mean()) > 2 * recent_returns.std():
                    # Market likely overreacting
                    sentiment = 0.5 + (sentiment - 0.5) * 0.5  # Dampen signal
                    
            sentiment_scores.append(sentiment)
            
        return np.mean(sentiment_scores)
    
    def detect_behavioral_patterns(self, prices, volume):
        """Identify common behavioral biases"""
        patterns = {
            'momentum_chase': False,
            'panic_selling': False,
            'fomo': False
        }
        
        # Detect momentum chasing
        returns = prices.pct_change()
        if returns.tail(5).mean() > 2 * returns.std():
            patterns['momentum_chase'] = True
            
        # Detect panic selling
        vol_spike = volume.tail(1).values[0] > 2 * volume.mean()
        price_drop = returns.tail(1).values[0] < -2 * returns.std()
        if vol_spike and price_drop:
            patterns['panic_selling'] = True
            
        # Detect FOMO
        if volume.tail(3).mean() > 2 * volume.mean() and returns.tail(3).mean() > 0:
            patterns['fomo'] = True
            
        return patterns
    
    def calculate_edge(self, sentiment, patterns, technicals):
        """Quantify trading edge based on inefficiencies"""
        edge = 0
        
        # Base edge from sentiment
        if abs(sentiment - 0.5) > 0.15:  # Require meaningful sentiment
            edge = (sentiment - 0.5) * 2  # Scale to [-1, 1]
            
        # Adjust for behavioral patterns
        if patterns['panic_selling'] and sentiment > 0.6:
            edge *= 1.2  # Stronger edge when market panics on good news
        elif patterns['fomo'] and sentiment < 0.4:
            edge *= 1.2  # Stronger edge when market FOMOs on bad news
            
        # Technical filters
        if technicals['rsi'] > 70 and edge > 0:
            edge *= 0.5  # Reduce long edge when overbought
        elif technicals['rsi'] < 30 and edge < 0:
            edge *= 0.5  # Reduce short edge when oversold
            
        return edge
    
    def calculate_technicals(self, prices):
        """Basic technical indicators"""
        close = prices['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return {
            'rsi': rsi.iloc[-1],
            'volatility': close.pct_change().std() * np.sqrt(252)
        }
    
    def size_position(self, edge, volatility):
        """Kelly-inspired position sizing with risk management"""
        # Basic Kelly fraction
        kelly = edge / (volatility ** 2)
        
        # Risk management overlays
        max_pos = 0.2  # Max 20% position
        kelly = np.clip(kelly, -max_pos, max_pos)
        
        # Reduce size if capital utilized > 50%
        utilized_capital = sum(abs(pos) for pos in self.positions.values())
        if utilized_capital > self.capital * 0.5:
            kelly *= 0.5
            
        return kelly
    
    def execute_trade(self, symbol, edge, technicals):
        """Execute trades with transaction cost modeling"""
        size = self.size_position(edge, technicals['volatility'])
        
        # Model transaction costs
        impact = 0.0002 * abs(size)  # 2bps market impact
        commission = 0.0001 * abs(size)  # 1bp commission
        total_cost = (impact + commission) * self.capital
        
        position_value = size * (self.capital - total_cost)
        
        if abs(size) > 0.05:  # Min position threshold
            self.positions[symbol] = position_value
            self.capital -= position_value + total_cost
            
    def run_strategy(self, symbol, lookback_days=30):
        """Main strategy loop"""
        # Fetch data
        data = yf.download(symbol, period=f"{lookback_days}d")
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        # Extract headlines
        headlines = [n['title'] for n in news]
        
        # Analyze inefficiencies
        sentiment = self.analyze_market_psychology(headlines, data['Close'])
        patterns = self.detect_behavioral_patterns(data['Close'], data['Volume'])
        technicals = self.calculate_technicals(data)
        
        # Calculate edge
        edge = self.calculate_edge(sentiment, patterns, technicals)
        
        if abs(edge) > 0.1:  # Only trade meaningful edges
            self.execute_trade(symbol, edge, technicals)
            
        return {
            'sentiment': sentiment,
            'patterns': patterns,
            'edge': edge,
            'position': self.positions.get(symbol, 0)
        }