from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
from alpaca_trade_api import REST
from timedelta import Timedelta
from sentiment import predict_sentiment
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
BASE_URL = os.getenv('BASE_URL')

ALPACA_CREDS = {
    'API_KEY': API_KEY,
    'API_SECRET': API_SECRET,
    'PAPER': True
}


class SentimentTrader(Strategy):
    def initialize(self, symbol: str = 'SPY', cash_at_risk: float = 0.5):
        self.symbol = symbol
        self.sleeptime = '24H'
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price)
        return cash, last_price, quantity

    def get_sentiment(self):
        news_list = self.api.get_news(
            symbol=self.symbol,
            start=(self.get_datetime() - Timedelta(days=3)).strftime('%Y-%m-%d'),
            end=self.get_datetime().strftime('%Y-%m-%d')
        )
        headlines = [news.headline for news in news_list]
        return predict_sentiment(headlines)
        

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()

        if cash > last_price:
            if sentiment == 'positive' and probability > 0.999: 
                if self.last_trade == 'sell':
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    'buy',
                    type='bracket',
                    take_profit_price = last_price * 1.20,
                    stop_loss_price = last_price * 0.95
                )
                self.submit_order(order)
                self.last_trade = 'buy'
            elif sentiment == 'negative' and probability > 0.999: 
                if self.last_trade == 'buy':
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    'sell',
                    type='bracket',
                    take_profit_price = last_price * 0.8,
                    stop_loss_price = last_price * 1.05
                )
                self.submit_order(order)
                self.last_trade = 'sell'


broker = Alpaca(ALPACA_CREDS)
strategy = SentimentTrader(name='sentiment_strat', broker=broker, parameters={'symbol': 'SPY', 'cash_at_risk': 0.5})

strategy.backtest(
    YahooDataBacktesting,
    datetime(2020, 1, 1),
    datetime(2025, 1, 1),
    parameters={'symbol': 'SPY', 'cash_at_risk': 0.5}
)