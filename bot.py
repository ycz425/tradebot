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
    def initialize(self, symbol: str = 'AAPL'):
        self.asset = symbol
        self.sleeptime = '24H'
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)


    def position_sizing(self):
        value = self.get_cash() * 0.25
        return int(value / self.get_last_price(self.asset))


    def get_sentiment(self):
        news_list = self.api.get_news(
            symbol=self.asset,
            start=(self.get_datetime() - Timedelta(days=3)).strftime('%Y-%m-%d'),
            end=self.get_datetime().strftime('%Y-%m-%d')
        )
        headlines = [news.headline for news in news_list]
        return predict_sentiment(headlines)


    def on_trading_iteration(self):
        probability, sentiment = self.get_sentiment()
        current_price = self.get_last_price(self.asset)

        if sentiment == 'positive' and probability > 0.7:
            if self.get_position(self.asset) is None:
                quantity = self.position_sizing()
                order = self.create_order(
                    self.asset,
                    quantity,
                    'buy',
                )
                stop_order = self.create_order(
                    self.asset,
                    quantity,
                    'sell',
                    type='trailing_stop',
                    trail_percent=0.02
                )  
                self.submit_order(order)
                self.submit_order(stop_order)
                self.max_price = current_price
        elif sentiment == 'negative' and probability > 0.7:
            if self.get_position(self.asset):
                self.cancel_open_orders()
                self.sell_all()
                # consider short selling...


broker = Alpaca(ALPACA_CREDS)
strategy = SentimentTrader(name='sentiment_strat', broker=broker, parameters={'symbol': 'AAPL'})

strategy.backtest(
    YahooDataBacktesting,
    datetime(2024, 1, 1),
    datetime(2025, 1, 1),
    parameters={'symbol': 'AAPL'}
)