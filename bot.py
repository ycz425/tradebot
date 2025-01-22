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
        self.recent_sentiments = ['neutral', 'neutral']


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
        self.recent_sentiments.append(sentiment)
        self.recent_sentiments.pop(0)
        print(f'\n{self.get_datetime().strftime('%Y-%m-%d')}: {sentiment}')
        current_price = self.get_last_price(self.asset)

        if all(sentiment == 'positive' for sentiment in self.recent_sentiments):
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
                    
                )  
                self.submit_order(order)
                self.submit_order(stop_order)
                self.max_price = current_price
        elif all(sentiment == 'negative' for sentiment in self.recent_sentiments):
            if self.get_position(self.asset):
                self.cancel_open_orders()
                self.sell_all()
                # consider short selling...


broker = Alpaca(ALPACA_CREDS)
strategy = SentimentTrader(name='sentiment_strat', broker=broker, parameters={'symbol': 'AAPL'})

strategy.backtest(
    YahooDataBacktesting,
    datetime(2024, 6, 1),
    datetime(2025, 1, 1),
    parameters={'symbol': 'AAPL'},
    benchmark_asset='AAPL'
)