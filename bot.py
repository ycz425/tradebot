from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
from alpaca_trade_api import REST
from timedelta import Timedelta


API_KEY = "PKOJ3Y5QN1Q36MP8YX6H"
API_SECRET = "5EoK9EyURnP3DplRV8GFQGqhSJ0QcQ85DhcsKW7T"
BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {
    'API_KEY': API_KEY,
    'API_SECRET': API_SECRET,
    'PAPER': True
}


class Trader(Strategy):
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

    def get_news(self):
        news_list = self.api.get_news(
            symbol=self.symbol,
            start=(self.get_datetime() - Timedelta(days=3)).strftime('%Y-%m-%d'),
            end=self.get_datetime().strftime('%Y-%m-%d')
        )
        return [news.headline for news in news_list]

    def on_trading_iteration(self):
        if self.get_position(self.symbol) is None:
            self.last_trade = None

        
        cash, last_price, quantity = self.position_sizing()
        if cash > last_price and self.last_trade is None:
            print(self.get_news())
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


broker = Alpaca(ALPACA_CREDS)
strategy = Trader(name='strat', broker=broker, parameters={'symbol': 'SPY', 'cash_at_risk': 0.5})

strategy.backtest(
    YahooDataBacktesting,
    datetime(2024, 12, 15),
    datetime(2024, 12, 31),
    parameters={'symbol': 'SPY', 'cash_at_risk': 0.5}
)