import json
from gymnasium import Env, spaces
import numpy as np
import yfinance as yf
import math


class TradeEnv(Env):
    """
    Observation Space:
        - Current cash
        - Current position (quantity)
        - Closing stock price of last day
        - Difference between recent closing price and EMA12
        - Difference between recent closing price and EMA26
        - Current MACD historgram
        - 2 days ago MACD historgram
        - 14 day volume ROC
        - On-balance volume
        - Price diff from Bollinger upper bound
        - Price diff from Bollinger lower bound
        - Bollinger bandwidth
        - Average sentiment score
        
    start and end must be in a YYYY-MM-DD format
    """

    ACTION_SPACE_SIZE = 3

    def __init__(self, symbol: str, start: str, end: str, risk_per_trade: float, sentiments_path: str, eval: bool = False):
        super().__init__()

        self._cash = 5000
        self._starting_portfolio = self._cash
        self._position = 0

        self.action_space = spaces.Discrete(TradeEnv.ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, -1], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1], dtype=np.float32),
            shape=(13,)
        )

        self._step_count = 0
        self.symbol = symbol
        self._risk_per_trade = risk_per_trade
        self._eval = eval

        self._init_market_data(start, end)

        with open(sentiments_path, 'r') as file:
            self._sentiments = json.load(file)


    def step(self, action):
        terminated = False
        truncated = False
        info = {
            'step': self._step_count,
            'date': self._get_date(),
            'cash': self._cash,
            'position': self._position,
            'action': ['buy', 'sell', 'hold'][action],
            'position_sizing': self._get_position_sizing() if action == 0 else None
        }

        if action not in {0, 1, 2}:
            raise ValueError(f"Received invalid action={action} which is not part of the action space")

        self._update_attributes(action)

        prev_price = self._get_market_data('Close')
        self._step_count += 1
        price_change = self._get_market_data('Close') - prev_price

        reward = self._get_reward(action, price_change)
        info['reward'] = reward

        if self._step_count == len(self._market_data) - 1:
            terminated = True
        
        next_obs = self._get_obs()

        return next_obs, reward, terminated, truncated, info

        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._eval:
            self._cash = 5000
            self._position = 0
        else:
            self._cash = self.np_random.uniform(1000, 10000)
            self._position = self.np_random.integers(0, 10)

        self._starting_portfolio = self._cash + self._position * self._market_data.iloc[0]['Close'].iat[0]
        self._step_count = 0

        return self._get_obs(), {}


    def render(self):
        pass


    def close(self):
        pass


    def _update_attributes(self, action: int) -> None:
        if action == 0 and self._cash > self._get_market_data('Close'):  # buy
            quantity = self._get_position_sizing()
            self._cash -= quantity * self._get_market_data('Close')
            self._position += quantity
        elif action == 1 and self._position >= 1:  # sell
            self._cash += self._position * self._get_market_data('Close')
            self._position = 0
        elif action == 2:  # hold
            pass


    def _get_reward(self, action: int, price_change: float) -> int:
        """
        Computes the reward after taking action in the previous (self._step_count - 1) step and experiencing price_change
        """
        reward = 0
        if action == 0 and self._cash > self._get_market_data('Close'):
            if price_change >= 0:
                reward = 1
            else:
                reward = -1
        elif action == 1 and self._position >= 1:
            if price_change > 0:
                reward = -1
            else:
                reward = 1
        else:
            reward = -0.1

        if self._cash <= self._get_market_data('Close') and self._position >= 1:
            reward = -10
        elif self._cash <= self._get_market_data('Close') and self._position == 0:
            reward = -100

        if self._step_count == len(self._market_data) - 1:
            reward = (self._cash + self._position * self._get_market_data('Close')) - self._starting_portfolio

        return reward


    def _get_obs(self) -> tuple:
        return np.array([
            self._cash,
            self._position,
            self._get_market_data('Close'),
            self._get_market_data('Diff_EMA12'),
            self._get_market_data('Diff_EMA26'),
            self._get_market_data('MACD_Histogram_t'),
            self._get_market_data('MACD_Histogram_t-2'),
            self._get_market_data('Volume_ROC_14d'),
            self._get_market_data('OBV'),
            self._get_market_data('Diff_Bollinger_Upper'),
            self._get_market_data('Diff_Bollinger_Lower'),
            self._get_market_data('Bollinger_Bandwidth'),
            self._get_sentiment_score()
        ], dtype=np.float32)


    def _get_date(self) -> str:
        return self._market_data.index[self._step_count].to_pydatetime().strftime('%Y-%m-%d')


    def _get_market_data(self, col_name: str):
        return self._market_data.iloc[self._step_count][col_name].iat[0]
    

    def _get_sentiment_score(self) -> float:
        return self._sentiments[self._get_date()] if self._get_date() in self._sentiments else 0


    def _get_position_sizing(self) -> int:
        """
        Assumes self._cash > price per unit of asset.
        Otherwise, this function will return an unaffordable quantity.
        """
        quantity = math.ceil(self._cash * self._risk_per_trade / self._get_market_data('Close'))
        return min(quantity, self._cash // self._get_market_data('Close'))
    

    def _init_market_data(self, start: str, end: str) -> None:
        data = yf.download(self.symbol, start=start, end=end)
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['Diff_EMA12'] = data['Close'].squeeze() - data['EMA12']
        data['Diff_EMA26'] = data['Close'].squeeze() - data['EMA26']
        data['MACD_Line'] = data['EMA12'] - data['EMA26']
        data['Signal_Line'] = data['MACD_Line'].ewm(span=9, adjust=False).mean()
        data['MACD_Histogram_t'] = data['MACD_Line'] - data['Signal_Line']
        data['MACD_Histogram_t-2'] = data['MACD_Histogram_t'].shift(2)
        data['Volume_ROC_14d'] = (data['Volume'] - data['Volume'].shift(14)) / data['Volume'].shift(14) * 100
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['Close_STD_20d'] = data['Close'].rolling(window=20).std()
        data['Bollinger_Upper'] = data['SMA20'] + 2 * data['Close_STD_20d']
        data['Bollinger_Lower'] = data['SMA20'] - 2 * data['Close_STD_20d']
        data['Diff_Bollinger_Upper'] = data['Close'].squeeze() - data['Bollinger_Upper']
        data['Diff_Bollinger_Lower'] = data['Close'].squeeze() - data['Bollinger_Lower']
        data['Bollinger_Bandwidth'] = np.where(
            data['SMA20'] > 0,
            (data['Bollinger_Upper'] - data['Bollinger_Lower']) / data['SMA20'],
            0
        )
        data['OBV'] = np.where(
            data['Close'] > data['Close'].shift(1), data['Volume'],
            np.where(
                data['Close'] < data['Close'].shift(1), -data['Volume'], 0
            )
        ).cumsum()
        
        data.drop(columns=[
            'High',
            'Low',
            'Open',
            'Volume',
            'EMA12',
            'EMA26',
            'MACD_Line',
            'Signal_Line',
            'SMA20',
            'Close_STD_20d',
            'Bollinger_Upper',
            'Bollinger_Lower'
        ], inplace=True)
        self._market_data = data.iloc[38:]

