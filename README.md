# Sentiment-Based Trading Bot
This project implements a sentiment-based trading bot using the Alpaca API for trading and the FinBERT model for sentiment analysis. The bot trades based on the sentiment of recent news headlines related to a specified stock symbol.

## Table of Contents
- **[Features](#features)**
- **[Installation](#installation)**
- **[Usage](#usage)**
- **[Files](#files)**
- **[License](#license)**

## Features
- **Alpaca Integration:** Uses the Alpaca API to place buy/sell orders and manage positions.
- **Sentiment Analysis:** Leverages the FinBERT model (a pre-trained BERT model) to predict the sentiment of financial news headlines.
- **Backtesting:** Includes the ability to backtest the strategy with historical data using Yahoo Finance.
- **Risk Management:** Implements stop-loss and take-profit mechanisms for automated trade execution.

## Installation
1. Clone the repository:

``` bash
git clone https://github.com/yourusername/sentiment-trading-bot.git
cd sentiment-trading-bot
```

2. Install the required packages:

``` bash
pip install -r requirements.txt
```

3. Set up environment variables: Create a .env file in the root directory and add your Alpaca API credentials:

```
API_KEY=your_alpaca_api_key
API_SECRET=your_alpaca_api_secret
BASE_URL=https://paper-api.alpaca.markets
```

## Usage
1. Run the trading bot:
``` bash
python bot.py
```
2. Backtest the strategy: The backtesting is already included in the bot.py script. It uses Yahoo Finance data to backtest the strategy from January 1, 2020, to January 1, 2025.

## Files
- **bot.py**: Contains the main trading bot implementation.
- **sentiment.py**: Contains the sentiment analysis function using the FinBERT model.
  
## License
This project is licensed under the MIT License. See the LICENSE file for details.
