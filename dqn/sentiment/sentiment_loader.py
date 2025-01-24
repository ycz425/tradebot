import requests
from dotenv import load_dotenv
import json
import os

load_dotenv()
EODHD_TOKEN = os.getenv('EODHD_TOKEN')

def fetch_sentiments(ticker: str, start: str, end: str) -> list[dict]:
    ENDPOINT = 'https://eodhd.com/api/sentiments'
    response = requests.get(ENDPOINT, params={'s': ticker, 'from': start, 'to': end, 'api_token': EODHD_TOKEN})

    if response.status_code == 200:
        return response.json()[ticker]
    else:
        raise requests.exceptions.HTTPError(f"HTTP ERROR {response.status_code}: {response.text}")
    

def save_sentiments(sentiments: list[dict], save_path: str) -> None:
    result = {}
    for sentiment in sentiments:
        result[sentiment['date']] = sentiment['normalized']

    with open(os.path.join(save_path if save_path.endswith('.json') else save_path + '.json'), 'w') as file:
        json.dump(result, file, indent=4)


if __name__ == '__main__':
    sentiments = fetch_sentiments('MSFT.US', '2019-01-01', '2025-01-20')
    save_sentiments(sentiments, 'dqn/data/MSFT.US_2019-01-01_to_2025-01-20.json')

