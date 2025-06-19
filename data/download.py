import requests
import pandas as pd

API_KEY = 'LY178EJT8NJRD4IU'
pairs = [('EUR','USD'), ('GBP','USD'), ('USD','JPY'), ('AUD','USD'), ('USD','CHF'), ('USD','CAD')]

for from_symbol, to_symbol in pairs:
    url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_symbol}&to_symbol={to_symbol}&apikey={API_KEY}&outputsize=full'
    data = requests.get(url).json()
    key = "Time Series FX (Daily)"

    if key in data:
        df = pd.DataFrame(data[key]).T
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close'
        }).astype(float)
        df.index.name = 'date'
        df = df.sort_index()
        file_name = f"{from_symbol}{to_symbol}_daily.csv"
        df.to_csv(file_name)
        print(f"Saved: {file_name}")
    else:
        print(f"Error fetching {from_symbol}/{to_symbol}: {data}")
