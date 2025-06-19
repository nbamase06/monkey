from flask import Flask, render_template, abort
import requests
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
import joblib
from datetime import datetime

app = Flask(__name__)
API_KEY = 'LY178EJT8NJRD4IU'
RSI_PERIOD = 14

# Load AI model
model = joblib.load("forex_model.pkl")

pairs = [
    ('EUR', 'USD'),
    ('GBP', 'USD'),
    ('USD', 'JPY'),
    ('AUD', 'USD'),
    ('USD', 'CHF'),
    ('USD', 'CAD')
]

def fetch_signals(from_symbol, to_symbol):
    url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_symbol}&to_symbol={to_symbol}&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()

    key = "Time Series FX (Daily)"
    if key not in data:
        return None

    df = pd.DataFrame(data[key]).T.astype(float).sort_index()
    df.columns = ['open', 'high', 'low', 'close']
    df.index = pd.to_datetime(df.index)
    
    latest_date = df.index[-1]
    entry_price = df['close'].iloc[-1]
    
    rsi = RSIIndicator(close=df['close'], window=RSI_PERIOD).rsi().iloc[-1]
    macd_diff = MACD(close=df['close']).macd_diff().iloc[-1]
    ema_fast = EMAIndicator(close=df['close'], window=12).ema_indicator().iloc[-1]
    ema_slow = EMAIndicator(close=df['close'], window=26).ema_indicator().iloc[-1]
    ema_cross = ema_fast - ema_slow

    features = pd.DataFrame({
        'rsi': [rsi],
        'macd': [macd_diff],
        'ema_cross': [ema_cross]
    })

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = max(proba)

    signal = "BUY 📈" if prediction == 1 else "SELL 📉"

    # SL/TP logic
    SL_PERCENT = 0.01
    TP_PERCENT = 0.02
    if signal.startswith("BUY"):
        sl = entry_price * (1 - SL_PERCENT)
        tp = entry_price * (1 + TP_PERCENT)
    else:
        sl = entry_price * (1 + SL_PERCENT)
        tp = entry_price * (1 - TP_PERCENT)

    return {
        "pair": f"{from_symbol}/{to_symbol}",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "datetime": latest_date.strftime("%Y-%m-%d %H:%M"),
        "entry_price": round(entry_price, 5),
        "rsi": round(rsi, 2),
        "macd_diff": round(macd_diff, 5),
        "ema_fast": round(ema_fast, 5),
        "ema_slow": round(ema_slow, 5),
        "ai_signal": signal,
        "confidence": f"{confidence * 100:.2f}%",
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "is_open": True
    }

@app.route('/')
def index():
    results = []
    for from_symbol, to_symbol in pairs:
        signal = fetch_signals(from_symbol, to_symbol)
        if signal:
            results.append(signal)
    return render_template("index.html", results=results)

@app.route('/pair/<from_symbol>/<to_symbol>')
def pair_detail(from_symbol, to_symbol):
    match = next(((fs, ts) for fs, ts in pairs if fs == from_symbol and ts == to_symbol), None)
    if not match:
        abort(404)

    detail = fetch_signals(from_symbol, to_symbol)
    if not detail:
        abort(500)

    return render_template("detail.html", data=detail)

if __name__ == '__main__':
    app.run(debug=True)
