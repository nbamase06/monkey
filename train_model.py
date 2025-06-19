# train_model.py

import os
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

DATA_DIR = "data"
MODEL_FILE = "forex_model.pkl"

def compute_features(df):
    df = df.copy()
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['macd'] = MACD(close=df['close']).macd_diff()
    df['ema_fast'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
    df['ema_slow'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
    df['ema_cross'] = df['ema_fast'] - df['ema_slow']
    
    # Target: 1 = Buy (price goes up), 0 = Sell (price goes down or flat)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    return df.dropna()

def load_and_prepare_data():
    all_data = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(DATA_DIR, file), parse_dates=['date'])
            df = df.rename(columns=str.lower)
            df = compute_features(df)
            all_data.append(df[['rsi', 'macd', 'ema_cross', 'target']])
    
    return pd.concat(all_data)

def train_and_save_model():
    df = load_and_prepare_data()
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("✅ Model Training Complete")
    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump(model, MODEL_FILE)
    print(f"✅ Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_and_save_model()
