import ccxt
import pandas as pd
import numpy as np
import ta
import lightgbm as lgb
import optuna
from datetime import datetime
from sklearn.metrics import log_loss
import joblib
import yfinance as yf
import requests
import os

# -------- 1. Fetch OHLCV from Yahoo Finance --------
def fetch_sp500_data(start='2015-01-01', end=None):
    df = yf.download('^GSPC', start=start, end=end, interval='1d')
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df.rename(columns={'Date': 'timestamp', 'Open': 'open', 'High': 'high',
                            'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return df

df = fetch_sp500_data()

# -------- 2. Add Technical Indicators --------
def add_indicators(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd_diff()
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['roc'] = ta.momentum.ROCIndicator(df['close']).roc()
    df['volatility'] = df['close'].rolling(50).std()
    for feat in ['rsi', 'macd']:
        df[f'{feat}_1'] = df[feat].shift(1)
    df['weekday'] = df['timestamp'].dt.weekday
    return df.dropna()

df = add_indicators(df)

# -------- 3. Create Target Variable --------
forward_steps = 5
future_return = df['close'].shift(-forward_steps) / df['close'] - 1
target_threshold = 0.01
df['target'] = (future_return > target_threshold).astype(int)
df.dropna(inplace=True)

features = ['rsi', 'macd', 'bb_high', 'bb_low', 'atr', 'mfi', 'obv', 'roc', 'volatility',
            'weekday', 'rsi_1', 'macd_1']

split_1 = int(len(df)*0.7)
split_2 = int(len(df)*0.85)

train_df = df.iloc[:split_1]
val_df = df.iloc[split_1:split_2]
test_df = df.iloc[split_2:]

X_train = train_df[features]
y_train = train_df['target']
X_val = val_df[features]
y_val = val_df['target']
X_test = test_df[features]
y_test = test_df['target']

pos_ratio = y_train.sum() / len(y_train)
scale_pos_weight = (1 - pos_ratio) / pos_ratio

# -------- 4. Optuna Hyperparameter Tuning --------
def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'scale_pos_weight': scale_pos_weight,
        'n_estimators': 100,
    }
    model = lgb.LGBMClassifier(**param)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    loss = log_loss(y_val, preds)
    return loss

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=200, show_progress_bar=False)

best_params = study.best_params
best_params.update({
    'random_state': 42,
    'deterministic': True,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'scale_pos_weight': scale_pos_weight
})

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X_train_val, y_train_val)

joblib.dump(final_model, 'model_lightgbm_spy.pkl')
model = joblib.load('model_lightgbm_spy.pkl')

# -------- Latest Data --------
latest_df = fetch_sp500_data(end=datetime.now().strftime('%Y-%m-%d'))
latest_df = add_indicators(latest_df)

# -------- Categorize Returns --------
def categorize_return_probs(latest_df, model, features):
    latest_features = latest_df[features].iloc[-1:]
    entry_price = latest_df['close'].iloc[-1]
    pred_proba = model.predict_proba(latest_features)[:, 1][0]

    recent_volatility = latest_df['volatility'].iloc[-50:].mean()
    if np.isnan(recent_volatility) or recent_volatility == 0:
        recent_volatility = latest_df['atr'].iloc[-1] if 'atr' in latest_df else 0.01 * entry_price

    n_samples = 5000
    mu = pred_proba * recent_volatility
    sigma = recent_volatility
    simulated_returns = np.random.normal(mu / entry_price, sigma / entry_price, n_samples)

    bins = [-10, -3, -2, -1, 0, 1, 2, 3, 10]
    categories = [
        "≤ -3%", "-3% to -2%", "-2% to -1%", "-1% to 0%",
        "0% to 1%", "1% to 2%", "2% to 3%", "≥ 3%"
    ]

    hist, _ = np.histogram(simulated_returns * 100, bins=bins)
    probs = hist / hist.sum()
    return dict(zip(categories, probs))

category_probs = categorize_return_probs(latest_df, model, features)

# -------- Telegram --------
def send_telegram_message(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception:
        pass

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if BOT_TOKEN and CHAT_ID:
    msg_lines = ["📊 Probability distribution for next day move:"]
    for cat, p in category_probs.items():
        msg_lines.append(f"{cat}: {p:.2%}")
    msg = "\n".join(msg_lines)
    send_telegram_message(BOT_TOKEN, CHAT_ID, msg)
