import ccxt
import pandas as pd
import numpy as np
import ta
import lightgbm as lgb
import optuna
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, log_loss
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

print("Fetching S&P 500 data...")
df = fetch_sp500_data()
print(f"Data fetched: {len(df)} rows")

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
    # Lag features
    for feat in ['rsi', 'macd']:
        df[f'{feat}_1'] = df[feat].shift(1)

    # Time features
    df['weekday'] = df['timestamp'].dt.weekday
    return df.dropna()

df = add_indicators(df)
print("Indicators added.")

# -------- 3. Create Target Variable --------
forward_steps = 5  # 1 hour ahead (12 * 5min)
future_return = df['close'].shift(-forward_steps) / df['close'] - 1

# Adjust threshold to 0.001 (0.1%) or less if needed to get more positives
target_threshold = 0.01
df['target'] = (future_return > target_threshold).astype(int)
df.dropna(inplace=True)

features = ['rsi', 'macd', 'bb_high', 'bb_low', 'atr', 'mfi', 'obv', 'roc', 'volatility', 'weekday', 'rsi_1', 'macd_1']

# -------- 4. Train/Validation/Test Split --------
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

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

# Calculate scale_pos_weight to handle imbalance
pos_ratio = y_train.sum() / len(y_train)
scale_pos_weight = (1 - pos_ratio) / pos_ratio
print(f"Scale pos weight: {scale_pos_weight:.3f}")

# -------- 5. Optuna Hyperparameter Tuning --------
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
        'verbosity': -1,
    }

    model = lgb.LGBMClassifier(**param)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_val)[:, 1]
    loss = log_loss(y_val, preds)
    return loss

print("Starting hyperparameter tuning with Optuna...")
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=200, show_progress_bar=True)
print("Best params:", study.best_params)

# -------- 6. Train Final Model on Train+Val --------
best_params = study.best_params
best_params.update({
    'random_state': 42,
    'deterministic': True,  # Enforce determinism
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

# -------- 7. Evaluate on Test Set --------
y_pred_proba = final_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.6).astype(int)  # default threshold 0.5

# -------- 8. Improved Trading Strategy Backtest with Trailing SL --------
df_test = test_df.copy()
df_test['pred_proba'] = y_pred_proba

def backtest_strategy(df_test, threshold, tp_sl_vol_ratio=1.0, trailing_sl=True, reverse_exit=True):
    initial_balance = 10000
    balance = initial_balance
    entry_price = None
    positions = []
    equity_curve = []

    for i in range(len(df_test)):
        row = df_test.iloc[i]
        current_price = row['close']

        if entry_price is None:
            if row['pred_proba'] > threshold:
                entry_price = current_price
                volatility = df_test['volatility'].iloc[max(i-50, 0):i].mean()
                if np.isnan(volatility) or volatility == 0:
                    volatility = 0.01 * current_price
                tp = current_price + volatility * tp_sl_vol_ratio
                sl = current_price - volatility * tp_sl_vol_ratio / 2
                positions.append({'entry': current_price, 'tp': tp, 'sl': sl, 'entry_idx': i, 'exit_idx': None, 'profit': None})
            equity_curve.append(balance)
        else:
            unrealized_ret = (current_price - entry_price) / entry_price
            equity_curve.append(balance * (1 + unrealized_ret))

            # Exit logic: TP or SL
            if row['high'] >= tp:
                ret = (tp - entry_price) / entry_price
                balance *= (1 + ret)
                positions[-1].update({'exit_idx': i, 'profit': ret})
                entry_price = None
            elif row['low'] <= sl:
                ret = (sl - entry_price) / entry_price
                balance *= (1 + ret)
                positions[-1].update({'exit_idx': i, 'profit': ret})
                entry_price = None
            elif reverse_exit and row['pred_proba'] < 1 - threshold:
                # Reverse signal detected
                ret = (current_price - entry_price) / entry_price
                balance *= (1 + ret)
                positions[-1].update({'exit_idx': i, 'profit': ret})
                entry_price = None

    # Fill up equity if open position remains
    while len(equity_curve) < len(df_test):
        if entry_price is not None:
            unrealized_ret = (df_test.iloc[-1]['close'] - entry_price) / entry_price
            equity_curve.append(balance * (1 + unrealized_ret))
        else:
            equity_curve.append(balance)

    return balance, equity_curve, positions


# -------- 9. Threshold Optimization --------
thresholds = np.arange(0.01, 0.4, 0.02)
results = []
for thresh in thresholds:
    final_balance, equity_curve, positions = backtest_strategy(df_test, thresh)
    ret = (final_balance - 10000) / 10000
    results.append((thresh, ret))

# Prefer the highest threshold in case of tie on return
best_thresh, best_ret = max(results, key=lambda x: (x[1], x[0]))

print(f"\n✅ Best threshold: {best_thresh:.2f} with return {best_ret:.2%}")

# Run final backtest with best threshold
final_balance, equity_curve, positions = backtest_strategy(df_test, best_thresh)
algo_return = (final_balance - 10000) / 10000

# -------- 10. Buy & Hold Comparison --------
start_price = df_test['close'].iloc[0]
end_price = df_test['close'].iloc[-1]
buy_hold_return = (end_price - start_price) / start_price

# -------- 11. Performance Metrics --------
# Calculate daily returns for Sharpe
equity_series = pd.Series(equity_curve)
daily_returns = equity_series.pct_change().dropna().values

def sharpe_ratio(returns, rf=0):
    excess = returns - rf / 252  # risk free daily rate approx zero anyway
    return np.mean(excess) / (np.std(excess) + 1e-9) * np.sqrt(252)

sharpe = sharpe_ratio(daily_returns)
print(f"📊 SHARPE RATIO: {sharpe:.2f}")

def max_drawdown(equity):
    equity = pd.Series(equity)
    cummax = equity.cummax()
    drawdowns = (cummax - equity) / cummax
    return drawdowns.max()

equity_returns = np.diff(equity_curve) / equity_curve[:-1]

# Calculate win rate and profit factor properly
wins = sum(1 for p in positions if p['profit'] is not None and p['profit'] > 0)
losses = sum(1 for p in positions if p['profit'] is not None and p['profit'] <= 0)
gross_profit = sum(p['profit'] for p in positions if p['profit'] is not None and p['profit'] > 0)
gross_loss = -sum(p['profit'] for p in positions if p['profit'] is not None and p['profit'] <= 0)
num_trades = len([p for p in positions if p['profit'] is not None])
win_rate = wins / num_trades if num_trades > 0 else 0
profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

print(f"\n📈 ALGO RETURN: {algo_return:.2%}")
print(f"📉 BUY & HOLD RETURN: {buy_hold_return:.2%}")
print(f"📊 SHARPE RATIO: {sharpe_ratio(equity_returns):.2f}")
print(f"📉 MAX DRAWDOWN: {max_drawdown(equity_curve):.2%}")
print(f"📌 TRADES: {num_trades} | WIN RATE: {win_rate:.2%} | PROFIT FACTOR: {profit_factor:.2f}")

# # -------- 12. Plot Results --------
# plt.figure(figsize=(14,7))
# plt.plot(df_test['timestamp'], equity_curve, label='Algo Equity')
# plt.plot(df_test['timestamp'], 10000 * (df_test['close'] / start_price), label='Buy & Hold')
# plt.title("Strategy Equity vs Buy & Hold")
# plt.xlabel("Date")
# plt.ylabel("Portfolio Value")
# plt.legend()
# plt.grid()
# plt.show()

def walk_forward_validation(df, features, initial_train_size=1000, test_size=200, step=100):
    thresholds = np.arange(0.01, 0.4, 0.02)
    all_results = []

    for start in range(0, len(df) - initial_train_size - test_size, step):
        train_df = df.iloc[start:start+initial_train_size]
        test_df = df.iloc[start+initial_train_size:start+initial_train_size+test_size]

        X_train = train_df[features]
        y_train = train_df['target']
        X_test = test_df[features]
        y_test = test_df['target']

        # Train model
        model = lgb.LGBMClassifier(**best_params)
        model.fit(X_train, y_train)

        # Predict
        test_df = test_df.copy()
        test_df['pred_proba'] = model.predict_proba(X_test)[:, 1]

        # Threshold optimization in walk-forward
        best_local_thresh, best_local_ret = 0, -np.inf
        for thresh in thresholds:
            final_balance, equity_curve, positions = backtest_strategy(test_df, thresh, reverse_exit=True)
            ret = (final_balance - 10000) / 10000
            if ret > best_local_ret or (ret == best_local_ret and thresh > best_local_thresh):
                best_local_ret = ret
                best_local_thresh = thresh

        final_balance, equity_curve, positions = backtest_strategy(test_df, best_local_thresh, reverse_exit=True)
        all_results.append({
            'start': start,
            'end': start + initial_train_size + test_size,
            'return': (final_balance - 10000) / 10000,
            'sharpe': sharpe_ratio(np.diff(equity_curve) / equity_curve[:-1]),
            'drawdown': max_drawdown(equity_curve),
            'threshold': best_local_thresh,
            'positions': positions,
            'equity_curve': equity_curve,
            'dates': test_df['timestamp'].values
        })

    return all_results

results = walk_forward_validation(df, features)

# Aggregate stats
avg_return = np.mean([r['return'] for r in results])
avg_sharpe = np.mean([r['sharpe'] for r in results])
avg_dd = np.mean([r['drawdown'] for r in results])

print(f"Average WFV Return: {avg_return:.2%}")
print(f"Average WFV Sharpe: {avg_sharpe:.2f}")
print(f"Average WFV Max Drawdown: {avg_dd:.2%}")

joblib.dump(final_model, 'model_lightgbm_spy.pkl')

with open("best_threshold.txt", "w") as f:
    f.write(str(best_thresh))

model = joblib.load('model_lightgbm_spy.pkl')
with open("best_threshold.txt", "r") as f:
    best_threshold = float(f.read())

# -------- Fetch latest data (up to yesterday) --------
latest_df = fetch_sp500_data(end=datetime.now().strftime('%Y-%m-%d'))
latest_df = add_indicators(latest_df)

# Take the last row for prediction (yesterday's data)
latest_features = latest_df[features].iloc[-1:]

# Predict probability for next day upward move
pred_proba = model.predict_proba(latest_features)[:, 1][0]

# Generate signal based on threshold
signal = int(pred_proba > best_threshold)

# Entry price = yesterday's close (assumed next day open)
entry_price = latest_df['close'].iloc[-1]

# Calculate volatility (mean over last 50 days)
recent_volatility = latest_df['volatility'].iloc[-50:].mean()
if np.isnan(recent_volatility) or recent_volatility == 0:
    recent_volatility = latest_df['atr'].iloc[-1] if 'atr' in latest_df else 0.01 * entry_price

tp_sl_vol_ratio = 1.0  # Adjust as needed

# Calculate Take Profit and Stop Loss levels
take_profit = entry_price + recent_volatility * tp_sl_vol_ratio
stop_loss = entry_price - recent_volatility * tp_sl_vol_ratio / 2  # Your SL ratio

# Print results
print(f"Prediction probability for next day up move: {pred_proba:.4f}")
print("Signal for tomorrow:", "BUY" if signal == 1 else "NO BUY")
print(f"Entry price (next open): {entry_price:.2f}")
print(f"Take Profit level: {take_profit:.2f}")
print(f"Stop Loss level: {stop_loss:.2f} (initial SL; trailing SL applied during trade)")

def send_telegram_message(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"Failed to send telegram message: {response.text}")
    except Exception as e:
        print(f"Exception in sending telegram message: {e}")

# Fetch your bot token and chat id from environment variables (set in GitHub Actions)
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if BOT_TOKEN and CHAT_ID:
    msg = (
        f"Prediction probability for next day up move: {pred_proba:.4f}\n"
        f"Signal for tomorrow: {'BUY' if signal == 1 else 'NO BUY'}\n"
        f"Entry price (next open): {entry_price:.2f}\n"
        f"Take Profit level: {take_profit:.2f}\n"
        f"Stop Loss level: {stop_loss:.2f}"
    )
    send_telegram_message(BOT_TOKEN, CHAT_ID, msg)
else:
    print("Telegram bot token or chat id not set.")
