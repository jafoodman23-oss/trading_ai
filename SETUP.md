# Trading AI — Setup Guide

## Step 1: Install Python
Download Python 3.11 from https://python.org/downloads
- CHECK "Add Python to PATH" during install
- Restart your terminal after installing

## Step 2: Install dependencies
Open a terminal in the trading_ai folder and run:

```
pip install -r requirements.txt
```

## Step 3: Get free Alpaca API keys (paper trading)
1. Go to https://alpaca.markets
2. Create a free account
3. Go to Paper Trading > API Keys
4. Generate keys — copy them

## Step 4: Add your API keys
Edit config/config.yaml and replace:
  paper_api_key: "YOUR_PAPER_API_KEY"
  paper_api_secret: "YOUR_PAPER_API_SECRET"

## Step 5: Train the AI
```
python main.py train --symbols SPY AAPL NVDA QQQ --days 365
```
This fetches 1 year of historical data and trains the PPO model.
Takes 10-30 minutes depending on your hardware.

## Step 6: Start paper trading
```
python main.py run --mode paper
```
The AI will now trade in real time using the paper account.
It also continuously retrains itself as new bars arrive.

## Step 7: Run a backtest (optional)
```
python main.py backtest --start 2024-01-01 --end 2025-01-01 --symbols SPY
```

## Switch to LIVE trading (when ready)
1. Get Alpaca live API keys (requires funded account)
2. Edit config/config.yaml:
   - Set mode: "live"
   - Fill in live_api_key and live_api_secret
3. Run: python main.py run

## Architecture Overview

```
main.py           <- CLI entry point (train / run / backtest / dashboard)
config/           <- Settings loaded from config.yaml
data/
  collector.py    <- Real-time WebSocket stream from Alpaca
  historical.py   <- Fetch historical bars via REST API
  preprocessor.py <- 25+ technical indicators as ML features
  storage.py      <- SQLite database for all data
environment/
  trading_env.py  <- Gymnasium RL environment (gym-compatible)
models/
  rl_agent.py     <- PPO reinforcement learning agent (self-trains)
  pattern_detector.py <- LSTM pattern recognition model
  ensemble.py     <- Combines RL + LSTM signals
trading/
  paper_broker.py <- Internal paper trading simulator (no API needed)
  alpaca_broker.py <- Alpaca paper/live broker
  risk_manager.py <- Kelly sizing, stop losses, drawdown halts
training/
  trainer.py      <- Orchestrates historical pre-train + continuous learning
  evaluator.py    <- Backtesting and performance metrics
monitoring/
  dashboard.py    <- Live Rich terminal dashboard
  metrics.py      <- Sharpe, Sortino, drawdown, win rate, etc.
```

## How it self-learns

1. Pre-trains on 1 year of historical data using PPO (Proximal Policy Optimization)
2. While trading live, buffers new bars
3. Every 500 bars, evaluates its own Sharpe ratio
4. If performance drops below threshold, automatically retrains on recent data
5. Saves new checkpoint only if performance improved
6. Over time, the model sees more market conditions and adapts
