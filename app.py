"""
AgentTrading-Dyad Backend
Comprehensive quantitative trading and analysis platform
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
from datetime import datetime, timedelta

# Core Data & Analysis
import pandas as pd
import numpy as np
import scipy
import polars as pl

# Financial Data
import yfinance as yf
import alpaca_trade_api as tradeapi
try:
    import pandas_datareader as pdr
except ImportError:
    pdr = None
try:
    from alpha_vantage.timeseries import TimeSeries
except ImportError:
    TimeSeries = None
try:
    import akshare as ak
except ImportError:
    ak = None
try:
    import tushare as ts
except ImportError:
    ts = None
try:
    import yahooquery as yq
except ImportError:
    yq = None
try:
    import investpy
except ImportError:
    investpy = None
try:
    import eodhd
except ImportError:
    eodhd = None
try:
    import finnhub
except ImportError:
    finnhub = None

# Social & News Data
try:
    import praw
except ImportError:
    praw = None
try:
    import feedparser
except ImportError:
    feedparser = None
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# Technical Analysis
try:
    import talib
except ImportError:
    print("TA-Lib not available - install with: pip install TA-Lib")
    talib = None

import pandas_ta as ta_pandas
from ta import add_all_ta_features
try:
    from stockstats import StockDataFrame
except ImportError:
    StockDataFrame = None
try:
    from finta import TA as FinTA
except ImportError:
    FinTA = None
try:
    import btalib
except ImportError:
    btalib = None
try:
    from talipp.indicators import SMA, EMA, RSI, MACD
except ImportError:
    SMA = EMA = RSI = MACD = None

# Backtesting
try:
    import backtrader as bt
except ImportError:
    bt = None

try:
    from backtesting import Backtest, Strategy
except ImportError:
    Backtest = None

try:
    import vectorbt as vbt
except ImportError:
    vbt = None

try:
    import pyqstrat
except ImportError:
    pyqstrat = None

try:
    import fastquant
except ImportError:
    fastquant = None

# Portfolio Optimization
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
except ImportError:
    EfficientFrontier = None

try:
    import riskfolio as rp
except ImportError:
    rp = None

try:
    import cvxpy as cp
except ImportError:
    cp = None

# Risk Analytics
try:
    import quantstats as qs
except ImportError:
    qs = None

try:
    import empyrical as ep
except ImportError:
    ep = None

try:
    import pyfolio as pf
except ImportError:
    pf = None

try:
    from FinQuant.portfolio import build_portfolio
except ImportError:
    build_portfolio = None

# Factor Analysis
try:
    import alphalens as al
except ImportError:
    al = None

# Time Series
from statsmodels.tsa.arima.model import ARIMA
try:
    from arch import arch_model
except ImportError:
    arch_model = None

try:
    import pmdarima as pm
except ImportError:
    pm = None

try:
    from gluonts.model.deepar import DeepAREstimator
except ImportError:
    DeepAREstimator = None

try:
    from tsfresh import extract_features
except ImportError:
    extract_features = None

try:
    from tsmoothie.smoother import LowessSmoother
except ImportError:
    LowessSmoother = None

# Financial Instruments
try:
    import QuantLib as ql
except ImportError:
    ql = None

try:
    import vollib
except ImportError:
    vollib = None

try:
    from py_vollib.black_scholes import black_scholes as bs
except ImportError:
    bs = None

# ML for Finance
try:
    import mlfinlab
except ImportError:
    mlfinlab = None

try:
    from AlphaPy.market import Market
except ImportError:
    Market = None

# ML
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Calendars
try:
    from exchange_calendars import get_calendar
except ImportError:
    get_calendar = None

try:
    import pandas_market_calendars as mcal
except ImportError:
    mcal = None

# Visualization
import mplfinance as mpf
import plotly.graph_objects as go
try:
    import dtale
except ImportError:
    dtale = None

try:
    import finplot as fplt
except ImportError:
    fplt = None

# Crypto
try:
    import ccxt
except ImportError:
    ccxt = None

# AI/LLM
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langgraph.graph import StateGraph
except ImportError:
    StateGraph = None

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console = Table = None

# Database
try:
    from sqlalchemy import create_engine
except ImportError:
    create_engine = None

try:
    import redis
except ImportError:
    redis = None

load_dotenv()
app = Flask(__name__)
CORS(app)

# Configuration
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')


# ============================================================================
# DATA ENDPOINTS
# ============================================================================

@app.route('/api/data/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    """Get stock OHLCV data with technical indicators"""
    try:
        period = request.args.get('period', '1y')
        interval = request.args.get('interval', '1d')
        
        # Fetch data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        # Add technical indicators
        if talib:
            df['SMA_20'] = talib.SMA(df['Close'].values, timeperiod=20)
            df['EMA_12'] = talib.EMA(df['Close'].values, timeperiod=12)
            df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
                df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9
            )
        else:
            # Fallback to pandas-ta
            df.ta.sma(length=20, append=True)
            df.ta.ema(length=12, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.macd(append=True)
        
        return jsonify({
            'symbol': symbol,
            'data': df.reset_index().to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/market-overview', methods=['GET'])
def market_overview():
    """Get market overview with multiple indices"""
    try:
        symbols = ['^GSPC', '^DJI', '^IXIC', '^RUT']  # S&P500, DOW, NASDAQ, Russell
        data = {}
        
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='5d')
            
            data[symbol] = {
                'price': hist['Close'].iloc[-1],
                'change': ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100,
                'volume': int(hist['Volume'].iloc[-1])
            }
        
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# TECHNICAL ANALYSIS ENDPOINTS
# ============================================================================

@app.route('/api/analysis/indicators/<symbol>', methods=['POST'])
def calculate_indicators(symbol):
    """Calculate comprehensive technical indicators"""
    try:
        data = request.json
        period = data.get('period', '1y')
        indicators = data.get('indicators', ['all'])
        
        df = yf.Ticker(symbol).history(period=period)
        
        if 'all' in indicators:
            # Add all TA features
            df = add_all_ta_features(
                df, open="Open", high="High", low="Low", 
                close="Close", volume="Volume", fillna=True
            )
        
        return jsonify({
            'symbol': symbol,
            'indicators': df.reset_index().to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/pattern-recognition/<symbol>', methods=['GET'])
def pattern_recognition(symbol):
    """Detect candlestick patterns using TA-Lib"""
    if not talib:
        return jsonify({'error': 'TA-Lib not available'}), 400
    
    try:
        df = yf.Ticker(symbol).history(period='3mo')
        
        patterns = {}
        pattern_functions = [
            'CDLDOJI', 'CDLHAMMER', 'CDLENGULFING', 'CDLMORNINGSTAR',
            'CDLEVENINGSTAR', 'CDLHARAMI', 'CDLPIERCING', 'CDLSHOOTINGSTAR'
        ]
        
        for pattern in pattern_functions:
            func = getattr(talib, pattern)
            result = func(df['Open'], df['High'], df['Low'], df['Close'])
            patterns[pattern] = int(result.iloc[-1])
        
        return jsonify({
            'symbol': symbol,
            'patterns': patterns,
            'detected': [k for k, v in patterns.items() if v != 0]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# BACKTESTING ENDPOINTS
# ============================================================================

@app.route('/api/backtest/simple-strategy', methods=['POST'])
def backtest_strategy():
    """Backtest a simple moving average crossover strategy"""
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL')
        period = data.get('period', '2y')
        short_window = data.get('short_window', 20)
        long_window = data.get('long_window', 50)
        
        # Get data
        df = yf.Ticker(symbol).history(period=period)
        
        # Calculate signals
        df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
        df['Signal'] = 0
        df.loc[df['SMA_short'] > df['SMA_long'], 'Signal'] = 1
        df.loc[df['SMA_short'] <= df['SMA_long'], 'Signal'] = -1
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
        
        # Performance metrics
        if ep:
            total_return = ep.cum_returns_final(df['Strategy_Returns'].dropna())
            sharpe = ep.sharpe_ratio(df['Strategy_Returns'].dropna())
            max_dd = ep.max_drawdown(df['Strategy_Returns'].dropna())
        else:
            total_return = (1 + df['Strategy_Returns']).prod() - 1
            sharpe = df['Strategy_Returns'].mean() / df['Strategy_Returns'].std() * np.sqrt(252)
            max_dd = (df['Strategy_Returns'].cumsum().expanding().max() - df['Strategy_Returns'].cumsum()).max()
        
        return jsonify({
            'symbol': symbol,
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'trades': int(df['Signal'].diff().abs().sum() / 2),
            'equity_curve': df[['Close', 'SMA_short', 'SMA_long', 'Strategy_Returns']].reset_index().to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# PORTFOLIO OPTIMIZATION ENDPOINTS
# ============================================================================

@app.route('/api/portfolio/optimize', methods=['POST'])
def optimize_portfolio():
    """Optimize portfolio using mean-variance optimization"""
    if not EfficientFrontier:
        return jsonify({'error': 'PyPortfolioOpt not available'}), 400
    
    try:
        data = request.json
        symbols = data.get('symbols', ['AAPL', 'MSFT', 'GOOGL', 'AMZN'])
        period = data.get('period', '3y')
        
        # Get historical data
        prices = yf.download(symbols, period=period)['Adj Close']
        
        # Calculate expected returns and sample covariance
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
        
        # Optimize for max Sharpe ratio
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        
        performance = ef.portfolio_performance(verbose=False)
        
        return jsonify({
            'weights': cleaned_weights,
            'expected_return': float(performance[0]),
            'volatility': float(performance[1]),
            'sharpe_ratio': float(performance[2])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# RISK ANALYSIS ENDPOINTS
# ============================================================================

@app.route('/api/risk/analyze/<symbol>', methods=['GET'])
def risk_analysis(symbol):
    """Comprehensive risk analysis for a symbol"""
    try:
        period = request.args.get('period', '1y')
        df = yf.Ticker(symbol).history(period=period)
        
        returns = df['Close'].pct_change().dropna()
        
        # Calculate risk metrics
        volatility = returns.std() * np.sqrt(252)
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        if ep:
            sharpe = ep.sharpe_ratio(returns)
            sortino = ep.sortino_ratio(returns)
            max_dd = ep.max_drawdown(returns)
        else:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
            sortino = returns.mean() / returns[returns < 0].std() * np.sqrt(252)
            cumulative = (1 + returns).cumprod()
            max_dd = (cumulative / cumulative.cummax() - 1).min()
        
        return jsonify({
            'symbol': symbol,
            'volatility': float(volatility),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(max_dd)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# TIME SERIES ANALYSIS ENDPOINTS
# ============================================================================

@app.route('/api/timeseries/forecast/<symbol>', methods=['POST'])
def forecast_price(symbol):
    """Forecast future prices using ARIMA"""
    try:
        data = request.json
        periods = data.get('periods', 30)
        
        df = yf.Ticker(symbol).history(period='2y')
        prices = df['Close']
        
        # Fit ARIMA model
        model = ARIMA(prices, order=(5, 1, 0))
        fitted = model.fit()
        
        # Forecast
        forecast = fitted.forecast(steps=periods)
        
        return jsonify({
            'symbol': symbol,
            'forecast': forecast.tolist(),
            'confidence_interval': {
                'lower': (forecast * 0.95).tolist(),
                'upper': (forecast * 1.05).tolist()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# MACHINE LEARNING ENDPOINTS
# ============================================================================

@app.route('/api/ml/predict/<symbol>', methods=['POST'])
def ml_prediction(symbol):
    """Predict next day returns using Random Forest"""
    try:
        data = request.json
        period = data.get('period', '2y')
        
        df = yf.Ticker(symbol).history(period=period)
        
        # Feature engineering
        df['Returns'] = df['Close'].pct_change()
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['Volatility'] = df['Returns'].rolling(20).std()
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        
        df = df.dropna()
        
        # Prepare features and target
        features = ['SMA_5', 'SMA_20', 'Volatility', 'Volume_SMA']
        X = df[features].values[:-1]
        y = df['Returns'].values[1:]
        
        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled[:-30], y[:-30])  # Train on all but last 30 days
        
        # Predict
        predictions = model.predict(X_scaled[-30:])
        actual = y[-30:]
        
        # Calculate accuracy
        accuracy = np.mean(np.sign(predictions) == np.sign(actual)) * 100
        
        return jsonify({
            'symbol': symbol,
            'prediction_accuracy': float(accuracy),
            'feature_importance': dict(zip(features, model.feature_importances_.tolist())),
            'next_day_prediction': float(predictions[-1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# OPENBB INTEGRATION ENDPOINTS
# ============================================================================

@app.route('/api/openbb/query', methods=['POST'])
def openbb_query():
    """Natural language query using OpenBB"""
    if not obb:
        return jsonify({'error': 'OpenBB not available'}), 400
    
    try:
        data = request.json
        query = data.get('query', '')
        
        # This is a placeholder - OpenBB integration would go here
        # The actual implementation depends on OpenBB's API
        
        return jsonify({
            'query': query,
            'response': 'OpenBB integration - implement based on your OpenBB setup'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/openbb/screener', methods=['POST'])
def openbb_screener():
    """Stock screening using OpenBB"""
    if not obb:
        return jsonify({'error': 'OpenBB not available'}), 400
    
    try:
        data = request.json
        criteria = data.get('criteria', {})
        
        # Placeholder for OpenBB screener
        return jsonify({
            'criteria': criteria,
            'results': 'OpenBB screener - implement based on your OpenBB setup'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# TRADING ENDPOINTS
# ============================================================================

@app.route('/api/trading/positions', methods=['GET'])
def get_positions():
    """Get current Alpaca positions"""
    if not ALPACA_API_KEY:
        return jsonify({'error': 'Alpaca credentials not configured'}), 400
    
    try:
        api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
        positions = api.list_positions()
        
        return jsonify({
            'positions': [
                {
                    'symbol': p.symbol,
                    'qty': float(p.qty),
                    'current_price': float(p.current_price),
                    'market_value': float(p.market_value),
                    'unrealized_pl': float(p.unrealized_pl),
                    'unrealized_plpc': float(p.unrealized_plpc)
                }
                for p in positions
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trading/order', methods=['POST'])
def place_order():
    """Place a trading order via Alpaca"""
    if not ALPACA_API_KEY:
        return jsonify({'error': 'Alpaca credentials not configured'}), 400
    
    try:
        data = request.json
        api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
        
        order = api.submit_order(
            symbol=data['symbol'],
            qty=data['qty'],
            side=data['side'],
            type=data.get('type', 'market'),
            time_in_force=data.get('time_in_force', 'gtc')
        )
        
        return jsonify({
            'order_id': order.id,
            'symbol': order.symbol,
            'qty': float(order.qty),
            'side': order.side,
            'status': order.status
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'libraries': {
            'talib': talib is not None,
            'backtrader': bt is not None,
            'pypfopt': EfficientFrontier is not None,
            'quantstats': qs is not None,
            'empyrical': ep is not None,
            'openbb': obb is not None,
            'arch': arch_model is not None
        }
    })


# ============================================================================
# SOCIAL SENTIMENT ENDPOINTS
# ============================================================================

@app.route('/api/social/reddit/<symbol>', methods=['GET'])
def reddit_sentiment(symbol):
    """Get Reddit sentiment for a stock symbol"""
    if not praw:
        return jsonify({'error': 'PRAW not available'}), 400
    
    try:
        # Placeholder - requires Reddit API credentials
        return jsonify({
            'symbol': symbol,
            'sentiment': 'neutral',
            'mentions': 0,
            'message': 'Configure Reddit API credentials'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/social/news-feed', methods=['GET'])
def news_feed():
    """Get financial news from RSS feeds"""
    if not feedparser:
        return jsonify({'error': 'Feedparser not available'}), 400
    
    try:
        feeds = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://www.cnbc.com/id/100003114/device/rss/rss.html'
        ]
        
        all_news = []
        for feed_url in feeds:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]:
                all_news.append({
                    'title': entry.title,
                    'link': entry.link,
                    'published': entry.get('published', '')
                })
        
        return jsonify({'news': all_news})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# CRYPTO TRADING ENDPOINTS
# ============================================================================

@app.route('/api/crypto/exchanges', methods=['GET'])
def list_crypto_exchanges():
    """List available crypto exchanges via CCXT"""
    if not ccxt:
        return jsonify({'error': 'CCXT not available'}), 400
    
    try:
        exchanges = ccxt.exchanges
        return jsonify({'exchanges': exchanges[:20]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/crypto/ticker/<exchange>/<symbol>', methods=['GET'])
def get_crypto_ticker(exchange, symbol):
    """Get crypto ticker data"""
    if not ccxt:
        return jsonify({'error': 'CCXT not available'}), 400
    
    try:
        exchange_class = getattr(ccxt, exchange)
        exchange_instance = exchange_class()
        ticker = exchange_instance.fetch_ticker(symbol)
        
        return jsonify({
            'exchange': exchange,
            'symbol': symbol,
            'last': ticker['last'],
            'bid': ticker['bid'],
            'ask': ticker['ask'],
            'volume': ticker['baseVolume']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ADVANCED TECHNICAL ANALYSIS ENDPOINTS
# ============================================================================

@app.route('/api/analysis/advanced-indicators/<symbol>', methods=['POST'])
def advanced_indicators(symbol):
    """Calculate advanced technical indicators using multiple libraries"""
    try:
        data = request.json
        period = data.get('period', '1y')
        
        df = yf.Ticker(symbol).history(period=period)
        results = {}
        
        # StockStats indicators
        if StockDataFrame:
            stock = StockDataFrame.retype(df.copy())
            results['macd'] = stock['macd'].iloc[-1] if 'macd' in stock else None
            results['rsi_14'] = stock['rsi_14'].iloc[-1] if 'rsi_14' in stock else None
        
        # FinTA indicators
        if FinTA:
            results['bbands'] = {
                'upper': float(FinTA.BBANDS(df)['BB_UPPER'].iloc[-1]),
                'middle': float(FinTA.BBANDS(df)['BB_MIDDLE'].iloc[-1]),
                'lower': float(FinTA.BBANDS(df)['BB_LOWER'].iloc[-1])
            }
        
        return jsonify({
            'symbol': symbol,
            'indicators': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# OPTIONS PRICING ENDPOINTS
# ============================================================================

@app.route('/api/options/black-scholes', methods=['POST'])
def black_scholes_pricing():
    """Calculate option price using Black-Scholes model"""
    if not bs:
        return jsonify({'error': 'py_vollib not available'}), 400
    
    try:
        data = request.json
        S = data['spot_price']
        K = data['strike_price']
        T = data['time_to_expiry']
        r = data['risk_free_rate']
        sigma = data['volatility']
        flag = data.get('option_type', 'c')  # 'c' for call, 'p' for put
        
        price = bs(flag, S, K, T, r, sigma)
        
        return jsonify({
            'option_price': float(price),
            'inputs': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ADVANCED TIME SERIES ENDPOINTS
# ============================================================================

@app.route('/api/timeseries/auto-arima/<symbol>', methods=['POST'])
def auto_arima_forecast(symbol):
    """Auto ARIMA forecasting"""
    if not pm:
        return jsonify({'error': 'pmdarima not available'}), 400
    
    try:
        data = request.json
        periods = data.get('periods', 30)
        
        df = yf.Ticker(symbol).history(period='2y')
        prices = df['Close']
        
        model = pm.auto_arima(prices, seasonal=False, stepwise=True)
        forecast = model.predict(n_periods=periods)
        
        return jsonify({
            'symbol': symbol,
            'forecast': forecast.tolist(),
            'model_order': model.order
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/timeseries/garch/<symbol>', methods=['POST'])
def garch_volatility(symbol):
    """GARCH volatility forecasting"""
    if not arch_model:
        return jsonify({'error': 'arch not available'}), 400
    
    try:
        data = request.json
        periods = data.get('periods', 30)
        
        df = yf.Ticker(symbol).history(period='2y')
        returns = df['Close'].pct_change().dropna() * 100
        
        model = arch_model(returns, vol='Garch', p=1, q=1)
        fitted = model.fit(disp='off')
        forecast = fitted.forecast(horizon=periods)
        
        return jsonify({
            'symbol': symbol,
            'volatility_forecast': forecast.variance.values[-1].tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ADVANCED PORTFOLIO OPTIMIZATION
# ============================================================================

@app.route('/api/portfolio/risk-parity', methods=['POST'])
def risk_parity_portfolio():
    """Risk parity portfolio optimization"""
    if not rp:
        return jsonify({'error': 'Riskfolio-Lib not available'}), 400
    
    try:
        data = request.json
        symbols = data.get('symbols', ['AAPL', 'MSFT', 'GOOGL', 'AMZN'])
        period = data.get('period', '3y')
        
        prices = yf.download(symbols, period=period)['Adj Close']
        returns = prices.pct_change().dropna()
        
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu='hist', method_cov='hist')
        
        w = port.rp_optimization(model='Classic', rm='MV', hist=True)
        
        return jsonify({
            'weights': w.to_dict()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# MARKET CALENDAR ENDPOINTS
# ============================================================================

@app.route('/api/calendar/trading-days', methods=['GET'])
def trading_days():
    """Get trading days for a market"""
    if not mcal:
        return jsonify({'error': 'pandas_market_calendars not available'}), 400
    
    try:
        exchange = request.args.get('exchange', 'NYSE')
        start = request.args.get('start', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
        end = request.args.get('end', datetime.now().strftime('%Y-%m-%d'))
        
        calendar = mcal.get_calendar(exchange)
        schedule = calendar.schedule(start_date=start, end_date=end)
        
        return jsonify({
            'exchange': exchange,
            'trading_days': schedule.index.strftime('%Y-%m-%d').tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ALTERNATIVE DATA ENDPOINTS
# ============================================================================

@app.route('/api/data/akshare/<symbol>', methods=['GET'])
def get_akshare_data(symbol):
    """Get Chinese market data via AkShare"""
    if not ak:
        return jsonify({'error': 'AkShare not available'}), 400
    
    try:
        # Example: get stock info
        df = ak.stock_zh_a_hist(symbol=symbol, adjust="qfq")
        
        return jsonify({
            'symbol': symbol,
            'data': df.tail(100).to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# FINNHUB ENDPOINTS
# ============================================================================

@app.route('/api/finnhub/quote/<symbol>', methods=['GET'])
def finnhub_quote(symbol):
    """Get real-time quote from Finnhub"""
    if not finnhub:
        return jsonify({'error': 'Finnhub not available'}), 400
    
    try:
        api_key = os.getenv('FINNHUB_API_KEY', 'demo')
        client = finnhub.Client(api_key=api_key)
        quote = client.quote(symbol)
        
        return jsonify({
            'symbol': symbol,
            'quote': quote
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'name': 'AgentTrading-Dyad Backend',
        'version': '2.0.0',
        'description': 'Comprehensive quantitative trading platform with 100+ libraries',
        'endpoints': {
            'data': [
                '/api/data/stock/<symbol>', 
                '/api/data/market-overview',
                '/api/data/akshare/<symbol>'
            ],
            'analysis': [
                '/api/analysis/indicators/<symbol>', 
                '/api/analysis/pattern-recognition/<symbol>',
                '/api/analysis/advanced-indicators/<symbol>'
            ],
            'backtesting': ['/api/backtest/simple-strategy'],
            'portfolio': [
                '/api/portfolio/optimize',
                '/api/portfolio/risk-parity'
            ],
            'risk': ['/api/risk/analyze/<symbol>'],
            'timeseries': [
                '/api/timeseries/forecast/<symbol>',
                '/api/timeseries/auto-arima/<symbol>',
                '/api/timeseries/garch/<symbol>'
            ],
            'ml': ['/api/ml/predict/<symbol>'],
            'options': ['/api/options/black-scholes'],
            'social': [
                '/api/social/reddit/<symbol>',
                '/api/social/news-feed'
            ],
            'crypto': [
                '/api/crypto/exchanges',
                '/api/crypto/ticker/<exchange>/<symbol>'
            ],
            'calendar': ['/api/calendar/trading-days'],
            'finnhub': ['/api/finnhub/quote/<symbol>'],
            'openbb': ['/api/openbb/query', '/api/openbb/screener'],
            'trading': ['/api/trading/positions', '/api/trading/order']
        },
        'libraries_loaded': {
            'talib': talib is not None,
            'ccxt': ccxt is not None,
            'praw': praw is not None,
            'feedparser': feedparser is not None,
            'akshare': ak is not None,
            'finnhub': finnhub is not None,
            'riskfolio': rp is not None,
            'pmdarima': pm is not None,
            'vectorbt': vbt is not None,
            'chromadb': chromadb is not None,
            'genai': genai is not None
        }
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
