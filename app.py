"""
AgentTrading-Dyad Backend
Comprehensive quantitative trading and analysis platform
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Core Data & Analysis
import pandas as pd
import numpy as np

# Financial Data
import yfinance as yf
import alpaca_trade_api as tradeapi

# Technical Analysis
try:
    import talib
except ImportError:
    print("TA-Lib not available - install with: pip install TA-Lib")
    talib = None

import pandas_ta as ta_pandas
from ta import add_all_ta_features

# Backtesting
try:
    import backtrader as bt
except ImportError:
    bt = None

try:
    from backtesting import Backtest, Strategy
except ImportError:
    Backtest = None

# Portfolio Optimization
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
except ImportError:
    EfficientFrontier = None

# Risk Analytics
try:
    import quantstats as qs
except ImportError:
    qs = None

try:
    import empyrical as ep
except ImportError:
    ep = None

# OpenBB
try:
    from openbb import obb
except ImportError:
    print("OpenBB not available - install with: pip install openbb")
    obb = None

# Time Series
from statsmodels.tsa.arima.model import ARIMA
try:
    from arch import arch_model
except ImportError:
    arch_model = None

# ML
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Calendars
try:
    from exchange_calendars import get_calendar
except ImportError:
    get_calendar = None

# Visualization
import mplfinance as mpf
import plotly.graph_objects as go

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


@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'name': 'AgentTrading-Dyad Backend',
        'version': '1.0.0',
        'endpoints': {
            'data': ['/api/data/stock/<symbol>', '/api/data/market-overview'],
            'analysis': ['/api/analysis/indicators/<symbol>', '/api/analysis/pattern-recognition/<symbol>'],
            'backtesting': ['/api/backtest/simple-strategy'],
            'portfolio': ['/api/portfolio/optimize'],
            'risk': ['/api/risk/analyze/<symbol>'],
            'timeseries': ['/api/timeseries/forecast/<symbol>'],
            'ml': ['/api/ml/predict/<symbol>'],
            'openbb': ['/api/openbb/query', '/api/openbb/screener'],
            'trading': ['/api/trading/positions', '/api/trading/order']
        }
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
