"""
AgentTrading-Dyad Backend
Comprehensive quantitative trading and analysis platform
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv
import json
import pandas as pd
import yfinance as yf
from ta import add_all_ta_features
from ib_insync import IB, util
from td import oauth as auth, client
from datetime import datetime, timedelta
import asyncio
from sqlalchemy import create_engine

# Import Alpaca Backtrader integration
from alpaca_integration import AlpacaBacktraderClient, SmaCross

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={"/api/*": {"origins": ["https://tradespark.app"]}})

# Initialize IBKR client
ib = IB()

# TD Ameritrade Configuration
TDA_REDIRECT_URI = 'http://localhost:5000/api/td/oauth'
TDA_TOKEN_PATH = 'tda_token.json'
TDA_API_KEY = os.getenv('TDA_API_KEY', '')

# Initialize Alpaca client
alpaca_client = AlpacaBacktraderClient(
    api_key=os.getenv('ALPACA_API_KEY'),
    api_secret=os.getenv('ALPACA_SECRET_KEY'),
    paper=True
)

# Serve static files (frontend)
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Serve frontend index.html
@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'libraries': {
            'ta': True,
            'yfinance': True,
            'ib_insync': True,
            'tda': True
        },
        'broker_connections': {
            'ibkr': ib.isConnected(),
            'tda': os.path.exists(TDA_TOKEN_PATH)
        }
    })

# Data endpoints
@app.route('/api/data/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    """Get stock OHLCV data with technical indicators"""
    try:
        # Get data from Yahoo Finance
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y")
        
        # Add technical indicators
        df = add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )
        
        # Convert to JSON-serializable format
        data = {
            'symbol': symbol,
            'data': json.loads(df.reset_index().to_json(orient='records', date_format='iso'))
        }
        
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# IBKR Endpoints
@app.route('/api/ib/connect', methods=['POST'])
def connect_ib():
    """Connect to Interactive Brokers TWS/IB Gateway"""
    try:
        host = request.json.get('host', '127.0.0.1')
        port = request.json.get('port', 7497)
        client_id = request.json.get('client_id', 1)
        
        if not ib.isConnected():
            ib.connect(host, port, clientId=client_id)
            return jsonify({
                'status': 'success',
                'message': f'Connected to TWS on {host}:{port} with client ID {client_id}'
            })
        return jsonify({
            'status': 'success',
            'message': 'Already connected to TWS'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to connect to TWS: {str(e)}'
        }), 500

@app.route('/api/ib/account', methods=['GET'])
def get_ib_account():
    """Get IBKR account information"""
    try:
        if not ib.isConnected():
            return jsonify({
                'status': 'error',
                'message': 'Not connected to TWS. Call /api/ib/connect first.'
            }), 400
            
        account = ib.accountSummary()
        return jsonify({
            'status': 'success',
            'data': [a.dict() for a in account]
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get account info: {str(e)}'
        }), 500

# TD Ameritrade Endpoints
@app.route('/api/td/auth')
def td_auth():
    """Initiate TD Ameritrade OAuth flow"""
    try:
        c = auth.client_from_token_file(TDA_TOKEN_PATH, TDA_API_KEY)
        return jsonify({'status': 'success', 'message': 'Already authenticated'})
    except FileNotFoundError:
        return jsonify({
            'status': 'redirect',
            'url': auth.client_from_login_flow(
                auth.easy_client(
                    TDA_API_KEY,
                    TDA_REDIRECT_URI,
                    TDA_TOKEN_PATH,
                    asyncio.get_event_loop()
                ),
                asyncio.get_event_loop()
            )
        })

@app.route('/api/td/account', methods=['GET'])
async def get_td_account():
    """Get TD Ameritrade account information"""
    try:
        c = auth.client_from_token_file(TDA_TOKEN_PATH, TDA_API_KEY)
        account = await c.get_accounts(fields=['positions', 'orders'])
        return jsonify({
            'status': 'success',
            'data': account
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get TD Ameritrade account: {str(e)}'
        }), 500

# Trading Endpoints
@app.route('/api/trade/place', methods=['POST'])
def place_order():
    """Place a trade order"""
    try:
        data = request.json
        broker = data.get('broker')
        symbol = data.get('symbol')
        quantity = data.get('quantity')
        order_type = data.get('order_type', 'market')
        
        if broker.lower() == 'ibkr':
            if not ib.isConnected():
                return jsonify({
                    'status': 'error',
                    'message': 'Not connected to TWS. Call /api/ib/connect first.'
                }), 400
                
            # Example: Place a market order
            contract = ib.Stock(symbol, 'SMART', 'USD')
            order = ib.marketOrder('BUY' if quantity > 0 else 'SELL', abs(quantity))
            trade = ib.placeOrder(contract, order)
            
            return jsonify({
                'status': 'success',
                'broker': 'ibkr',
                'order_id': trade.order.orderId,
                'status': trade.orderStatus.status
            })
            
        elif broker.lower() in ['tda', 'tdameritrade', 'schwab']:
            return jsonify({
                'status': 'error',
                'message': 'TD Ameritrade/Schwab order execution not implemented yet'
            }), 501
            
        else:
            return jsonify({
                'status': 'error',
                'message': f'Unsupported broker: {broker}'
            }), 400
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to place order: {str(e)}'
        }), 500

# Root endpoint
@app.route('/api', methods=['GET'])
def index():
    """Root API endpoint with documentation"""
    return jsonify({
        'name': 'AgentTrading-Dyad Backend',
        'version': '2.0.0',
        'description': 'Quantitative trading platform with multi-broker support',
        'endpoints': {
            'health': '/api/health',
            'data': {
                'stock_data': '/api/data/stock/<symbol>'
            },
            'ibkr': {
                'connect': '/api/ib/connect (POST)',
                'account': '/api/ib/account (GET)'
            },
            'td_ameritrade': {
                'auth': '/api/td/auth (GET)',
                'account': '/api/td/account (GET)'
            },
            'trading': {
                'place_order': '/api/trade/place (POST)'
            }
        }
    })

# Alpaca Backtrader Endpoints
@app.route('/api/alpaca/backtest', methods=['POST'])
def run_alpaca_backtest():
    """Run a backtest with Alpaca data"""
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL')
        strategy = data.get('strategy', 'sma_cross')
        fast = int(data.get('fast', 10))
        slow = int(data.get('slow', 30))
        
        # Map strategy names to classes
        strategies = {
            'sma_cross': SmaCross
        }
        
        if strategy not in strategies:
            return jsonify({
                'status': 'error',
                'message': f'Unsupported strategy: {strategy}'
            }), 400
        
        # Run backtest
        results = alpaca_client.run_backtest(
            strategy=strategies[strategy],
            symbol=symbol,
            fast=fast,
            slow=slow,
            printlog=False
        )
        
        return jsonify({
            'status': 'success',
            'data': results
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/alpaca/account', methods=['GET'])
def get_alpaca_account():
    """Get Alpaca account information"""
    try:
        if not alpaca_client.connect():
            return jsonify({
                'status': 'error',
                'message': 'Failed to connect to Alpaca'
            }), 500
            
        # Get account info through the broker
        broker = alpaca_client.get_broker()
        account = broker.get_balance()
        
        return jsonify({
            'status': 'success',
            'data': {
                'cash': account.cash,
                'value': account.value,
                'unrealized_pnl': account.unrealized_pnl,
                'realized_pnl': account.realized_pnl
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Create token directory if it doesn't exist
    os.makedirs(os.path.dirname(TDA_TOKEN_PATH), exist_ok=True)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
    

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        # Placeholder for analysis functionality
        return jsonify({
            'status': 'success',
            'message': 'Analysis endpoint ready',
            'data': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Placeholder for prediction functionality
        return jsonify({
            'status': 'success',
            'message': 'Prediction endpoint ready',
            'data': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/backtest', methods=['POST'])
def backtest():
    try:
        data = request.get_json()
        # Placeholder for backtesting functionality
        return jsonify({
            'status': 'success',
            'message': 'Backtest endpoint ready',
            'data': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
