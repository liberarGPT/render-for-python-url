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
from tda import auth, client
from datetime import datetime, timedelta
import asyncio
from sqlalchemy import create_engine

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["https://tradespark.app"]}})

# Initialize IBKR client
ib = IB()

   # Health check
   @app.route('/health', methods=['GET'])
   def health_check():
       """Health check endpoint"""
       return jsonify({
           'status': 'healthy',
           'libraries': {
               'ta': True,
               'yfinance': True
           }
       })

   # Data endpoints
   @app.route('/api/data/stock/<symbol>', methods=['GET'])
   def get_stock_data(symbol):
       """Get stock OHLCV data with technical indicators"""
       try:
           period = request.args.get('period', '1y')
           interval = request.args.get('interval', '1d')
           
           ticker = yf.Ticker(symbol)
           df = ticker.history(period=period, interval=interval)
           
           # Use ta for technical indicators
           df = add_all_ta_features(
               df, open="Open", high="High", low="Low",
               close="Close", volume="Volume", fillna=True
           )
           
           result = {
               'symbol': symbol,
               'data': df.reset_index().to_dict(orient='records')
           }
           
           return jsonify(result)
       except Exception as e:
           return jsonify({'error': str(e)}), 500

   @app.route('/api/data/market-overview', methods=['GET'])
   def market_overview():
       """Get market overview with multiple indices"""
       try:
           symbols = ['^GSPC', '^DJI', '^IXIC', '^RUT']
           data = {}
           
           for symbol in symbols:
               ticker = yf.Ticker(symbol)
               hist = ticker.history(period='5d')
               data[symbol] = {
                   'price': hist['Close'].iloc[-1],
                   'change': ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100,
                   'volume': int(hist['Volume'].iloc[-1])
               }
           
           return jsonify(data)
       except Exception as e:
           return jsonify({'error': str(e)}), 500

   # Technical analysis endpoints
   @app.route('/api/analysis/pattern-recognition/<symbol>', methods=['GET'])
   def pattern_recognition(symbol):
       """Detect technical indicators using ta"""
       try:
           df = yf.Ticker(symbol).history(period='3mo')
           
           # Use ta for indicators
           df = add_all_ta_features(
               df, open="Open", high="High", low="Low",
               close="Close", volume="Volume", fillna=True
           )
           
           indicators = {
               'sma_20': df['trend_sma_fast'].iloc[-1] if 'trend_sma_fast' in df else None,
               'sma_50': df['trend_sma_slow'].iloc[-1] if 'trend_sma_slow' in df else None,
               'rsi_14': df['momentum_rsi'].iloc[-1] if 'momentum_rsi' in df else None,
               'macd': df['trend_macd'].iloc[-1] if 'trend_macd' in df else None
           }
           
           return jsonify({
               'symbol': symbol,
               'indicators': indicators
           })
       except Exception as e:
           return jsonify({'error': str(e)}), 500

   # Database endpoints
   @app.route('/api/database/store/<symbol>', methods=['POST'])
   def store_data(symbol):
       """Store stock data in PostgreSQL database"""
       try:
           from sqlalchemy import create_engine
       except ImportError:
           return jsonify({'error': 'SQLAlchemy not available'}), 400
       
       try:
           data = request.json or {}
           period = data.get('period', '1y')
           db_url = data.get('db_url', os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/trading'))
           
           df = yf.Ticker(symbol).history(period=period)
           
           engine = create_engine(db_url)
           df.to_sql('stock_data', engine, if_exists='append', index=True)
           
           return jsonify({
               'symbol': symbol,
               'message': f'Data stored for {symbol}'
           })
       except Exception as e:
           return jsonify({'error': str(e)}), 500

   @app.route('/api/database/retrieve/<symbol>', methods=['GET'])
   def retrieve_data(symbol):
       """Retrieve stock data from PostgreSQL database"""
       try:
           from sqlalchemy import create_engine
       except ImportError:
           return jsonify({'error': 'SQLAlchemy not available'}), 400
       
       try:
           data = request.args
           db_url = data.get('db_url', os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/trading'))
           
           engine = create_engine(db_url)
           df = pd.read_sql(f"SELECT * FROM stock_data WHERE symbol = '{symbol}'", engine)
           
           return jsonify({
               'symbol': symbol,
               'data': df.to_dict(orient='records')
           })
       except Exception as e:
           return jsonify({'error': str(e)}), 500

   # Initialize IBKR client
   ib = IB()

   # TD Ameritrade Configuration
   TDA_REDIRECT_URI = 'http://localhost:5000/api/td/oauth'
   TDA_TOKEN_PATH = 'tda_token.json'
   TDA_API_KEY = os.getenv('TDA_API_KEY', '')

   # IBKR Connection
   @app.route('/api/ib/connect', methods=['POST'])
   def connect_ib():
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

   # Get IBKR Account Info
   @app.route('/api/ib/account', methods=['GET'])
   def get_ib_account():
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

   # TD Ameritrade Authentication
   @app.route('/api/td/auth')
   def td_auth():
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

   # Get TD Ameritrade Account Info
   @app.route('/api/td/account', methods=['GET'])
   async def get_td_account():
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

   # Place Order (Example for IBKR)
   @app.route('/api/trade/place', methods=['POST'])
   def place_order():
       try:
           if not ib.isConnected():
               return jsonify({
                   'status': 'error',
                   'message': 'Not connected to TWS'
               }), 400
                
           data = request.json
           contract = data.get('contract')
           order = data.get('order')
           
           # Example: Place a simple market order
           # This is a simplified example - you'll need to implement proper contract and order creation
           trade = ib.placeOrder(contract, order)
           
           return jsonify({
               'status': 'success',
               'data': {
                   'order_id': trade.order.orderId,
                   'status': trade.orderStatus.status
               }
           })
       except Exception as e:
           return jsonify({
               'status': 'error',
               'message': f'Failed to place order: {str(e)}'
           }), 500

   # Root endpoint
   @app.route('/', methods=['GET'])
   def index():
       """Root endpoint"""
       return jsonify({
           'name': 'AgentTrading-Dyad Backend',
           'version': '2.0.0',
           'description': 'Quantitative trading platform',
           'endpoints': {
               'data': [
                   '/api/data/stock/<symbol>',
                   '/api/data/market-overview'
               ],
               'analysis': [
                   '/api/analysis/pattern-recognition/<symbol>'
               ],
               'database': [
                   '/api/database/store/<symbol>',
                   '/api/database/retrieve/<symbol>'
               ],
               'ibkr': [
                   '/api/ib/connect',
                   '/api/ib/account'
               ],
               'tda': [
                   '/api/td/auth',
                   '/api/td/account'
               ],
               'trade': [
                   '/api/trade/place'
               ]
           }
       })

   if __name__ == '__main__':
       # Create token directory if it doesn't exist
       os.makedirs(os.path.dirname(TDA_TOKEN_PATH), exist_ok=True)
       
       port = int(os.getenv('PORT', 5000))
       app.run(host='0.0.0.0', port=port, debug=True)