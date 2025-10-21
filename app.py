from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import random
import os
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import backtrader as bt
import vectorbt as vbt
from backtesting import Backtest, Strategy
import ta
import finta
import plotly.graph_objects as go
import plotly.express as px
from alpaca_trade_api import REST
import requests

app = Flask(__name__)
CORS(app) # Allow all origins for simplicity in Vercel deployment

# =============================================================================
# SUPABASE INTEGRATION
# =============================================================================

try:
    from supabase import create_client, Client
    import jwt
    
    # Initialize Supabase client
    SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://mtozccjlxpiohllyteta.supabase.co')
    SUPABASE_KEY = os.getenv('SUPABASE_ANON_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im10b3pjY2pseHBpb2hsbHl0ZXRhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjAzMjQ0ODcsImV4cCI6MjA3NTkwMDQ4N30.RsbZd2iUlmOP56iYg0F7mgm0oYqUEcQtO8r2XAOAeTQ')
    
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    SUPABASE_ENABLED = True
except Exception as e:
    print(f"Supabase initialization failed: {e}")
    SUPABASE_ENABLED = False

# =============================================================================
# OPENBB INTEGRATION
# =============================================================================

class OpenBBIntegration:
    def __init__(self):
        self.base_url = "https://api.openbb.co"
        self.pat_token = None
    
    def set_pat_token(self, token):
        """Set the OpenBB PAT token for API access"""
        self.pat_token = token
    
    def get_equity_data(self, symbol, timeframe="1d", limit=100):
        """Get equity data from OpenBB"""
        try:
            if not self.pat_token:
                return {"error": "OpenBB PAT token not set"}
            
            # Mock data for now - replace with actual OpenBB API calls
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "data": {
                    "prices": [random.uniform(100, 200) for _ in range(limit)],
                    "volumes": [random.randint(1000, 10000) for _ in range(limit)],
                    "timestamps": [(datetime.now() - timedelta(days=i)).isoformat() for i in range(limit, 0, -1)]
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_crypto_data(self, symbol, timeframe="1d"):
        """Get cryptocurrency data"""
        return self.get_equity_data(symbol, timeframe)
    
    def get_forex_data(self, pair, timeframe="1d"):
        """Get forex data"""
        return self.get_equity_data(pair, timeframe)

openbb = OpenBBIntegration()

# =============================================================================
# MCP (MODEL CONTEXT PROTOCOL) INTEGRATION
# =============================================================================

class MCPIntegration:
    def __init__(self):
        self.tools = {
            "alpaca_trading": {
                "description": "Execute trades on Alpaca",
                "parameters": ["symbol", "qty", "side", "type"]
            },
            "market_data": {
                "description": "Get real-time market data",
                "parameters": ["symbol", "timeframe"]
            },
            "portfolio_analysis": {
                "description": "Analyze portfolio performance",
                "parameters": ["portfolio_id"]
            },
            "risk_management": {
                "description": "Calculate risk metrics",
                "parameters": ["positions", "risk_tolerance"]
            }
        }
    
    def execute_tool(self, tool_name, parameters, user_id=None):
        """Execute an MCP tool"""
        try:
            if tool_name not in self.tools:
                return {"error": f"Tool {tool_name} not found"}
            
            # Mock execution - replace with actual MCP tool execution
            result = {
                "tool": tool_name,
                "parameters": parameters,
                "result": f"Mock execution of {tool_name} with {parameters}",
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id
            }
            
            # Store in database if Supabase is available
            if SUPABASE_ENABLED and user_id:
                try:
                    supabase.table("mcp_tool_usage").insert({
                        "session_id": f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        "tool_name": tool_name,
                        "input_data": parameters,
                        "output_data": result,
                        "success": True
                    }).execute()
                except Exception as e:
                    print(f"Failed to store MCP usage: {e}")
            
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def get_available_tools(self):
        """Get list of available MCP tools"""
        return self.tools

mcp = MCPIntegration()

# =============================================================================
# AI CHAT INTEGRATION
# =============================================================================

class AIChatIntegration:
    def __init__(self):
        self.conversations = {}
    
    def process_message(self, message, user_id, context="strategy_building"):
        """Process AI chat message for strategy building"""
        try:
            # Mock AI response - replace with actual AI integration
            responses = {
                "strategy_building": [
                    "I can help you build algorithmic trading strategies. What type of strategy are you interested in?",
                    "Let's analyze your portfolio and suggest some strategies based on your risk tolerance.",
                    "I can create custom indicators and backtest them for you.",
                    "Would you like me to help you implement a momentum strategy or a mean reversion strategy?"
                ],
                "market_analysis": [
                    "I can analyze market trends and provide insights on potential opportunities.",
                    "Let me check the latest market data and identify patterns.",
                    "Based on current market conditions, I recommend focusing on defensive stocks.",
                    "I can help you identify oversold or overbought conditions in the market."
                ],
                "alpaca_integration": [
                    "I can help you set up automated trading with Alpaca API.",
                    "Let's configure your Alpaca credentials and create trading strategies.",
                    "I can help you implement paper trading first, then move to live trading.",
                    "Would you like me to create a portfolio rebalancing strategy?"
                ]
            }
            
            # Get appropriate response based on context
            context_responses = responses.get(context, responses["strategy_building"])
            response = random.choice(context_responses)
            
            # Store conversation in database if Supabase is available
            if SUPABASE_ENABLED:
                try:
                    # Create or get session
                    session_result = supabase.table("ai_chat_sessions").select("id").eq("user_id", user_id).eq("context", context).order("created_at", desc=True).limit(1).execute()
                    
                    if session_result.data:
                        session_id = session_result.data[0]["id"]
                    else:
                        session_result = supabase.table("ai_chat_sessions").insert({
                            "user_id": user_id,
                            "session_name": f"AI Chat - {context}",
                            "context": context
                        }).execute()
                        session_id = session_result.data[0]["id"]
                    
                    # Store messages
                    supabase.table("ai_chat_messages").insert([
                        {
                            "session_id": session_id,
                            "role": "user",
                            "content": message,
                            "metadata": {"context": context}
                        },
                        {
                            "session_id": session_id,
                            "role": "assistant",
                            "content": response,
                            "metadata": {"context": context}
                        }
                    ]).execute()
                except Exception as e:
                    print(f"Failed to store chat messages: {e}")
            
            return {
                "response": response,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}

ai_chat = AIChatIntegration()

# =============================================================================
# CORE ENDPOINTS
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'platform': 'vercel',
        'supabase_enabled': SUPABASE_ENABLED,
        'timestamp': datetime.utcnow().isoformat()
    })

# =============================================================================
# TRADE IDEA GENERATION ENDPOINTS
# =============================================================================

@app.route('/api/trade-ideas', methods=['GET'])
def get_trade_ideas():
    """Returns AI-generated comprehensive trade ideas."""
    mock_ideas = [
        {
            "id": "ti-001",
            "symbol": "NVDA",
            "idea": "AI chip demand surge, strong technicals, unusual call options activity.",
            "recommendation": "BUY",
            "confidence": 0.9,
            "risk": "Medium",
            "entry_price": 450.00,
            "target_price": 500.00,
            "stop_loss": 430.00,
            "sources": ["Quant Indicators", "Unusual Options", "News Analysis"],
            "timestamp": (datetime.utcnow() - timedelta(hours=1)).isoformat()
        },
        {
            "id": "ti-002",
            "symbol": "TSLA",
            "idea": "Oversold conditions, potential short squeeze, recent insider buying.",
            "recommendation": "BUY",
            "confidence": 0.7,
            "risk": "High",
            "entry_price": 220.00,
            "target_price": 240.00,
            "stop_loss": 210.00,
            "sources": ["Quant Indicators", "Insider Trading", "Social Sentiment"],
            "timestamp": (datetime.utcnow() - timedelta(hours=3)).isoformat()
        }
    ]
    
    return jsonify({
        'status': 'success',
        'ideas': mock_ideas
    })

@app.route('/api/quant-indicators', methods=['POST'])
def get_quant_indicators():
    """Returns quantitative indicators for a given symbol."""
    data = request.get_json() or {}
    symbol = data.get('symbol', 'AAPL')
    
    return jsonify({
        'status': 'success',
        'symbol': symbol,
        'indicators': {
            'RSI': random.uniform(30, 70),
            'MACD': random.uniform(-5, 5),
            'BollingerBands': {'upper': 160, 'middle': 150, 'lower': 140},
            'Volume': random.randint(1000000, 50000000)
        }
    })

# =============================================================================
# OPENBB INTEGRATION ENDPOINTS
# =============================================================================

@app.route('/api/openbb/set-token', methods=['POST'])
def set_openbb_token():
    """Set OpenBB PAT token"""
    data = request.get_json() or {}
    token = data.get('token')
    
    if not token:
        return jsonify({'error': 'Token required'}), 400
    
    openbb.set_pat_token(token)
    return jsonify({'status': 'success', 'message': 'OpenBB token set'})

@app.route('/api/openbb/equity/<symbol>', methods=['GET'])
def get_openbb_equity(symbol):
    """Get equity data from OpenBB"""
    timeframe = request.args.get('timeframe', '1d')
    limit = int(request.args.get('limit', 100))
    
    result = openbb.get_equity_data(symbol, timeframe, limit)
    return jsonify(result)

@app.route('/api/openbb/crypto/<symbol>', methods=['GET'])
def get_openbb_crypto(symbol):
    """Get crypto data from OpenBB"""
    timeframe = request.args.get('timeframe', '1d')
    
    result = openbb.get_crypto_data(symbol, timeframe)
    return jsonify(result)

@app.route('/api/openbb/forex/<pair>', methods=['GET'])
def get_openbb_forex(pair):
    """Get forex data from OpenBB"""
    timeframe = request.args.get('timeframe', '1d')
    
    result = openbb.get_forex_data(pair, timeframe)
    return jsonify(result)

# =============================================================================
# MCP INTEGRATION ENDPOINTS
# =============================================================================

@app.route('/api/mcp/tools', methods=['GET'])
def get_mcp_tools():
    """Get available MCP tools"""
    return jsonify({
        'status': 'success',
        'tools': mcp.get_available_tools()
    })

@app.route('/api/mcp/execute', methods=['POST'])
def execute_mcp_tool():
    """Execute an MCP tool"""
    data = request.get_json() or {}
    tool_name = data.get('tool_name')
    parameters = data.get('parameters', {})
    user_id = data.get('user_id')
    
    if not tool_name:
        return jsonify({'error': 'tool_name required'}), 400
    
    result = mcp.execute_tool(tool_name, parameters, user_id)
    return jsonify(result)

# =============================================================================
# AI CHAT ENDPOINTS
# =============================================================================

@app.route('/api/ai-chat', methods=['POST'])
def ai_chat_endpoint():
    """AI chat for strategy building"""
    data = request.get_json() or {}
    message = data.get('message')
    user_id = data.get('user_id')
    context = data.get('context', 'strategy_building')
    
    if not message:
        return jsonify({'error': 'message required'}), 400
    
    result = ai_chat.process_message(message, user_id, context)
    return jsonify(result)

@app.route('/api/ai-chat/sessions', methods=['GET'])
def get_chat_sessions():
    """Get AI chat sessions for a user"""
    user_id = request.args.get('user_id')
    context = request.args.get('context', 'strategy_building')
    
    if not user_id:
        return jsonify({'error': 'user_id required'}), 400
    
    if not SUPABASE_ENABLED:
        return jsonify({'error': 'Supabase not available'}), 500
    
    try:
        result = supabase.table("ai_chat_sessions").select("*").eq("user_id", user_id).eq("context", context).order("created_at", desc=True).execute()
        return jsonify({'status': 'success', 'sessions': result.data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-chat/messages/<session_id>', methods=['GET'])
def get_chat_messages(session_id):
    """Get messages for a chat session"""
    if not SUPABASE_ENABLED:
        return jsonify({'error': 'Supabase not available'}), 500
    
    try:
        result = supabase.table("ai_chat_messages").select("*").eq("session_id", session_id).order("created_at", asc=True).execute()
        return jsonify({'status': 'success', 'messages': result.data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ALPACA INTEGRATION ENDPOINTS
# =============================================================================

@app.route('/api/alpaca/account', methods=['GET'])
def get_alpaca_account():
    """Get Alpaca account information"""
    # This would integrate with actual Alpaca API
    return jsonify({
        'status': 'success',
        'account': {
            'buying_power': 10000.00,
            'cash': 5000.00,
            'portfolio_value': 15000.00,
            'equity': 15000.00
        }
    })

@app.route('/api/alpaca/positions', methods=['GET'])
def get_alpaca_positions():
    """Get Alpaca positions"""
    return jsonify({
        'status': 'success',
        'positions': [
            {
                'symbol': 'AAPL',
                'qty': 10,
                'market_value': 1500.00,
                'unrealized_pl': 50.00
            }
        ]
    })

# =============================================================================
# BACKTESTING INTEGRATION
# =============================================================================

class BacktestEngine:
    def __init__(self):
        self.results = {}
    
    def run_backtrader_backtest(self, symbol, start_date, end_date, strategy_class, initial_cash=10000):
        """Run backtest using Backtrader"""
        try:
            cerebro = bt.Cerebro()
            
            # Add data feed
            data = bt.feeds.YahooFinanceData(
                dataname=symbol,
                fromdate=datetime.strptime(start_date, '%Y-%m-%d'),
                todate=datetime.strptime(end_date, '%Y-%m-%d')
            )
            cerebro.adddata(data)
            
            # Add strategy
            cerebro.addstrategy(strategy_class)
            
            # Set initial cash
            cerebro.broker.setcash(initial_cash)
            
            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            
            # Run backtest
            results = cerebro.run()
            
            # Extract results
            strat = results[0]
            sharpe = strat.analyzers.sharpe.get_analysis()
            drawdown = strat.analyzers.drawdown.get_analysis()
            returns = strat.analyzers.returns.get_analysis()
            
            return {
                'total_return': returns.get('rtot', 0) * 100,
                'sharpe_ratio': sharpe.get('sharperatio', 0),
                'max_drawdown': drawdown.get('max', {}).get('drawdown', 0),
                'final_value': cerebro.broker.getvalue(),
                'initial_cash': initial_cash
            }
        except Exception as e:
            return {'error': str(e)}
    
    def run_vectorbt_backtest(self, symbol, start_date, end_date, strategy_func, initial_cash=10000):
        """Run backtest using VectorBT"""
        try:
            # Download data
            data = yf.download(symbol, start=start_date, end=end_date)
            
            # Run strategy
            portfolio = vbt.Portfolio.from_signals(
                data['Close'],
                entries=strategy_func(data)['entries'],
                exits=strategy_func(data)['exits'],
                init_cash=initial_cash
            )
            
            return {
                'total_return': portfolio.total_return() * 100,
                'sharpe_ratio': portfolio.sharpe_ratio(),
                'max_drawdown': portfolio.max_drawdown() * 100,
                'final_value': portfolio.value().iloc[-1],
                'initial_cash': initial_cash
            }
        except Exception as e:
            return {'error': str(e)}

backtest_engine = BacktestEngine()

# =============================================================================
# TECHNICAL ANALYSIS INTEGRATION
# =============================================================================

class TechnicalAnalysis:
    def __init__(self):
        self.indicators = {}
    
    def calculate_indicators(self, data):
        """Calculate comprehensive technical indicators"""
        try:
            df = data.copy()
            
            # RSI
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['Close'])
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_middle'] = bb.bollinger_mavg()
            df['BB_lower'] = bb.bollinger_lband()
            
            # Moving Averages
            df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
            df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
            df['EMA_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
            df['EMA_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
            
            # Volume indicators
            df['Volume_SMA'] = ta.volume.VolumeSMAIndicator(df['Close'], df['Volume']).volume_sma()
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            # Williams %R
            df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
            
            return df
        except Exception as e:
            return {'error': str(e)}

tech_analysis = TechnicalAnalysis()

# =============================================================================
# ALPACA TRADING INTEGRATION
# =============================================================================

class AlpacaTrading:
    def __init__(self):
        self.api = None
        self.connected = False
    
    def connect(self, api_key, secret_key, base_url="https://paper-api.alpaca.markets"):
        """Connect to Alpaca API"""
        try:
            self.api = REST(api_key, secret_key, base_url, api_version='v2')
            # Test connection
            account = self.api.get_account()
            self.connected = True
            return {'status': 'success', 'account': account._raw}
        except Exception as e:
            return {'error': str(e)}
    
    def get_account(self):
        """Get account information"""
        if not self.connected:
            return {'error': 'Not connected to Alpaca'}
        try:
            account = self.api.get_account()
            return {
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'day_trade_count': account.day_trade_count
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_positions(self):
        """Get current positions"""
        if not self.connected:
            return {'error': 'Not connected to Alpaca'}
        try:
            positions = self.api.list_positions()
            return [{
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'current_price': float(pos.current_price)
            } for pos in positions]
        except Exception as e:
            return {'error': str(e)}
    
    def place_order(self, symbol, qty, side, order_type, time_in_force='day'):
        """Place a trade order"""
        if not self.connected:
            return {'error': 'Not connected to Alpaca'}
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )
            return {'status': 'success', 'order': order._raw}
        except Exception as e:
            return {'error': str(e)}

alpaca_trading = AlpacaTrading()

# =============================================================================
# STRATEGY BUILDING ENDPOINTS
# =============================================================================

@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """Get trading strategies for a user"""
    user_id = request.args.get('user_id')
    
    if not SUPABASE_ENABLED:
        return jsonify({'error': 'Supabase not available'}), 500
    
    try:
        result = supabase.table("trading_strategies").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return jsonify({'status': 'success', 'strategies': result.data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies', methods=['POST'])
def create_strategy():
    """Create a new trading strategy"""
    data = request.get_json() or {}
    
    if not SUPABASE_ENABLED:
        return jsonify({'error': 'Supabase not available'}), 500
    
    try:
        result = supabase.table("trading_strategies").insert(data).execute()
        return jsonify({'status': 'success', 'strategy': result.data[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtests', methods=['POST'])
def run_backtest():
    """Run a backtest for a strategy"""
    data = request.get_json() or {}
    
    symbol = data.get('symbol', 'AAPL')
    start_date = data.get('start_date', '2023-01-01')
    end_date = data.get('end_date', '2024-01-01')
    initial_capital = data.get('initial_capital', 10000)
    strategy_type = data.get('strategy_type', 'simple_moving_average')
    
    try:
        # Download data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            return jsonify({'error': 'No data available for the specified period'}), 400
        
        # Calculate technical indicators
        data_with_indicators = tech_analysis.calculate_indicators(data)
        
        # Simple moving average strategy
        if strategy_type == 'simple_moving_average':
            data_with_indicators['signal'] = 0
            data_with_indicators.loc[data_with_indicators['SMA_20'] > data_with_indicators['SMA_50'], 'signal'] = 1
            data_with_indicators.loc[data_with_indicators['SMA_20'] < data_with_indicators['SMA_50'], 'signal'] = -1
            
            # Calculate returns
            data_with_indicators['returns'] = data_with_indicators['Close'].pct_change()
            data_with_indicators['strategy_returns'] = data_with_indicators['signal'].shift(1) * data_with_indicators['returns']
            
            # Calculate performance metrics
            total_return = (1 + data_with_indicators['strategy_returns']).cumprod().iloc[-1] - 1
            sharpe_ratio = data_with_indicators['strategy_returns'].mean() / data_with_indicators['strategy_returns'].std() * np.sqrt(252)
            max_drawdown = (data_with_indicators['strategy_returns'].cumsum() - data_with_indicators['strategy_returns'].cumsum().expanding().max()).min()
            
            backtest_result = {
                'strategy_id': data.get('strategy_id'),
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'final_capital': initial_capital * (1 + total_return),
                'total_return': total_return * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown * 100,
                'win_rate': (data_with_indicators['strategy_returns'] > 0).mean() * 100,
                'total_trades': abs(data_with_indicators['signal'].diff()).sum() // 2
            }
        else:
            # Mock results for other strategies
            backtest_result = {
                'strategy_id': data.get('strategy_id'),
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'final_capital': random.uniform(8000, 15000),
                'total_return': random.uniform(-20, 50),
                'sharpe_ratio': random.uniform(0.5, 2.5),
                'max_drawdown': random.uniform(-30, -5),
                'win_rate': random.uniform(0.4, 0.8),
                'total_trades': random.randint(10, 100)
            }
        
        return jsonify({'status': 'success', 'backtest': backtest_result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# TECHNICAL ANALYSIS ENDPOINTS
# =============================================================================

@app.route('/api/technical-analysis/<symbol>', methods=['GET'])
def get_technical_analysis(symbol):
    """Get comprehensive technical analysis for a symbol"""
    try:
        # Download data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1y")
        
        if data.empty:
            return jsonify({'error': 'No data available for symbol'}), 400
        
        # Calculate indicators
        data_with_indicators = tech_analysis.calculate_indicators(data)
        
        # Get latest values
        latest = data_with_indicators.iloc[-1]
        
        analysis = {
            'symbol': symbol,
            'current_price': float(latest['Close']),
            'indicators': {
                'RSI': float(latest['RSI']) if not pd.isna(latest['RSI']) else None,
                'MACD': float(latest['MACD']) if not pd.isna(latest['MACD']) else None,
                'MACD_signal': float(latest['MACD_signal']) if not pd.isna(latest['MACD_signal']) else None,
                'BB_upper': float(latest['BB_upper']) if not pd.isna(latest['BB_upper']) else None,
                'BB_middle': float(latest['BB_middle']) if not pd.isna(latest['BB_middle']) else None,
                'BB_lower': float(latest['BB_lower']) if not pd.isna(latest['BB_lower']) else None,
                'SMA_20': float(latest['SMA_20']) if not pd.isna(latest['SMA_20']) else None,
                'SMA_50': float(latest['SMA_50']) if not pd.isna(latest['SMA_50']) else None,
                'Stoch_K': float(latest['Stoch_K']) if not pd.isna(latest['Stoch_K']) else None,
                'Stoch_D': float(latest['Stoch_D']) if not pd.isna(latest['Stoch_D']) else None,
                'Williams_R': float(latest['Williams_R']) if not pd.isna(latest['Williams_R']) else None
            },
            'signals': {
                'rsi_signal': 'oversold' if latest['RSI'] < 30 else 'overbought' if latest['RSI'] > 70 else 'neutral',
                'macd_signal': 'bullish' if latest['MACD'] > latest['MACD_signal'] else 'bearish',
                'bb_signal': 'overbought' if latest['Close'] > latest['BB_upper'] else 'oversold' if latest['Close'] < latest['BB_lower'] else 'neutral',
                'ma_signal': 'bullish' if latest['SMA_20'] > latest['SMA_50'] else 'bearish',
                'stoch_signal': 'oversold' if latest['Stoch_K'] < 20 else 'overbought' if latest['Stoch_K'] > 80 else 'neutral'
            }
        }
        
        return jsonify({'status': 'success', 'analysis': analysis})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ALPACA TRADING ENDPOINTS
# =============================================================================

@app.route('/api/alpaca/connect', methods=['POST'])
def connect_alpaca():
    """Connect to Alpaca API"""
    data = request.get_json() or {}
    api_key = data.get('api_key')
    secret_key = data.get('secret_key')
    paper_trading = data.get('paper_trading', True)
    
    if not api_key or not secret_key:
        return jsonify({'error': 'API key and secret key required'}), 400
    
    base_url = "https://paper-api.alpaca.markets" if paper_trading else "https://api.alpaca.markets"
    result = alpaca_trading.connect(api_key, secret_key, base_url)
    return jsonify(result)

@app.route('/api/alpaca/account', methods=['GET'])
def get_alpaca_account():
    """Get Alpaca account information"""
    result = alpaca_trading.get_account()
    return jsonify(result)

@app.route('/api/alpaca/positions', methods=['GET'])
def get_alpaca_positions():
    """Get Alpaca positions"""
    result = alpaca_trading.get_positions()
    return jsonify(result)

@app.route('/api/alpaca/orders', methods=['POST'])
def place_alpaca_order():
    """Place an order on Alpaca"""
    data = request.get_json() or {}
    symbol = data.get('symbol')
    qty = data.get('qty')
    side = data.get('side')
    order_type = data.get('order_type', 'market')
    time_in_force = data.get('time_in_force', 'day')
    
    if not all([symbol, qty, side]):
        return jsonify({'error': 'symbol, qty, and side required'}), 400
    
    result = alpaca_trading.place_order(symbol, qty, side, order_type, time_in_force)
    return jsonify(result)

# =============================================================================
# CRYPTO TRADING ENDPOINTS
# =============================================================================

@app.route('/api/crypto/exchanges', methods=['GET'])
def get_crypto_exchanges():
    """Get available crypto exchanges"""
    try:
        exchanges = ccxt.exchanges
        return jsonify({'status': 'success', 'exchanges': exchanges})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crypto/ticker/<exchange>/<symbol>', methods=['GET'])
def get_crypto_ticker(exchange, symbol):
    """Get crypto ticker data"""
    try:
        exchange_class = getattr(ccxt, exchange)
        exchange_instance = exchange_class()
        ticker = exchange_instance.fetch_ticker(symbol)
        return jsonify({'status': 'success', 'ticker': ticker})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# VISUALIZATION ENDPOINTS
# =============================================================================

@app.route('/api/charts/candlestick/<symbol>', methods=['GET'])
def get_candlestick_chart(symbol):
    """Generate candlestick chart data"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="6mo")
        
        if data.empty:
            return jsonify({'error': 'No data available'}), 400
        
        # Create candlestick chart
        fig = go.Figure(data=go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=symbol
        ))
        
        fig.update_layout(
            title=f'{symbol} Candlestick Chart',
            yaxis_title='Price',
            xaxis_title='Date'
        )
        
        return jsonify({'status': 'success', 'chart': fig.to_json()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/indicators/<symbol>', methods=['GET'])
def get_indicators_chart(symbol):
    """Generate technical indicators chart"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="6mo")
        
        if data.empty:
            return jsonify({'error': 'No data available'}), 400
        
        # Calculate indicators
        data_with_indicators = tech_analysis.calculate_indicators(data)
        
        # Create subplot with indicators
        fig = go.Figure()
        
        # Price chart
        fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['Close'], name='Close'))
        fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['SMA_20'], name='SMA 20'))
        fig.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['SMA_50'], name='SMA 50'))
        
        fig.update_layout(
            title=f'{symbol} Technical Indicators',
            yaxis_title='Price',
            xaxis_title='Date'
        )
        
        return jsonify({'status': 'success', 'chart': fig.to_json()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)