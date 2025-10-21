"""
Comprehensive Trading Platform Backend
Supabase + OpenBB + MCP Integration
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import asyncio
from sqlalchemy import create_engine
from supabase import create_client, Client
import hashlib
import secrets
import jwt
from functools import wraps

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={"/api/*": {"origins": ["https://tradespark.app"]}})

# Initialize Supabase client
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_ANON_KEY')
supabase: Client = None

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# JWT Secret
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')

# Authentication decorator
def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            current_user_id = data['user_id']
        except:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user_id, *args, **kwargs)
    return decorated_function

# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        first_name = data.get('first_name', '')
        last_name = data.get('last_name', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        # Hash password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Check if user exists
        existing_user = supabase.table('users').select('id').eq('email', email).execute()
        if existing_user.data:
            return jsonify({'error': 'User already exists'}), 400
        
        # Create user
        user_data = {
            'email': email,
            'password_hash': password_hash,
            'first_name': first_name,
            'last_name': last_name
        }
        
        result = supabase.table('users').insert(user_data).execute()
        user_id = result.data[0]['id']
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': user_id,
            'email': email,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, JWT_SECRET, algorithm='HS256')
        
        return jsonify({
            'status': 'success',
            'message': 'User registered successfully',
            'token': token,
            'user_id': user_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        # Hash password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Check user credentials
        result = supabase.table('users').select('*').eq('email', email).eq('password_hash', password_hash).execute()
        
        if not result.data:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        user = result.data[0]
        
        # Update last login
        supabase.table('users').update({'last_login': datetime.utcnow().isoformat()}).eq('id', user['id']).execute()
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': user['id'],
            'email': user['email'],
            'exp': datetime.utcnow() + timedelta(days=30)
        }, JWT_SECRET, algorithm='HS256')
        
        return jsonify({
            'status': 'success',
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'first_name': user['first_name'],
                'last_name': user['last_name'],
                'subscription_status': user['subscription_status']
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# USER MANAGEMENT ENDPOINTS
# =============================================================================

@app.route('/api/user/profile', methods=['GET'])
@require_auth
def get_user_profile(user_id):
    """Get user profile"""
    try:
        result = supabase.table('users').select('*').eq('id', user_id).execute()
        if not result.data:
            return jsonify({'error': 'User not found'}), 404
        
        user = result.data[0]
        del user['password_hash']  # Remove sensitive data
        
        return jsonify({
            'status': 'success',
            'user': user
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/profile', methods=['PUT'])
@require_auth
def update_user_profile(user_id):
    """Update user profile"""
    try:
        data = request.get_json()
        update_data = {}
        
        if 'first_name' in data:
            update_data['first_name'] = data['first_name']
        if 'last_name' in data:
            update_data['last_name'] = data['last_name']
        
        if update_data:
            update_data['updated_at'] = datetime.utcnow().isoformat()
            result = supabase.table('users').update(update_data).eq('id', user_id).execute()
        
        return jsonify({
            'status': 'success',
            'message': 'Profile updated successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# API KEYS MANAGEMENT
# =============================================================================

@app.route('/api/settings/api-keys', methods=['GET'])
@require_auth
def get_api_keys(user_id):
    """Get user API keys"""
    try:
        # Get broker API keys
        broker_keys = supabase.table('user_api_keys').select('*').eq('user_id', user_id).execute()
        
        # Get OpenBB API keys
        openbb_keys = supabase.table('openbb_api_keys').select('*').eq('user_id', user_id).execute()
        
        return jsonify({
            'status': 'success',
            'broker_keys': broker_keys.data,
            'openbb_keys': openbb_keys.data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings/api-keys', methods=['POST'])
@require_auth
def add_api_key(user_id):
    """Add API key"""
    try:
        data = request.get_json()
        key_type = data.get('type')  # 'broker' or 'openbb'
        broker = data.get('broker')
        api_key = data.get('api_key')
        secret_key = data.get('secret_key')
        pat_token = data.get('pat_token')
        
        if key_type == 'broker':
            key_data = {
                'user_id': user_id,
                'broker': broker,
                'api_key': api_key,
                'secret_key': secret_key
            }
            result = supabase.table('user_api_keys').insert(key_data).execute()
        elif key_type == 'openbb':
            key_data = {
                'user_id': user_id,
                'provider': broker,
                'api_key': api_key,
                'pat_token': pat_token
            }
            result = supabase.table('openbb_api_keys').insert(key_data).execute()
        else:
            return jsonify({'error': 'Invalid key type'}), 400
        
        return jsonify({
            'status': 'success',
            'message': 'API key added successfully',
            'key_id': result.data[0]['id']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings/api-keys/<key_id>', methods=['DELETE'])
@require_auth
def delete_api_key(user_id, key_id):
    """Delete API key"""
    try:
        # Try broker keys first
        result = supabase.table('user_api_keys').delete().eq('id', key_id).eq('user_id', user_id).execute()
        
        if not result.data:
            # Try OpenBB keys
            result = supabase.table('openbb_api_keys').delete().eq('id', key_id).eq('user_id', user_id).execute()
        
        return jsonify({
            'status': 'success',
            'message': 'API key deleted successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# MCP SESSION MANAGEMENT
# =============================================================================

@app.route('/api/mcp/session', methods=['POST'])
@require_auth
def create_mcp_session(user_id):
    """Create new MCP session"""
    try:
        data = request.get_json()
        session_name = data.get('session_name', f'Session {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        
        session_data = {
            'user_id': user_id,
            'session_name': session_name,
            'context_data': {},
            'is_active': True
        }
        
        result = supabase.table('mcp_sessions').insert(session_data).execute()
        
        return jsonify({
            'status': 'success',
            'message': 'MCP session created',
            'session_id': result.data[0]['id']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mcp/sessions', methods=['GET'])
@require_auth
def get_mcp_sessions(user_id):
    """Get user MCP sessions"""
    try:
        result = supabase.table('mcp_sessions').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
        
        return jsonify({
            'status': 'success',
            'sessions': result.data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mcp/execute', methods=['POST'])
@require_auth
def execute_mcp_tool(user_id):
    """Execute MCP tool"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        tool_name = data.get('tool_name')
        parameters = data.get('parameters', {})
        
        # Record tool usage
        tool_usage_data = {
            'session_id': session_id,
            'tool_name': tool_name,
            'parameters': parameters,
            'result': {'status': 'executed'},
            'execution_time_ms': 0
        }
        
        result = supabase.table('mcp_tool_usage').insert(tool_usage_data).execute()
        
        # Simulate tool execution (replace with actual MCP tool calls)
        if tool_name == 'get_stock_data':
            symbol = parameters.get('symbol', 'AAPL')
            # This would call OpenBB to get stock data
            mock_result = {
                'symbol': symbol,
                'price': 150.25,
                'change': 2.15,
                'change_percent': 1.45
            }
        elif tool_name == 'technical_analysis':
            symbol = parameters.get('symbol', 'AAPL')
            mock_result = {
                'symbol': symbol,
                'rsi': 45.2,
                'macd': 0.15,
                'bollinger_upper': 155.30,
                'bollinger_lower': 145.20
            }
        else:
            mock_result = {'message': f'Tool {tool_name} executed successfully'}
        
        # Update tool usage with actual result
        supabase.table('mcp_tool_usage').update({
            'result': mock_result,
            'execution_time_ms': 150
        }).eq('id', result.data[0]['id']).execute()
        
        return jsonify({
            'status': 'success',
            'result': mock_result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# CHAT ENDPOINTS
# =============================================================================

@app.route('/api/chat/sessions', methods=['GET'])
@require_auth
def get_chat_sessions(user_id):
    """Get user chat sessions"""
    try:
        result = supabase.table('chat_sessions').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
        
        return jsonify({
            'status': 'success',
            'sessions': result.data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/session', methods=['POST'])
@require_auth
def create_chat_session(user_id):
    """Create new chat session"""
    try:
        data = request.get_json()
        session_name = data.get('session_name', f'Chat {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        mcp_session_id = data.get('mcp_session_id')
        
        session_data = {
            'user_id': user_id,
            'session_name': session_name,
            'mcp_session_id': mcp_session_id
        }
        
        result = supabase.table('chat_sessions').insert(session_data).execute()
        
        return jsonify({
            'status': 'success',
            'session_id': result.data[0]['id']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/message', methods=['POST'])
@require_auth
def send_chat_message(user_id):
    """Send chat message"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        message = data.get('message')
        
        # Save user message
        user_message_data = {
            'session_id': session_id,
            'role': 'user',
            'content': message
        }
        supabase.table('chat_messages').insert(user_message_data).execute()
        
        # Generate AI response (this would integrate with your AI model)
        ai_response = f"I received your message: '{message}'. This is a placeholder response. In a real implementation, this would connect to your AI model."
        
        # Save AI response
        ai_message_data = {
            'session_id': session_id,
            'role': 'assistant',
            'content': ai_response
        }
        supabase.table('chat_messages').insert(ai_message_data).execute()
        
        return jsonify({
            'status': 'success',
            'response': ai_response
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# TRADING ENDPOINTS
# =============================================================================

@app.route('/api/trades/place', methods=['POST'])
@require_auth
def place_trade(user_id):
    """Place a trade"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        action = data.get('action')  # 'buy' or 'sell'
        quantity = data.get('quantity')
        price = data.get('price')
        portfolio_id = data.get('portfolio_id')
        
        trade_data = {
            'user_id': user_id,
            'portfolio_id': portfolio_id,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'status': 'completed'
        }
        
        result = supabase.table('trades').insert(trade_data).execute()
        
        return jsonify({
            'status': 'success',
            'message': 'Trade placed successfully',
            'trade_id': result.data[0]['id']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trades/history', methods=['GET'])
@require_auth
def get_trade_history(user_id):
    """Get trading history"""
    try:
        result = supabase.table('trades').select('*').eq('user_id', user_id).order('timestamp', desc=True).execute()
        
        return jsonify({
            'status': 'success',
            'trades': result.data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# PORTFOLIO ENDPOINTS
# =============================================================================

@app.route('/api/portfolios', methods=['GET'])
@require_auth
def get_portfolios(user_id):
    """Get user portfolios"""
    try:
        result = supabase.table('portfolios').select('*').eq('user_id', user_id).execute()
        
        return jsonify({
            'status': 'success',
            'portfolios': result.data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolios', methods=['POST'])
@require_auth
def create_portfolio(user_id):
    """Create new portfolio"""
    try:
        data = request.get_json()
        name = data.get('name')
        description = data.get('description', '')
        
        portfolio_data = {
            'user_id': user_id,
            'name': name,
            'description': description
        }
        
        result = supabase.table('portfolios').insert(portfolio_data).execute()
        
        return jsonify({
            'status': 'success',
            'portfolio_id': result.data[0]['id']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# STRATEGY ENDPOINTS
# =============================================================================

@app.route('/api/strategies', methods=['GET'])
@require_auth
def get_strategies(user_id):
    """Get user strategies"""
    try:
        result = supabase.table('saved_strategies').select('*').eq('user_id', user_id).execute()
        
        return jsonify({
            'status': 'success',
            'strategies': result.data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies', methods=['POST'])
@require_auth
def save_strategy(user_id):
    """Save strategy"""
    try:
        data = request.get_json()
        name = data.get('name')
        description = data.get('description', '')
        strategy_config = data.get('strategy_config', {})
        is_public = data.get('is_public', False)
        
        strategy_data = {
            'user_id': user_id,
            'name': name,
            'description': description,
            'strategy_config': strategy_config,
            'is_public': is_public
        }
        
        result = supabase.table('saved_strategies').insert(strategy_data).execute()
        
        return jsonify({
            'status': 'success',
            'strategy_id': result.data[0]['id']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# BACKTESTING ENDPOINTS
# =============================================================================

@app.route('/api/backtest', methods=['POST'])
@require_auth
def run_backtest(user_id):
    """Run backtest"""
    try:
        data = request.get_json()
        strategy_id = data.get('strategy_id')
        symbols = data.get('symbols', ['AAPL'])
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        initial_capital = data.get('initial_capital', 10000)
        
        # This would run actual backtesting with backtrader/vectorbt
        # For now, return mock results
        mock_results = {
            'total_return': 15.5,
            'annual_return': 12.3,
            'volatility': 18.2,
            'sharpe_ratio': 0.85,
            'max_drawdown': -8.5,
            'win_rate': 65.0
        }
        
        # Save backtest results
        backtest_data = {
            'user_id': user_id,
            'strategy_id': strategy_id,
            'name': f'Backtest {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'final_value': initial_capital * (1 + mock_results['total_return'] / 100),
            'total_return': mock_results['total_return'],
            'annual_return': mock_results['annual_return'],
            'volatility': mock_results['volatility'],
            'sharpe_ratio': mock_results['sharpe_ratio'],
            'max_drawdown': mock_results['max_drawdown'],
            'win_rate': mock_results['win_rate'],
            'strategy_parameters': {},
            'backtest_engine': 'backtrader'
        }
        
        result = supabase.table('backtest_results').insert(backtest_data).execute()
        
        return jsonify({
            'status': 'success',
            'backtest_id': result.data[0]['id'],
            'results': mock_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ANALYSIS ENDPOINTS
# =============================================================================

@app.route('/api/analyze', methods=['POST'])
@require_auth
def analyze_market(user_id):
    """Analyze market data"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        analysis_type = data.get('analysis_type', 'technical')
        
        # This would perform actual analysis using the trading agents
        analysis_result = {
            'symbol': symbol,
            'analysis_type': analysis_type,
            'recommendation': 'BUY',
            'confidence': 0.75,
            'key_insights': [
                'Strong technical indicators',
                'Positive momentum',
                'Good risk/reward ratio'
            ],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'analysis': analysis_result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
@require_auth
def predict_price(user_id):
    """Get price prediction"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        timeframe = data.get('timeframe', '1d')
        
        # This would use ML models for prediction
        prediction_result = {
            'symbol': symbol,
            'current_price': 150.25,
            'predicted_price': 155.80,
            'confidence': 0.68,
            'timeframe': timeframe,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'prediction': prediction_result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'supabase_connected': supabase is not None,
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/supabase/test', methods=['GET'])
def test_supabase():
    """Test Supabase connection"""
    if not supabase:
        return jsonify({'error': 'Supabase not configured'}), 500
    
    try:
        # Test connection by getting a simple query
        result = supabase.table('users').select('*').limit(1).execute()
        return jsonify({
            'status': 'success',
            'message': 'Supabase connected successfully',
            'data': result.data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
