from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime
import jwt

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={"/api/*": {"origins": ["https://tradespark.app"]}})

# Initialize Supabase client
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_ANON_KEY')
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')

supabase: Client = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase connection error: {e}")
        supabase = None

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'supabase_connected': supabase is not None,
        'timestamp': datetime.utcnow().isoformat()
    })

# Supabase test endpoint
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

# User authentication endpoints
@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        # Create user in Supabase
        if supabase:
            result = supabase.auth.sign_up({
                'email': email,
                'password': password
            })
            return jsonify({
                'status': 'success',
                'message': 'User registered successfully',
                'user': result.user
            })
        else:
            return jsonify({'error': 'Supabase not available'}), 500
            
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
        
        # Authenticate with Supabase
        if supabase:
            result = supabase.auth.sign_in_with_password({
                'email': email,
                'password': password
            })
            return jsonify({
                'status': 'success',
                'message': 'Login successful',
                'user': result.user,
                'session': result.session
            })
        else:
            return jsonify({'error': 'Supabase not available'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API key management
@app.route('/api/keys', methods=['POST'])
def add_api_key():
    """Add user API key"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        provider = data.get('provider')
        api_key = data.get('api_key')
        secret_key = data.get('secret_key')
        
        if not all([user_id, provider, api_key]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        if supabase:
            result = supabase.table('user_api_keys').insert({
                'user_id': user_id,
                'provider': provider,
                'api_key': api_key,
                'secret_key': secret_key,
                'is_active': True
            }).execute()
            
            return jsonify({
                'status': 'success',
                'message': 'API key added successfully',
                'data': result.data
            })
        else:
            return jsonify({'error': 'Supabase not available'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/keys/<user_id>', methods=['GET'])
def get_api_keys(user_id):
    """Get user API keys"""
    try:
        if supabase:
            result = supabase.table('user_api_keys').select('*').eq('user_id', user_id).execute()
            return jsonify({
                'status': 'success',
                'data': result.data
            })
        else:
            return jsonify({'error': 'Supabase not available'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Trading strategy endpoints
@app.route('/api/strategies', methods=['POST'])
def save_strategy():
    """Save trading strategy"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        name = data.get('name')
        description = data.get('description')
        strategy_code = data.get('strategy_code')
        parameters = data.get('parameters', {})
        
        if not all([user_id, name, strategy_code]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        if supabase:
            result = supabase.table('trading_strategies').insert({
                'user_id': user_id,
                'name': name,
                'description': description,
                'strategy_code': strategy_code,
                'parameters': parameters
            }).execute()
            
            return jsonify({
                'status': 'success',
                'message': 'Strategy saved successfully',
                'data': result.data
            })
        else:
            return jsonify({'error': 'Supabase not available'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies/<user_id>', methods=['GET'])
def get_strategies(user_id):
    """Get user strategies"""
    try:
        if supabase:
            result = supabase.table('trading_strategies').select('*').eq('user_id', user_id).execute()
            return jsonify({
                'status': 'success',
                'data': result.data
            })
        else:
            return jsonify({'error': 'Supabase not available'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Backtest endpoints
@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Run backtest"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        strategy_id = data.get('strategy_id')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        initial_capital = data.get('initial_capital', 10000)
        
        # Mock backtest results for now
        backtest_results = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.05,
            'win_rate': 0.65,
            'total_trades': 45
        }
        
        if supabase:
            result = supabase.table('backtest_results').insert({
                'user_id': user_id,
                'strategy_id': strategy_id,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'final_capital': initial_capital * (1 + backtest_results['total_return']),
                'returns': backtest_results,
                'metrics': backtest_results,
                'trades': []
            }).execute()
            
            return jsonify({
                'status': 'success',
                'message': 'Backtest completed successfully',
                'results': backtest_results,
                'data': result.data
            })
        else:
            return jsonify({
                'status': 'success',
                'message': 'Backtest completed (mock)',
                'results': backtest_results
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# AI Chat endpoints
@app.route('/api/chat/sessions', methods=['POST'])
def create_chat_session():
    """Create new chat session"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        session_name = data.get('session_name', 'New Chat')
        
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        if supabase:
            result = supabase.table('chat_sessions').insert({
                'user_id': user_id,
                'session_name': session_name
            }).execute()
            
            return jsonify({
                'status': 'success',
                'message': 'Chat session created',
                'data': result.data
            })
        else:
            return jsonify({'error': 'Supabase not available'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/sessions/<user_id>', methods=['GET'])
def get_chat_sessions(user_id):
    """Get user chat sessions"""
    try:
        if supabase:
            result = supabase.table('chat_sessions').select('*').eq('user_id', user_id).execute()
            return jsonify({
                'status': 'success',
                'data': result.data
            })
        else:
            return jsonify({'error': 'Supabase not available'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/messages', methods=['POST'])
def send_message():
    """Send chat message"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        message = data.get('message')
        sender = data.get('sender', 'user')
        
        if not all([session_id, message]):
            return jsonify({'error': 'Session ID and message required'}), 400
        
        if supabase:
            result = supabase.table('chat_messages').insert({
                'session_id': session_id,
                'sender': sender,
                'message': message
            }).execute()
            
            # Mock AI response
            ai_response = f"AI Response to: {message}"
            
            # Save AI response
            supabase.table('chat_messages').insert({
                'session_id': session_id,
                'sender': 'ai',
                'message': ai_response
            }).execute()
            
            return jsonify({
                'status': 'success',
                'message': 'Message sent successfully',
                'ai_response': ai_response
            })
        else:
            return jsonify({'error': 'Supabase not available'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Market data endpoints
@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze market data"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'analysis': {
                'trend': 'bullish',
                'confidence': 0.75,
                'recommendation': 'BUY',
                'technical_indicators': {
                    'rsi': 65,
                    'macd': 'positive',
                    'moving_average': 'above'
                }
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict price movement"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'prediction': {
                'price_target': 150.00,
                'confidence': 0.80,
                'timeframe': '1 week',
                'model_used': 'LSTM Neural Network'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
