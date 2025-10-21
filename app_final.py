from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from datetime import datetime
import jwt

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={
    "/api/*": {"origins": ["https://tradespark.app"]},
    "/analyze": {"origins": ["https://tradespark.app"]},
    "/predict": {"origins": ["https://tradespark.app"]},
    "/backtest": {"origins": ["https://tradespark.app"]},
    "/health": {"origins": ["https://tradespark.app"]}
})

# Initialize variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_ANON_KEY')
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'supabase_configured': bool(SUPABASE_URL and SUPABASE_KEY),
        'timestamp': datetime.utcnow().isoformat()
    })

# Supabase test endpoint
@app.route('/api/supabase/test', methods=['GET'])
def test_supabase():
    """Test Supabase connection"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return jsonify({'error': 'Supabase not configured'}), 500
    
    return jsonify({
        'status': 'success',
        'message': 'Supabase credentials configured',
        'url': SUPABASE_URL[:50] + '...' if len(SUPABASE_URL) > 50 else SUPABASE_URL
    })

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
        
        # Mock user creation (replace with actual Supabase call)
        user_id = f"user_{datetime.now().timestamp()}"
        
        return jsonify({
            'status': 'success',
            'message': 'User registered successfully',
            'user': {
                'id': user_id,
                'email': email,
                'created_at': datetime.utcnow().isoformat()
            }
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
        
        # Mock authentication (replace with actual Supabase call)
        user_id = f"user_{hash(email) % 10000}"
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': user_id,
            'email': email,
            'exp': datetime.utcnow().timestamp() + 3600  # 1 hour
        }, JWT_SECRET, algorithm='HS256')
        
        return jsonify({
            'status': 'success',
            'message': 'Login successful',
            'user': {
                'id': user_id,
                'email': email
            },
            'token': token
        })
            
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
        
        # Mock API key storage (replace with actual Supabase call)
        key_id = f"key_{datetime.now().timestamp()}"
        
        return jsonify({
            'status': 'success',
            'message': 'API key added successfully',
            'data': {
                'id': key_id,
                'user_id': user_id,
                'provider': provider,
                'is_active': True,
                'created_at': datetime.utcnow().isoformat()
            }
        })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/keys/<user_id>', methods=['GET'])
def get_api_keys(user_id):
    """Get user API keys"""
    try:
        # Mock API keys (replace with actual Supabase call)
        mock_keys = [
            {
                'id': f"key_{user_id}_1",
                'provider': 'alpaca',
                'is_active': True,
                'created_at': datetime.utcnow().isoformat()
            },
            {
                'id': f"key_{user_id}_2", 
                'provider': 'fmp',
                'is_active': True,
                'created_at': datetime.utcnow().isoformat()
            }
        ]
        
        return jsonify({
            'status': 'success',
            'data': mock_keys
        })
            
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
        
        # Mock strategy storage (replace with actual Supabase call)
        strategy_id = f"strategy_{datetime.now().timestamp()}"
        
        return jsonify({
            'status': 'success',
            'message': 'Strategy saved successfully',
            'data': {
                'id': strategy_id,
                'user_id': user_id,
                'name': name,
                'description': description,
                'created_at': datetime.utcnow().isoformat()
            }
        })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies/<user_id>', methods=['GET'])
def get_strategies(user_id):
    """Get user strategies"""
    try:
        # Mock strategies (replace with actual Supabase call)
        mock_strategies = [
            {
                'id': f"strategy_{user_id}_1",
                'name': 'Moving Average Crossover',
                'description': 'Buy when short MA crosses above long MA',
                'created_at': datetime.utcnow().isoformat()
            },
            {
                'id': f"strategy_{user_id}_2",
                'name': 'RSI Strategy',
                'description': 'Buy when RSI < 30, sell when RSI > 70',
                'created_at': datetime.utcnow().isoformat()
            }
        ]
        
        return jsonify({
            'status': 'success',
            'data': mock_strategies
        })
            
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
        
        # Mock backtest results
        backtest_results = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.05,
            'win_rate': 0.65,
            'total_trades': 45,
            'final_capital': initial_capital * 1.15
        }
        
        # Mock backtest storage (replace with actual Supabase call)
        backtest_id = f"backtest_{datetime.now().timestamp()}"
        
        return jsonify({
            'status': 'success',
            'message': 'Backtest completed successfully',
            'results': backtest_results,
            'data': {
                'id': backtest_id,
                'user_id': user_id,
                'strategy_id': strategy_id,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'created_at': datetime.utcnow().isoformat()
            }
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
        
        # Mock session creation (replace with actual Supabase call)
        session_id = f"session_{datetime.now().timestamp()}"
        
        return jsonify({
            'status': 'success',
            'message': 'Chat session created',
            'data': {
                'id': session_id,
                'user_id': user_id,
                'session_name': session_name,
                'created_at': datetime.utcnow().isoformat()
            }
        })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/sessions/<user_id>', methods=['GET'])
def get_chat_sessions(user_id):
    """Get user chat sessions"""
    try:
        # Mock chat sessions (replace with actual Supabase call)
        mock_sessions = [
            {
                'id': f"session_{user_id}_1",
                'session_name': 'Trading Strategy Discussion',
                'created_at': datetime.utcnow().isoformat()
            },
            {
                'id': f"session_{user_id}_2",
                'session_name': 'Market Analysis Chat',
                'created_at': datetime.utcnow().isoformat()
            }
        ]
        
        return jsonify({
            'status': 'success',
            'data': mock_sessions
        })
            
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
        
        # Mock AI response
        ai_response = f"AI Response to: {message}. This is a mock response that would be replaced with actual AI integration."
        
        return jsonify({
            'status': 'success',
            'message': 'Message sent successfully',
            'ai_response': ai_response,
            'data': {
                'session_id': session_id,
                'user_message': message,
                'ai_response': ai_response,
                'timestamp': datetime.utcnow().isoformat()
            }
        })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Root-level endpoints for frontend compatibility
@app.route('/analyze', methods=['POST'])
def analyze_root():
    """Analyze market data - root endpoint for frontend compatibility"""
    return analyze()

@app.route('/predict', methods=['POST'])
def predict_root():
    """Predict price movement - root endpoint for frontend compatibility"""
    return predict()

@app.route('/backtest', methods=['POST'])
def backtest_root():
    """Run backtest - root endpoint for frontend compatibility"""
    return run_backtest()

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
                    'moving_average': 'above',
                    'support_level': 140.00,
                    'resistance_level': 160.00
                },
                'fundamental_analysis': {
                    'pe_ratio': 25.5,
                    'revenue_growth': 0.08,
                    'profit_margin': 0.22
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
                'model_used': 'LSTM Neural Network',
                'factors': [
                    'Technical indicators',
                    'Market sentiment',
                    'Volume analysis',
                    'Historical patterns'
                ]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# OpenBB integration endpoints
@app.route('/api/openbb/data', methods=['POST'])
def get_openbb_data():
    """Get OpenBB market data"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        data_type = data.get('data_type', 'stock_data')
        
        # Mock OpenBB data (replace with actual OpenBB integration)
        mock_data = {
            'symbol': symbol,
            'data_type': data_type,
            'price': 145.50,
            'change': 2.30,
            'change_percent': 1.60,
            'volume': 45000000,
            'market_cap': 2300000000000,
            'pe_ratio': 25.5,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'data': mock_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# MCP integration endpoints
@app.route('/api/mcp/sessions', methods=['POST'])
def create_mcp_session():
    """Create MCP session"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        session_name = data.get('session_name', 'MCP Session')
        context_data = data.get('context_data', {})
        
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        # Mock MCP session creation
        session_id = f"mcp_session_{datetime.now().timestamp()}"
        
        return jsonify({
            'status': 'success',
            'message': 'MCP session created',
            'data': {
                'id': session_id,
                'user_id': user_id,
                'session_name': session_name,
                'context_data': context_data,
                'is_active': True,
                'created_at': datetime.utcnow().isoformat()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mcp/tools', methods=['POST'])
def use_mcp_tool():
    """Use MCP tool"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        tool_name = data.get('tool_name')
        parameters = data.get('parameters', {})
        
        if not all([session_id, tool_name]):
            return jsonify({'error': 'Session ID and tool name required'}), 400
        
        # Mock MCP tool usage
        result = {
            'tool_name': tool_name,
            'parameters': parameters,
            'result': f"Mock result from {tool_name}",
            'execution_time_ms': 150,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'message': 'MCP tool executed successfully',
            'data': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
