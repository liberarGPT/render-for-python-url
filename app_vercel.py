from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime
import jwt
import time

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

# Initialize Supabase client
supabase: Client = None
supabase_configured = False
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        supabase_configured = True
    except Exception as e:
        print(f"Supabase client initialization failed: {e}")
        supabase_configured = False

# Helper function for authentication
def require_auth(f):
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Authorization header missing'}), 401
        
        try:
            token = auth_header.split(' ')[1]
            payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            request.user_id = payload['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        except Exception as e:
            return jsonify({'error': f'Authentication error: {str(e)}'}), 401
        return f(*args, **kwargs)
    decorated.__name__ = f.__name__
    return decorated

# =============================================================================
# HEALTH CHECK & SUPABASE TEST
# =============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'supabase_configured': supabase_configured,
        'timestamp': datetime.utcnow().isoformat(),
        'platform': 'vercel'
    })

@app.route('/api/supabase/test', methods=['GET'])
def test_supabase():
    """Test Supabase connection"""
    if not supabase_configured or not supabase:
        return jsonify({'error': 'Supabase not configured or failed to connect'}), 500
    
    try:
        result = supabase.table('users').select('*').limit(1).execute()
        return jsonify({
            'status': 'success',
            'message': 'Supabase connected successfully',
            'data': result.data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

@app.route('/api/auth/register', methods=['POST'])
def register_user():
    """Register a new user"""
    if not supabase_configured or not supabase:
        return jsonify({'error': 'Supabase not configured'}), 500
    
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    try:
        user_id = f"user_{time.time()}".replace('.', '')
        new_user = {
            'id': user_id,
            'email': email,
            'hashed_password': 'mock_hashed_password',
            'created_at': datetime.utcnow().isoformat()
        }
        
        result = supabase.table('users').insert(new_user).execute()
        
        return jsonify({
            'status': 'success',
            'message': 'User registered successfully',
            'user': result.data[0]
        }), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login_user():
    """Login a user and return JWT token"""
    if not supabase_configured or not supabase:
        return jsonify({'error': 'Supabase not configured'}), 500
    
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    try:
        result = supabase.table('users').select('*').eq('email', email).limit(1).execute()
        user = result.data[0] if result.data else None

        if user and user['hashed_password'] == 'mock_hashed_password':
            payload = {
                'user_id': user['id'],
                'exp': datetime.utcnow().timestamp() + 3600
            }
            token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
            
            return jsonify({
                'status': 'success',
                'message': 'Logged in successfully',
                'access_token': token,
                'token_type': 'bearer'
            })
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ROOT-LEVEL ENDPOINTS FOR FRONTEND COMPATIBILITY
# =============================================================================

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze market data"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        
        # Mock analysis result
        analysis_result = {
            'symbol': symbol,
            'analysis_type': 'technical',
            'recommendation': 'BUY',
            'confidence': 0.75,
            'technical_indicators': {
                'rsi': 65,
                'macd': 'positive',
                'moving_average': 'above',
                'support_level': 140.0,
                'resistance_level': 160.0
            },
            'fundamental_analysis': {
                'pe_ratio': 25.5,
                'profit_margin': 0.22,
                'revenue_growth': 0.08
            },
            'trend': 'bullish',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'analysis': analysis_result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict price movement"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL')
        timeframe = data.get('timeframe', '1 week')
        
        # Mock prediction result
        prediction_result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'price_target': 150.0,
            'confidence': 0.8,
            'model_used': 'LSTM Neural Network',
            'factors': [
                'Technical indicators',
                'Market sentiment',
                'Volume analysis',
                'Historical patterns'
            ],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'prediction': prediction_result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/backtest', methods=['POST'])
def backtest():
    """Run backtest"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'user123')
        strategy_id = data.get('strategy_id', 'strategy456')
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', '2023-12-31')
        initial_capital = data.get('initial_capital', 10000)
        
        # Mock backtest results
        mock_results = {
            'total_return': 15.5,
            'annual_return': 12.3,
            'volatility': 18.2,
            'sharpe_ratio': 0.85,
            'max_drawdown': -8.5,
            'win_rate': 65.0
        }
        
        backtest_data = {
            'user_id': user_id,
            'strategy_id': strategy_id,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'final_capital': initial_capital * (1 + mock_results['total_return'] / 100),
            'total_return': mock_results['total_return'],
            'annual_return': mock_results['annual_return'],
            'volatility': mock_results['volatility'],
            'sharpe_ratio': mock_results['sharpe_ratio'],
            'max_drawdown': mock_results['max_drawdown'],
            'win_rate': mock_results['win_rate'],
            'total_trades': 45,
            'created_at': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'message': 'Backtest completed successfully',
            'data': backtest_data,
            'results': {
                'final_capital': backtest_data['final_capital'],
                'total_return': mock_results['total_return'],
                'sharpe_ratio': mock_results['sharpe_ratio'],
                'max_drawdown': mock_results['max_drawdown'],
                'win_rate': mock_results['win_rate']
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# API ENDPOINTS (for future use)
# =============================================================================

@app.route('/api/analyze', methods=['POST'])
@require_auth
def analyze_api():
    """Analyze market data - API version"""
    return analyze()

@app.route('/api/predict', methods=['POST'])
@require_auth
def predict_api():
    """Predict price movement - API version"""
    return predict()

@app.route('/api/backtest', methods=['POST'])
@require_auth
def backtest_api():
    """Run backtest - API version"""
    return backtest()

# Vercel handler
def handler(request):
    return app(request.environ, lambda *args: None)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
