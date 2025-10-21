"""
Simple Trading Platform Backend
Supabase + Basic API Integration
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from supabase import create_client, Client
import hashlib
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
        
        return jsonify({
            'status': 'success',
            'backtest_id': 'mock_backtest_123',
            'results': mock_results
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
