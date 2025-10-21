from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from datetime import datetime

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

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'platform': 'vercel'
    })

# Root-level endpoints for frontend compatibility
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

# Vercel handler
def handler(request):
    return app(request.environ, lambda *args: None)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)