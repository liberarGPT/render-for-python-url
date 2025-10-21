from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={"/api/*": {"origins": ["https://tradespark.app"]}})

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })

# Basic API endpoints
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
                'recommendation': 'BUY'
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
                'timeframe': '1 week'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def backtest():
    """Run backtest"""
    try:
        data = request.get_json()
        strategy = data.get('strategy', 'simple_moving_average')
        
        return jsonify({
            'status': 'success',
            'strategy': strategy,
            'results': {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.05
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
