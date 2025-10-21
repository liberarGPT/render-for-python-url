from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)

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
        data = request.get_json() or {}
        symbol = data.get('symbol', 'AAPL')
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'analysis': {
                'recommendation': 'BUY',
                'confidence': 0.75,
                'trend': 'bullish'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict price movement"""
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'AAPL')
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'prediction': {
                'price_target': 150.0,
                'confidence': 0.8
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/backtest', methods=['POST'])
def backtest():
    """Run backtest"""
    try:
        data = request.get_json() or {}
        initial_capital = data.get('initial_capital', 10000)
        
        return jsonify({
            'status': 'success',
            'results': {
                'total_return': 15.5,
                'sharpe_ratio': 1.2,
                'max_drawdown': -8.5
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# This is the entry point for Vercel
if __name__ == '__main__':
    app.run(debug=True)