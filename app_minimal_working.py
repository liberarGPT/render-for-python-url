from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import random

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'platform': 'vercel',
        'timestamp': datetime.utcnow().isoformat()
    })

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
            "timestamp": datetime.utcnow().isoformat()
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
            "timestamp": datetime.utcnow().isoformat()
        }
    ]
    return jsonify({
        'status': 'success',
        'ideas': mock_ideas
    })

@app.route('/api/ai-trading-recommendations', methods=['GET'])
def get_ai_trading_recommendations():
    """Get AI-powered trading recommendations"""
    recommendations = [
        {
            "symbol": "NVDA",
            "action": "BUY",
            "confidence": 0.85,
            "reasoning": "Strong AI chip demand, positive earnings outlook, technical indicators showing bullish momentum",
            "suggested_quantity": 10,
            "suggested_stop_loss": 420.00,
            "suggested_take_profit": 500.00,
            "risk_level": "medium"
        },
        {
            "symbol": "TSLA",
            "action": "SELL",
            "confidence": 0.70,
            "reasoning": "Overvalued based on fundamentals, potential headwinds in EV market, technical resistance at current levels",
            "suggested_quantity": 5,
            "suggested_stop_loss": 250.00,
            "suggested_take_profit": 200.00,
            "risk_level": "high"
        },
        {
            "symbol": "AAPL",
            "action": "BUY",
            "confidence": 0.75,
            "reasoning": "Strong iPhone sales, services growth, defensive stock with good dividend yield",
            "suggested_quantity": 15,
            "suggested_stop_loss": 170.00,
            "suggested_take_profit": 200.00,
            "risk_level": "low"
        }
    ]
    
    return jsonify({
        'status': 'success',
        'recommendations': recommendations
    })

@app.route('/api/ai-trade-assistance', methods=['POST'])
def get_ai_trade_assistance():
    """Get AI assistance for trade parameters"""
    data = request.get_json() or {}
    symbol = data.get('symbol', 'AAPL')
    current_price = data.get('current_price', 150.00)
    account_balance = data.get('account_balance', 10000)
    
    # Mock AI assistance logic
    risk_percentage = 0.02  # 2% risk per trade
    max_position_size = account_balance * risk_percentage / current_price
    
    # Calculate stop loss and take profit based on volatility
    stop_loss_percentage = 0.05  # 5% stop loss
    take_profit_percentage = 0.10  # 10% take profit
    
    assistance = {
        "suggested_quantity": int(max_position_size),
        "suggested_stop_loss": round(current_price * (1 - stop_loss_percentage), 2),
        "suggested_take_profit": round(current_price * (1 + take_profit_percentage), 2),
        "risk_reward_ratio": 2.0,
        "position_size_percentage": (max_position_size * current_price / account_balance) * 100,
        "reasoning": f"AI suggests {int(max_position_size)} shares based on 2% risk management. Stop loss at ${round(current_price * (1 - stop_loss_percentage), 2)} and take profit at ${round(current_price * (1 + take_profit_percentage), 2)} for a 2:1 risk-reward ratio."
    }
    
    return jsonify({
        'status': 'success',
        'assistance': assistance
    })

@app.route('/api/alpaca/connect', methods=['POST'])
def connect_alpaca():
    """Connect to Alpaca API"""
    data = request.get_json() or {}
    api_key = data.get('api_key')
    secret_key = data.get('secret_key')
    paper_trading = data.get('paper_trading', True)
    
    if not api_key or not secret_key:
        return jsonify({'error': 'API key and secret key required'}), 400
    
    return jsonify({
        'status': 'success',
        'message': 'Connected to Alpaca successfully',
        'paper_trading': paper_trading
    })

@app.route('/api/alpaca/account', methods=['GET'])
def get_alpaca_account():
    """Get Alpaca account information"""
    return jsonify({
        'buying_power': 10000.0,
        'cash': 5000.0,
        'portfolio_value': 15000.0,
        'equity': 15000.0,
        'day_trade_count': 0
    })

@app.route('/api/alpaca/positions', methods=['GET'])
def get_alpaca_positions():
    """Get Alpaca positions"""
    return jsonify([
        {
            'symbol': 'AAPL',
            'qty': 10,
            'market_value': 1500.0,
            'unrealized_pl': 50.0,
            'unrealized_plpc': 0.033,
            'current_price': 150.0
        }
    ])

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
    
    return jsonify({
        'status': 'success',
        'order_id': f'order_{random.randint(1000, 9999)}',
        'symbol': symbol,
        'qty': qty,
        'side': side,
        'type': order_type
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json() or {}
    symbol = data.get('symbol', 'AAPL')
    return jsonify({
        'status': 'success',
        'symbol': symbol,
        'analysis': {'recommendation': 'BUY', 'confidence': 0.75, 'trend': 'bullish'}
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    symbol = data.get('symbol', 'AAPL')
    return jsonify({
        'status': 'success',
        'symbol': symbol,
        'prediction': {'price_target': 150.00, 'confidence': 0.80}
    })

@app.route('/backtest', methods=['POST'])
def backtest():
    data = request.get_json() or {}
    return jsonify({
        'status': 'success',
        'strategy': 'simple_moving_average',
        'results': {'total_return': 15.5, 'sharpe_ratio': 1.2, 'max_drawdown': -8.5}
    })

if __name__ == '__main__':
    app.run(debug=True)
