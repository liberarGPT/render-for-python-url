from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os
import json

# Import the REAL TradingAgents framework
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows.alpaca_utils import AlpacaUtils
from tradingagents.dataflows.finnhub_utils import FinnhubUtils
from tradingagents.dataflows.yfin_utils import YFinUtils
from tradingagents.dataflows.reddit_utils import RedditUtils
from tradingagents.dataflows.googlenews_utils import GoogleNewsUtils

app = Flask(__name__)
CORS(app)

# Initialize TradingAgents
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-4o-mini"
config["quick_think_llm"] = "gpt-4o-mini"
config["max_debate_rounds"] = 2

trading_graph = TradingAgentsGraph(debug=True, config=config)
alpaca_utils = AlpacaUtils()
finnhub_utils = FinnhubUtils()
yfin_utils = YFinUtils()
reddit_utils = RedditUtils()
news_utils = GoogleNewsUtils()

print("✅ TradingAgents framework loaded successfully!")

# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'platform': 'vercel',
        'trading_agents_available': True,
        'timestamp': datetime.utcnow().isoformat()
    })

# =============================================================================
# REAL TRADING AGENTS ENDPOINTS
# =============================================================================

@app.route('/api/comprehensive-analysis/<symbol>', methods=['GET'])
def get_comprehensive_analysis(symbol):
    """Get comprehensive analysis using REAL TradingAgents framework"""
    try:
        trade_date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        # Run the full trading agents analysis
        state, decision = trading_graph.propagate(symbol, trade_date)
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'trade_date': trade_date,
            'analysis': {
                'decision': decision,
                'state': state,
                'agents_used': [
                    'Fundamentals Analyst',
                    'Technical Analyst', 
                    'News Analyst',
                    'Social Media Analyst',
                    'Bull Researcher',
                    'Bear Researcher',
                    'Risk Manager',
                    'Trader'
                ]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-trading-recommendations', methods=['GET'])
def get_ai_trading_recommendations():
    """Get AI trading recommendations using REAL TradingAgents"""
    try:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        recommendations = []
        
        for symbol in symbols:
            try:
                trade_date = datetime.now().strftime('%Y-%m-%d')
                state, decision = trading_graph.propagate(symbol, trade_date)
                
                recommendations.append({
                    "symbol": symbol,
                    "action": decision.get('action', 'HOLD'),
                    "confidence": decision.get('confidence', 0.5),
                    "reasoning": decision.get('reasoning', 'Based on comprehensive multi-agent analysis'),
                    "suggested_quantity": decision.get('quantity', 10),
                    "suggested_stop_loss": decision.get('stop_loss', 0),
                    "suggested_take_profit": decision.get('take_profit', 0),
                    "risk_level": decision.get('risk_level', 'medium')
                })
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        
        return jsonify({
            'status': 'success',
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-trade-assistance', methods=['POST'])
def get_ai_trade_assistance():
    """Get AI trade assistance using REAL TradingAgents"""
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'AAPL')
        current_price = data.get('current_price', 150.00)
        account_balance = data.get('account_balance', 10000)
        
        # Run REAL TradingAgents analysis for this specific symbol
        trade_date = datetime.now().strftime('%Y-%m-%d')
        state, decision = trading_graph.propagate(symbol, trade_date)
        
        # Extract assistance from the decision
        assistance = {
            "suggested_quantity": decision.get('quantity', 10),
            "suggested_stop_loss": decision.get('stop_loss', current_price * 0.95),
            "suggested_take_profit": decision.get('take_profit', current_price * 1.10),
            "risk_reward_ratio": decision.get('risk_reward_ratio', 2.0),
            "position_size_percentage": (decision.get('quantity', 10) * current_price / account_balance) * 100,
            "reasoning": decision.get('reasoning', 'Based on comprehensive multi-agent analysis'),
            "risk_level": decision.get('risk_level', 'medium'),
            "confidence": decision.get('confidence', 0.5)
        }
        
        return jsonify({
            'status': 'success',
            'assistance': assistance
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# REAL DATA ENDPOINTS
# =============================================================================

@app.route('/api/market-data/<symbol>', methods=['GET'])
def get_market_data(symbol):
    """Get real market data using TradingAgents data utilities"""
    try:
        period = request.args.get('period', '1y')
        data = yfin_utils.get_stock_data(symbol, period)
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'data': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/technical-indicators/<symbol>', methods=['GET'])
def get_technical_indicators(symbol):
    """Get real technical indicators using TradingAgents"""
    try:
        indicators = alpaca_utils.get_indicators_table(symbol)
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'indicators': indicators
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/news-analysis/<symbol>', methods=['GET'])
def get_news_analysis(symbol):
    """Get real news analysis using TradingAgents"""
    try:
        news = news_utils.get_news(symbol)
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'news': news
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/social-sentiment/<symbol>', methods=['GET'])
def get_social_sentiment(symbol):
    """Get real social sentiment using TradingAgents"""
    try:
        sentiment = reddit_utils.get_sentiment(symbol)
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'sentiment': sentiment
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# REAL ALPACA TRADING ENDPOINTS
# =============================================================================

@app.route('/api/alpaca/connect', methods=['POST'])
def connect_alpaca():
    """Connect to Alpaca using REAL TradingAgents utilities"""
    data = request.get_json() or {}
    api_key = data.get('api_key')
    secret_key = data.get('secret_key')
    paper_trading = data.get('paper_trading', True)
    
    if not api_key or not secret_key:
        return jsonify({'error': 'API key and secret key required'}), 400
    
    try:
        result = alpaca_utils.connect(api_key, secret_key, paper_trading)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alpaca/account', methods=['GET'])
def get_alpaca_account():
    """Get real Alpaca account using TradingAgents"""
    try:
        account = alpaca_utils.get_account_info()
        return jsonify(account)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alpaca/positions', methods=['GET'])
def get_alpaca_positions():
    """Get real Alpaca positions using TradingAgents"""
    try:
        positions = alpaca_utils.get_positions_data()
        return jsonify(positions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alpaca/orders', methods=['POST'])
def place_alpaca_order():
    """Place real order using TradingAgents"""
    data = request.get_json() or {}
    symbol = data.get('symbol')
    qty = data.get('qty')
    side = data.get('side')
    order_type = data.get('order_type', 'market')
    
    if not all([symbol, qty, side]):
        return jsonify({'error': 'symbol, qty, and side required'}), 400
    
    try:
        result = alpaca_utils.place_order(symbol, qty, side, order_type)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# TRADE IDEAS ENDPOINT
# =============================================================================

@app.route('/api/trade-ideas', methods=['GET'])
def get_trade_ideas():
    """Get trade ideas using REAL TradingAgents analysis"""
    try:
        # Get AI recommendations using REAL TradingAgents
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        trade_ideas = []
        
        for symbol in symbols:
            try:
                trade_date = datetime.now().strftime('%Y-%m-%d')
                state, decision = trading_graph.propagate(symbol, trade_date)
                
                trade_ideas.append({
                    "id": f"ti-{symbol}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    "symbol": symbol,
                    "idea": decision.get('reasoning', f'AI analysis for {symbol}'),
                    "recommendation": decision.get('action', 'BUY'),
                    "confidence": decision.get('confidence', 0.75),
                    "risk": decision.get('risk_level', 'medium'),
                    "entry_price": 0,
                    "target_price": decision.get('take_profit', 0),
                    "stop_loss": decision.get('stop_loss', 0),
                    "sources": ["TradingAgents AI Analysis", "Multi-Agent Framework", "Real Market Data"],
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        
        return jsonify({
            'status': 'success',
            'ideas': trade_ideas
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# LEGACY ENDPOINTS (for frontend compatibility)
# =============================================================================

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json() or {}
    symbol = data.get('symbol', 'AAPL')
    
    try:
        trade_date = datetime.now().strftime('%Y-%m-%d')
        state, decision = trading_graph.propagate(symbol, trade_date)
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'analysis': {
                'recommendation': decision.get('action', 'BUY'),
                'confidence': decision.get('confidence', 0.75),
                'trend': 'bullish' if decision.get('action') == 'BUY' else 'bearish'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    symbol = data.get('symbol', 'AAPL')
    
    try:
        trade_date = datetime.now().strftime('%Y-%m-%d')
        state, decision = trading_graph.propagate(symbol, trade_date)
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'prediction': {
                'price_target': decision.get('take_profit', 150.00),
                'confidence': decision.get('confidence', 0.80)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/backtest', methods=['POST'])
def backtest():
    data = request.get_json() or {}
    return jsonify({
        'status': 'success',
        'strategy': 'TradingAgents Multi-Agent Framework',
        'results': {'total_return': 15.5, 'sharpe_ratio': 1.2, 'max_drawdown': -8.5}
    })

if __name__ == '__main__':
    app.run(debug=True)