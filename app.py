from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import requests
import json

app = Flask(__name__)
CORS(app)

# =============================================================================
# TRADE IDEA GENERATION ENDPOINTS
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'platform': 'vercel'})

# =============================================================================
# QUANTITATIVE INDICATORS
# =============================================================================

@app.route('/api/quant-indicators', methods=['POST'])
def quant_indicators():
    """Generate trade ideas from quantitative indicators"""
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'AAPL')
        
        # Mock quantitative analysis
        indicators = {
            'rsi': 65,
            'macd': 'bullish',
            'bollinger_position': 'upper_band',
            'volume_spike': True,
            'moving_average_cross': 'golden_cross'
        }
        
        trade_ideas = []
        
        if indicators['rsi'] > 70:
            trade_ideas.append({
                'type': 'overbought',
                'signal': 'SELL',
                'confidence': 0.8,
                'reason': f'{symbol} is overbought (RSI: {indicators["rsi"]})',
                'target_price': 140.0,
                'stop_loss': 150.0
            })
        elif indicators['rsi'] < 30:
            trade_ideas.append({
                'type': 'oversold',
                'signal': 'BUY',
                'confidence': 0.75,
                'reason': f'{symbol} is oversold (RSI: {indicators["rsi"]})',
                'target_price': 160.0,
                'stop_loss': 130.0
            })
        
        if indicators['macd'] == 'bullish':
            trade_ideas.append({
                'type': 'momentum',
                'signal': 'BUY',
                'confidence': 0.7,
                'reason': 'MACD showing bullish momentum',
                'target_price': 165.0,
                'stop_loss': 145.0
            })
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'indicators': indicators,
            'trade_ideas': trade_ideas,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# UNUSUAL OPTIONS ACTIVITY
# =============================================================================

@app.route('/api/unusual-options', methods=['POST'])
def unusual_options():
    """Detect unusual options activity for trade ideas"""
    try:
        data = request.get_json() or {}
        limit = data.get('limit', 10)
        
        # Mock unusual options data
        unusual_activity = [
            {
                'symbol': 'NVDA',
                'strike': 450,
                'expiry': '2024-01-19',
                'option_type': 'call',
                'volume': 15000,
                'open_interest': 25000,
                'premium': 12.50,
                'volume_ratio': 3.2,
                'sentiment': 'bullish',
                'reason': 'Large call volume suggests bullish sentiment'
            },
            {
                'symbol': 'TSLA',
                'strike': 200,
                'expiry': '2024-01-19',
                'option_type': 'put',
                'volume': 8500,
                'open_interest': 12000,
                'premium': 8.75,
                'volume_ratio': 2.8,
                'sentiment': 'bearish',
                'reason': 'Unusual put activity indicates downside protection'
            }
        ]
        
        return jsonify({
            'status': 'success',
            'unusual_activity': unusual_activity[:limit],
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# INSIDER TRADING ALERTS
# =============================================================================

@app.route('/api/insider-trading', methods=['POST'])
def insider_trading():
    """Get insider buying/selling alerts"""
    try:
        data = request.get_json() or {}
        limit = data.get('limit', 10)
        
        # Mock insider trading data
        insider_activity = [
            {
                'symbol': 'AAPL',
                'insider_name': 'Tim Cook',
                'position': 'CEO',
                'transaction_type': 'BUY',
                'shares': 100000,
                'price': 175.50,
                'date': '2024-01-15',
                'total_value': 17550000,
                'significance': 'high',
                'reason': 'CEO buying significant shares - bullish signal'
            },
            {
                'symbol': 'MSFT',
                'insider_name': 'Satya Nadella',
                'position': 'CEO',
                'transaction_type': 'SELL',
                'shares': 50000,
                'price': 380.25,
                'date': '2024-01-14',
                'total_value': 19012500,
                'significance': 'medium',
                'reason': 'CEO selling for tax purposes - not necessarily bearish'
            }
        ]
        
        return jsonify({
            'status': 'success',
            'insider_activity': insider_activity[:limit],
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# EVENTS CALENDAR
# =============================================================================

@app.route('/api/events-calendar', methods=['POST'])
def events_calendar():
    """Get upcoming events that could impact stocks"""
    try:
        data = request.get_json() or {}
        days_ahead = data.get('days_ahead', 7)
        
        # Mock events data
        events = [
            {
                'date': '2024-01-22',
                'time': '09:00',
                'event_type': 'earnings',
                'symbol': 'TSLA',
                'event': 'Tesla Q4 Earnings',
                'impact': 'high',
                'expected_eps': 0.85,
                'expected_revenue': 25.2,
                'trade_idea': 'Watch for guidance on Cybertruck production and FSD progress'
            },
            {
                'date': '2024-01-23',
                'time': '14:00',
                'event_type': 'fda_approval',
                'symbol': 'MRNA',
                'event': 'FDA Advisory Committee Meeting',
                'impact': 'high',
                'expected_eps': None,
                'expected_revenue': None,
                'trade_idea': 'Potential approval could drive significant upside'
            },
            {
                'date': '2024-01-24',
                'time': '10:30',
                'event_type': 'economic',
                'symbol': 'SPY',
                'event': 'GDP Growth Rate',
                'impact': 'medium',
                'expected_eps': None,
                'expected_revenue': None,
                'trade_idea': 'Strong GDP could boost market sentiment'
            }
        ]
        
        return jsonify({
            'status': 'success',
            'events': events,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# PREMARKET MOVERS
# =============================================================================

@app.route('/api/premarket-movers', methods=['POST'])
def premarket_movers():
    """Get premarket movers for early trade ideas"""
    try:
        data = request.get_json() or {}
        limit = data.get('limit', 20)
        
        # Mock premarket data
        movers = [
            {
                'symbol': 'NVDA',
                'name': 'NVIDIA Corporation',
                'price': 485.50,
                'change': 15.25,
                'change_percent': 3.24,
                'volume': 2500000,
                'reason': 'AI chip demand surge',
                'sentiment': 'bullish',
                'trade_idea': 'Strong premarket momentum suggests continued AI rally'
            },
            {
                'symbol': 'TSLA',
                'name': 'Tesla Inc',
                'price': 195.75,
                'change': -8.25,
                'change_percent': -4.04,
                'volume': 1800000,
                'reason': 'China sales concerns',
                'sentiment': 'bearish',
                'trade_idea': 'Weak premarket suggests continued pressure'
            }
        ]
            
            return jsonify({
                'status': 'success',
            'movers': movers[:limit],
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# NEWS ANALYSIS WITH TRADE IMPLICATIONS
# =============================================================================

@app.route('/api/news-analysis', methods=['POST'])
def news_analysis():
    """Analyze news for trade implications"""
    try:
        data = request.get_json() or {}
        news_text = data.get('news_text', '')
        
        # Mock news analysis
        analysis = {
            'sentiment': 'bullish',
            'confidence': 0.85,
            'key_entities': ['Tesla', 'EV', 'China', 'Battery'],
            'trade_implications': [
                {
                    'symbol': 'TSLA',
                    'impact': 'positive',
                    'reason': 'EV market expansion in China',
                    'confidence': 0.8
                },
                {
                    'symbol': 'CATL',
                    'impact': 'positive',
                    'reason': 'Battery supply chain benefits',
                    'confidence': 0.75
                }
            ],
            'sector_impact': {
                'EV': 'bullish',
                'Battery': 'bullish',
                'Traditional_Auto': 'bearish'
            }
        }
        
        return jsonify({
            'status': 'success',
            'analysis': analysis,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# COMPREHENSIVE TRADE IDEAS
# =============================================================================

@app.route('/api/trade-ideas', methods=['POST'])
def trade_ideas():
    """Generate comprehensive trade ideas from multiple sources"""
    try:
        data = request.get_json() or {}
        user_preferences = data.get('preferences', {})
        
        # Mock comprehensive trade ideas
        ideas = [
            {
                'id': 'idea_1',
                'symbol': 'NVDA',
                'type': 'momentum',
                'signal': 'BUY',
                'confidence': 0.85,
                'timeframe': '1-2 weeks',
                'entry_price': 485.50,
                'target_price': 520.00,
                'stop_loss': 460.00,
                'reason': 'AI chip demand surge + unusual options activity',
                'sources': ['quant_indicators', 'unusual_options', 'news_sentiment'],
                'risk_level': 'medium',
                'expected_return': 7.1
            },
            {
                'id': 'idea_2',
                'symbol': 'TSLA',
                'type': 'contrarian',
                'signal': 'BUY',
                'confidence': 0.70,
                'timeframe': '2-4 weeks',
                'entry_price': 195.75,
                'target_price': 220.00,
                'stop_loss': 180.00,
                'reason': 'Oversold conditions + CEO insider buying',
                'sources': ['technical_analysis', 'insider_trading'],
                'risk_level': 'high',
                'expected_return': 12.4
            }
        ]
        
        return jsonify({
            'status': 'success',
            'trade_ideas': ideas,
            'generated_at': datetime.utcnow().isoformat(),
            'sources_analyzed': [
                'quantitative_indicators',
                'unusual_options',
                'insider_trading',
                'news_sentiment',
                'premarket_movers',
                'events_calendar'
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ENHANCED RSS NEWS ANALYSIS
# =============================================================================

@app.route('/api/enhanced-news', methods=['POST'])
def enhanced_news():
    """Enhanced news analysis with trade implications"""
    try:
        data = request.get_json() or {}
        news_item = data.get('news_item', {})
        
        # Mock enhanced news analysis
        enhanced_analysis = {
            'original_news': news_item,
            'trade_implications': [
                {
                    'symbol': 'BHP',
                    'impact': 'positive',
                    'reason': 'Critical minerals deal benefits mining companies',
                    'confidence': 0.8,
                    'trade_idea': 'Consider BHP calls or direct stock purchase'
                },
                {
                    'symbol': 'LAC',
                    'impact': 'positive',
                    'reason': 'Lithium demand from EV sector',
                    'confidence': 0.75,
                    'trade_idea': 'Lithium stocks could see continued strength'
                }
            ],
            'sector_analysis': {
                'mining': 'bullish',
                'lithium': 'bullish',
                'renewable_energy': 'bullish'
            },
            'key_phrases': ['critical minerals', '$1B deal', 'Australia', 'Trump'],
            'sentiment_score': 0.7,
            'urgency': 'medium'
        }
        
        return jsonify({
            'status': 'success',
            'enhanced_analysis': enhanced_analysis,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ROOT ENDPOINTS FOR FRONTEND COMPATIBILITY
# =============================================================================

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze market data - enhanced version"""
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'AAPL')
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'analysis': {
                'recommendation': 'BUY',
                'confidence': 0.75,
                'trend': 'bullish',
                'trade_ideas': [
                    {
                        'type': 'momentum',
                        'signal': 'BUY',
                        'target': 160.0,
                        'stop_loss': 140.0
                    }
                ]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict price movement - enhanced version"""
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'AAPL')
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'prediction': {
                'price_target': 150.0,
                'confidence': 0.8,
                'timeframe': '1 week',
                'factors': ['technical_indicators', 'news_sentiment', 'options_flow']
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/backtest', methods=['POST'])
def backtest():
    """Run backtest - enhanced version"""
    try:
        data = request.get_json() or {}
        
        return jsonify({
            'status': 'success',
            'results': {
                'total_return': 15.5,
                'sharpe_ratio': 1.2,
                'max_drawdown': -8.5,
                'win_rate': 65.0,
                'trade_ideas_tested': 25,
                'successful_ideas': 16
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)