from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import requests
import json
import os
import random

app = Flask(__name__)
CORS(app)

# =============================================================================
# SUPABASE INTEGRATION
# =============================================================================

try:
    from supabase import create_client, Client
    import jwt
    
    # Initialize Supabase client
    SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://mtozccjlxpiohllyteta.supabase.co')
    SUPABASE_KEY = os.getenv('SUPABASE_ANON_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im10b3pjY2pseHBpb2hsbHl0ZXRhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjAzMjQ0ODcsImV4cCI6MjA3NTkwMDQ4N30.RsbZd2iUlmOP56iYg0F7mgm0oYqUEcQtO8r2XAOAeTQ')
    
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    SUPABASE_ENABLED = True
except Exception as e:
    print(f"Supabase initialization failed: {e}")
    SUPABASE_ENABLED = False

# =============================================================================
# OPENBB INTEGRATION
# =============================================================================

class OpenBBIntegration:
    def __init__(self):
        self.base_url = "https://api.openbb.co"
        self.pat_token = None
    
    def set_pat_token(self, token):
        """Set the OpenBB PAT token for API access"""
        self.pat_token = token
    
    def get_equity_data(self, symbol, timeframe="1d", limit=100):
        """Get equity data from OpenBB"""
        try:
            if not self.pat_token:
                return {"error": "OpenBB PAT token not set"}
            
            headers = {
                "Authorization": f"Bearer {self.pat_token}",
                "Content-Type": "application/json"
            }
            
            # Mock data for now - replace with actual OpenBB API calls
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "data": {
                    "prices": [random.uniform(100, 200) for _ in range(limit)],
                    "volumes": [random.randint(1000, 10000) for _ in range(limit)],
                    "timestamps": [(datetime.now() - timedelta(days=i)).isoformat() for i in range(limit, 0, -1)]
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_crypto_data(self, symbol, timeframe="1d"):
        """Get cryptocurrency data"""
        return self.get_equity_data(symbol, timeframe)
    
    def get_forex_data(self, pair, timeframe="1d"):
        """Get forex data"""
        return self.get_equity_data(pair, timeframe)

openbb = OpenBBIntegration()

# =============================================================================
# MCP (MODEL CONTEXT PROTOCOL) INTEGRATION
# =============================================================================

class MCPIntegration:
    def __init__(self):
        self.tools = {
            "alpaca_trading": {
                "description": "Execute trades on Alpaca",
                "parameters": ["symbol", "qty", "side", "type"]
            },
            "market_data": {
                "description": "Get real-time market data",
                "parameters": ["symbol", "timeframe"]
            },
            "portfolio_analysis": {
                "description": "Analyze portfolio performance",
                "parameters": ["portfolio_id"]
            },
            "risk_management": {
                "description": "Calculate risk metrics",
                "parameters": ["positions", "risk_tolerance"]
            }
        }
    
    def execute_tool(self, tool_name, parameters, user_id=None):
        """Execute an MCP tool"""
        try:
            if tool_name not in self.tools:
                return {"error": f"Tool {tool_name} not found"}
            
            # Mock execution - replace with actual MCP tool execution
            result = {
                "tool": tool_name,
                "parameters": parameters,
                "result": f"Mock execution of {tool_name} with {parameters}",
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id
            }
            
            # Store in database if Supabase is available
            if SUPABASE_ENABLED and user_id:
                try:
                    supabase.table("mcp_tool_usage").insert({
                        "session_id": f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        "tool_name": tool_name,
                        "input_data": parameters,
                        "output_data": result,
                        "success": True
                    }).execute()
                except Exception as e:
                    print(f"Failed to store MCP usage: {e}")
            
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def get_available_tools(self):
        """Get list of available MCP tools"""
        return self.tools

mcp = MCPIntegration()

# =============================================================================
# AI CHAT INTEGRATION
# =============================================================================

class AIChatIntegration:
    def __init__(self):
        self.conversations = {}
    
    def process_message(self, message, user_id, context="strategy_building"):
        """Process AI chat message for strategy building"""
        try:
            # Mock AI response - replace with actual AI integration
            responses = {
                "strategy_building": [
                    "I can help you build algorithmic trading strategies. What type of strategy are you interested in?",
                    "Let's analyze your portfolio and suggest some strategies based on your risk tolerance.",
                    "I can create custom indicators and backtest them for you.",
                    "Would you like me to help you implement a momentum strategy or a mean reversion strategy?"
                ],
                "market_analysis": [
                    "I can analyze market trends and provide insights on potential opportunities.",
                    "Let me check the latest market data and identify patterns.",
                    "Based on current market conditions, I recommend focusing on defensive stocks.",
                    "I can help you identify oversold or overbought conditions in the market."
                ],
                "alpaca_integration": [
                    "I can help you set up automated trading with Alpaca API.",
                    "Let's configure your Alpaca credentials and create trading strategies.",
                    "I can help you implement paper trading first, then move to live trading.",
                    "Would you like me to create a portfolio rebalancing strategy?"
                ]
            }
            
            # Get appropriate response based on context
            context_responses = responses.get(context, responses["strategy_building"])
            response = random.choice(context_responses)
            
            # Store conversation in database if Supabase is available
            if SUPABASE_ENABLED:
                try:
                    # Create or get session
                    session_result = supabase.table("ai_chat_sessions").select("id").eq("user_id", user_id).eq("context", context).order("created_at", desc=True).limit(1).execute()
                    
                    if session_result.data:
                        session_id = session_result.data[0]["id"]
                    else:
                        session_result = supabase.table("ai_chat_sessions").insert({
                            "user_id": user_id,
                            "session_name": f"AI Chat - {context}",
                            "context": context
                        }).execute()
                        session_id = session_result.data[0]["id"]
                    
                    # Store messages
                    supabase.table("ai_chat_messages").insert([
                        {
                            "session_id": session_id,
                            "role": "user",
                            "content": message,
                            "metadata": {"context": context}
                        },
                        {
                            "session_id": session_id,
                            "role": "assistant",
                            "content": response,
                            "metadata": {"context": context}
                        }
                    ]).execute()
                except Exception as e:
                    print(f"Failed to store chat messages: {e}")
            
            return {
                "response": response,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}

ai_chat = AIChatIntegration()

# =============================================================================
# CORE ENDPOINTS
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'platform': 'vercel',
        'supabase_enabled': SUPABASE_ENABLED,
        'timestamp': datetime.utcnow().isoformat()
    })

# =============================================================================
# TRADE IDEA GENERATION ENDPOINTS
# =============================================================================

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
            "timestamp": (datetime.utcnow() - timedelta(hours=1)).isoformat()
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
            "timestamp": (datetime.utcnow() - timedelta(hours=3)).isoformat()
        }
    ]
    
    return jsonify({
        'status': 'success',
        'ideas': mock_ideas
    })

@app.route('/api/quant-indicators', methods=['POST'])
def get_quant_indicators():
    """Returns quantitative indicators for a given symbol."""
    data = request.get_json() or {}
    symbol = data.get('symbol', 'AAPL')
    
    return jsonify({
        'status': 'success',
        'symbol': symbol,
        'indicators': {
            'RSI': random.uniform(30, 70),
            'MACD': random.uniform(-5, 5),
            'BollingerBands': {'upper': 160, 'middle': 150, 'lower': 140},
            'Volume': random.randint(1000000, 50000000)
        }
    })

# =============================================================================
# OPENBB INTEGRATION ENDPOINTS
# =============================================================================

@app.route('/api/openbb/set-token', methods=['POST'])
def set_openbb_token():
    """Set OpenBB PAT token"""
    data = request.get_json() or {}
    token = data.get('token')
    
    if not token:
        return jsonify({'error': 'Token required'}), 400
    
    openbb.set_pat_token(token)
    return jsonify({'status': 'success', 'message': 'OpenBB token set'})

@app.route('/api/openbb/equity/<symbol>', methods=['GET'])
def get_openbb_equity(symbol):
    """Get equity data from OpenBB"""
    timeframe = request.args.get('timeframe', '1d')
    limit = int(request.args.get('limit', 100))
    
    result = openbb.get_equity_data(symbol, timeframe, limit)
    return jsonify(result)

@app.route('/api/openbb/crypto/<symbol>', methods=['GET'])
def get_openbb_crypto(symbol):
    """Get crypto data from OpenBB"""
    timeframe = request.args.get('timeframe', '1d')
    
    result = openbb.get_crypto_data(symbol, timeframe)
    return jsonify(result)

@app.route('/api/openbb/forex/<pair>', methods=['GET'])
def get_openbb_forex(pair):
    """Get forex data from OpenBB"""
    timeframe = request.args.get('timeframe', '1d')
    
    result = openbb.get_forex_data(pair, timeframe)
    return jsonify(result)

# =============================================================================
# MCP INTEGRATION ENDPOINTS
# =============================================================================

@app.route('/api/mcp/tools', methods=['GET'])
def get_mcp_tools():
    """Get available MCP tools"""
    return jsonify({
        'status': 'success',
        'tools': mcp.get_available_tools()
    })

@app.route('/api/mcp/execute', methods=['POST'])
def execute_mcp_tool():
    """Execute an MCP tool"""
    data = request.get_json() or {}
    tool_name = data.get('tool_name')
    parameters = data.get('parameters', {})
    user_id = data.get('user_id')
    
    if not tool_name:
        return jsonify({'error': 'tool_name required'}), 400
    
    result = mcp.execute_tool(tool_name, parameters, user_id)
    return jsonify(result)

# =============================================================================
# AI CHAT ENDPOINTS
# =============================================================================

@app.route('/api/ai-chat', methods=['POST'])
def ai_chat_endpoint():
    """AI chat for strategy building"""
    data = request.get_json() or {}
    message = data.get('message')
    user_id = data.get('user_id')
    context = data.get('context', 'strategy_building')
    
    if not message:
        return jsonify({'error': 'message required'}), 400
    
    result = ai_chat.process_message(message, user_id, context)
    return jsonify(result)

@app.route('/api/ai-chat/sessions', methods=['GET'])
def get_chat_sessions():
    """Get AI chat sessions for a user"""
    user_id = request.args.get('user_id')
    context = request.args.get('context', 'strategy_building')
    
    if not user_id:
        return jsonify({'error': 'user_id required'}), 400
    
    if not SUPABASE_ENABLED:
        return jsonify({'error': 'Supabase not available'}), 500
    
    try:
        result = supabase.table("ai_chat_sessions").select("*").eq("user_id", user_id).eq("context", context).order("created_at", desc=True).execute()
        return jsonify({'status': 'success', 'sessions': result.data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-chat/messages/<session_id>', methods=['GET'])
def get_chat_messages(session_id):
    """Get messages for a chat session"""
    if not SUPABASE_ENABLED:
        return jsonify({'error': 'Supabase not available'}), 500
    
    try:
        result = supabase.table("ai_chat_messages").select("*").eq("session_id", session_id).order("created_at", asc=True).execute()
        return jsonify({'status': 'success', 'messages': result.data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ALPACA INTEGRATION ENDPOINTS
# =============================================================================

@app.route('/api/alpaca/account', methods=['GET'])
def get_alpaca_account():
    """Get Alpaca account information"""
    # This would integrate with actual Alpaca API
    return jsonify({
        'status': 'success',
        'account': {
            'buying_power': 10000.00,
            'cash': 5000.00,
            'portfolio_value': 15000.00,
            'equity': 15000.00
        }
    })

@app.route('/api/alpaca/positions', methods=['GET'])
def get_alpaca_positions():
    """Get Alpaca positions"""
    return jsonify({
        'status': 'success',
        'positions': [
            {
                'symbol': 'AAPL',
                'qty': 10,
                'market_value': 1500.00,
                'unrealized_pl': 50.00
            }
        ]
    })

# =============================================================================
# STRATEGY BUILDING ENDPOINTS
# =============================================================================

@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """Get trading strategies for a user"""
    user_id = request.args.get('user_id')
    
    if not SUPABASE_ENABLED:
        return jsonify({'error': 'Supabase not available'}), 500
    
    try:
        result = supabase.table("trading_strategies").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return jsonify({'status': 'success', 'strategies': result.data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies', methods=['POST'])
def create_strategy():
    """Create a new trading strategy"""
    data = request.get_json() or {}
    
    if not SUPABASE_ENABLED:
        return jsonify({'error': 'Supabase not available'}), 500
    
    try:
        result = supabase.table("trading_strategies").insert(data).execute()
        return jsonify({'status': 'success', 'strategy': result.data[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtests', methods=['POST'])
def run_backtest():
    """Run a backtest for a strategy"""
    data = request.get_json() or {}
    
    # Mock backtest results
    backtest_result = {
        'strategy_id': data.get('strategy_id'),
        'start_date': data.get('start_date'),
        'end_date': data.get('end_date'),
        'initial_capital': data.get('initial_capital', 10000),
        'final_capital': random.uniform(8000, 15000),
        'total_return': random.uniform(-20, 50),
        'sharpe_ratio': random.uniform(0.5, 2.5),
        'max_drawdown': random.uniform(-30, -5),
        'win_rate': random.uniform(0.4, 0.8)
    }
    
    return jsonify({'status': 'success', 'backtest': backtest_result})

if __name__ == '__main__':
    app.run(debug=True)
