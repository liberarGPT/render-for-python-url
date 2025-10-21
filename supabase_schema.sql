-- Comprehensive Trading Platform Database Schema
-- Supabase Integration with OpenBB + MCP

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  first_name TEXT,
  last_name TEXT,
  subscription_status TEXT DEFAULT 'free',
  subscription_tier TEXT DEFAULT 'basic',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  last_login TIMESTAMP,
  is_active BOOLEAN DEFAULT true
);

-- User API Keys (encrypted)
CREATE TABLE user_api_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  broker TEXT NOT NULL, -- 'alpaca', 'td_ameritrade', 'ibkr', 'openbb_fmp', 'openbb_polygon', etc.
  api_key TEXT NOT NULL,
  secret_key TEXT,
  pat_token TEXT, -- For OpenBB PAT tokens
  paper_trading BOOLEAN DEFAULT true,
  is_active BOOLEAN DEFAULT true,
  rate_limit_remaining INTEGER,
  rate_limit_reset TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- OpenBB API Keys
CREATE TABLE openbb_api_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  provider TEXT NOT NULL, -- 'fmp', 'polygon', 'quandl', 'fred', 'econdb', 'openbb_pat'
  api_key TEXT NOT NULL,
  is_active BOOLEAN DEFAULT true,
  rate_limit_remaining INTEGER,
  rate_limit_reset TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW()
);

-- OpenBB Data Cache
CREATE TABLE openbb_data_cache (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  data_type TEXT NOT NULL, -- 'stock_data', 'economic_data', 'news', 'crypto_data'
  symbol TEXT,
  data JSONB NOT NULL,
  expires_at TIMESTAMP NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

-- MCP Sessions
CREATE TABLE mcp_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  session_name TEXT NOT NULL,
  context_data JSONB,
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- MCP Tools Usage
CREATE TABLE mcp_tool_usage (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES mcp_sessions(id) ON DELETE CASCADE,
  tool_name TEXT NOT NULL,
  parameters JSONB,
  result JSONB,
  execution_time_ms INTEGER,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Chat Sessions
CREATE TABLE chat_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  session_name TEXT,
  mcp_session_id UUID REFERENCES mcp_sessions(id),
  created_at TIMESTAMP DEFAULT NOW()
);

-- Chat Messages
CREATE TABLE chat_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL, -- 'user', 'assistant'
  content TEXT NOT NULL,
  metadata JSONB, -- Additional context, tool usage, etc.
  timestamp TIMESTAMP DEFAULT NOW()
);

-- User Portfolios
CREATE TABLE portfolios (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  description TEXT,
  total_value DECIMAL DEFAULT 0,
  cash_balance DECIMAL DEFAULT 0,
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Portfolio Holdings
CREATE TABLE portfolio_holdings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,
  symbol TEXT NOT NULL,
  quantity DECIMAL NOT NULL,
  average_cost DECIMAL NOT NULL,
  current_price DECIMAL,
  last_updated TIMESTAMP DEFAULT NOW()
);

-- Trading History
CREATE TABLE trades (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  portfolio_id UUID REFERENCES portfolios(id),
  symbol TEXT NOT NULL,
  action TEXT NOT NULL, -- 'buy', 'sell'
  quantity DECIMAL NOT NULL,
  price DECIMAL NOT NULL,
  commission DECIMAL DEFAULT 0,
  timestamp TIMESTAMP DEFAULT NOW(),
  strategy_used TEXT,
  profit_loss DECIMAL,
  status TEXT DEFAULT 'completed' -- 'pending', 'completed', 'cancelled'
);

-- Saved Strategies
CREATE TABLE saved_strategies (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  description TEXT,
  strategy_config JSONB NOT NULL, -- strategy parameters
  is_public BOOLEAN DEFAULT false,
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Backtest Results
CREATE TABLE backtest_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  strategy_id UUID REFERENCES saved_strategies(id),
  name TEXT NOT NULL,
  symbols TEXT[] NOT NULL,
  start_date DATE NOT NULL,
  end_date DATE NOT NULL,
  initial_capital DECIMAL NOT NULL,
  final_value DECIMAL NOT NULL,
  total_return DECIMAL NOT NULL,
  annual_return DECIMAL,
  volatility DECIMAL,
  sharpe_ratio DECIMAL,
  max_drawdown DECIMAL,
  win_rate DECIMAL,
  profit_factor DECIMAL,
  strategy_parameters JSONB NOT NULL,
  backtest_engine TEXT NOT NULL, -- 'backtrader', 'vectorbt'
  created_at TIMESTAMP DEFAULT NOW()
);

-- Backtest Trades
CREATE TABLE backtest_trades (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  backtest_id UUID REFERENCES backtest_results(id) ON DELETE CASCADE,
  symbol TEXT NOT NULL,
  action TEXT NOT NULL, -- 'buy', 'sell'
  quantity DECIMAL NOT NULL,
  price DECIMAL NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  profit_loss DECIMAL,
  commission DECIMAL DEFAULT 0
);

-- Market Data Storage
CREATE TABLE market_data (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  symbol TEXT NOT NULL,
  date TIMESTAMP NOT NULL,
  open DECIMAL NOT NULL,
  high DECIMAL NOT NULL,
  low DECIMAL NOT NULL,
  close DECIMAL NOT NULL,
  volume BIGINT NOT NULL,
  source TEXT NOT NULL, -- 'yfinance', 'alpha_vantage', 'alpaca', 'openbb'
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(symbol, date, source)
);

-- Technical Indicators
CREATE TABLE technical_indicators (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  symbol TEXT NOT NULL,
  date TIMESTAMP NOT NULL,
  indicator_name TEXT NOT NULL,
  value DECIMAL NOT NULL,
  parameters JSONB, -- indicator parameters
  created_at TIMESTAMP DEFAULT NOW()
);

-- ML Models
CREATE TABLE ml_models (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  model_name TEXT NOT NULL,
  model_type TEXT NOT NULL, -- 'classification', 'regression', 'lstm', 'transformer'
  symbol TEXT NOT NULL,
  training_data_start DATE NOT NULL,
  training_data_end DATE NOT NULL,
  model_parameters JSONB NOT NULL,
  model_weights BYTEA, -- serialized model
  accuracy_score DECIMAL,
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Model Predictions
CREATE TABLE model_predictions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id UUID REFERENCES ml_models(id) ON DELETE CASCADE,
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  symbol TEXT NOT NULL,
  prediction_date TIMESTAMP NOT NULL,
  predicted_price DECIMAL,
  confidence DECIMAL,
  actual_price DECIMAL,
  accuracy DECIMAL,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Portfolio Optimizations
CREATE TABLE portfolio_optimizations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  optimization_type TEXT NOT NULL, -- 'max_sharpe', 'min_volatility', 'efficient_return'
  target_return DECIMAL,
  risk_tolerance DECIMAL,
  symbols TEXT[] NOT NULL,
  weights JSONB NOT NULL,
  expected_return DECIMAL,
  volatility DECIMAL,
  sharpe_ratio DECIMAL,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Risk Metrics
CREATE TABLE risk_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,
  date DATE NOT NULL,
  var_95 DECIMAL, -- Value at Risk 95%
  var_99 DECIMAL, -- Value at Risk 99%
  expected_shortfall DECIMAL,
  beta DECIMAL,
  alpha DECIMAL,
  information_ratio DECIMAL,
  calmar_ratio DECIMAL,
  sortino_ratio DECIMAL,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Crypto Exchanges
CREATE TABLE crypto_exchanges (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  exchange_name TEXT NOT NULL, -- 'binance', 'coinbase', 'kraken'
  api_key TEXT NOT NULL,
  secret_key TEXT,
  sandbox BOOLEAN DEFAULT true,
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Crypto Trades
CREATE TABLE crypto_trades (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  exchange_id UUID REFERENCES crypto_exchanges(id) ON DELETE CASCADE,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL, -- 'buy', 'sell'
  amount DECIMAL NOT NULL,
  price DECIMAL NOT NULL,
  fee DECIMAL DEFAULT 0,
  timestamp TIMESTAMP NOT NULL,
  order_id TEXT,
  status TEXT NOT NULL -- 'open', 'closed', 'cancelled'
);

-- Chart Configurations
CREATE TABLE chart_configurations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  chart_name TEXT NOT NULL,
  chart_type TEXT NOT NULL, -- 'candlestick', 'line', 'volume'
  symbols TEXT[] NOT NULL,
  indicators TEXT[] NOT NULL,
  timeframes TEXT[] NOT NULL,
  layout_config JSONB NOT NULL,
  is_public BOOLEAN DEFAULT false,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Price Alerts
CREATE TABLE price_alerts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  symbol TEXT NOT NULL,
  alert_type TEXT NOT NULL, -- 'above', 'below', 'cross'
  target_price DECIMAL NOT NULL,
  is_active BOOLEAN DEFAULT true,
  triggered_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Strategy Sharing
CREATE TABLE strategy_likes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  strategy_id UUID REFERENCES saved_strategies(id) ON DELETE CASCADE,
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(user_id, strategy_id)
);

-- Comments on Strategies
CREATE TABLE strategy_comments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  strategy_id UUID REFERENCES saved_strategies(id) ON DELETE CASCADE,
  comment TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_user_api_keys_user_id ON user_api_keys(user_id);
CREATE INDEX idx_openbb_api_keys_user_id ON openbb_api_keys(user_id);
CREATE INDEX idx_mcp_sessions_user_id ON mcp_sessions(user_id);
CREATE INDEX idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX idx_trades_user_id ON trades(user_id);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_market_data_symbol_date ON market_data(symbol, date);
CREATE INDEX idx_technical_indicators_symbol ON technical_indicators(symbol);
CREATE INDEX idx_ml_models_user_id ON ml_models(user_id);
CREATE INDEX idx_model_predictions_user_id ON model_predictions(user_id);
CREATE INDEX idx_portfolio_optimizations_user_id ON portfolio_optimizations(user_id);
CREATE INDEX idx_risk_metrics_user_id ON risk_metrics(user_id);
CREATE INDEX idx_crypto_trades_user_id ON crypto_trades(user_id);
CREATE INDEX idx_price_alerts_user_id ON price_alerts(user_id);
CREATE INDEX idx_strategy_likes_user_id ON strategy_likes(user_id);
CREATE INDEX idx_strategy_comments_user_id ON strategy_comments(user_id);

-- Row Level Security (RLS) policies
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE openbb_api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE mcp_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE saved_strategies ENABLE ROW LEVEL SECURITY;
ALTER TABLE backtest_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio_optimizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE crypto_exchanges ENABLE ROW LEVEL SECURITY;
ALTER TABLE crypto_trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE chart_configurations ENABLE ROW LEVEL SECURITY;
ALTER TABLE price_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategy_likes ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategy_comments ENABLE ROW LEVEL SECURITY;

-- Create RLS policies (users can only access their own data)
CREATE POLICY "Users can view own data" ON users FOR ALL USING (auth.uid() = id);
CREATE POLICY "Users can view own api keys" ON user_api_keys FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can view own openbb keys" ON openbb_api_keys FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can view own mcp sessions" ON mcp_sessions FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can view own chat sessions" ON chat_sessions FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can view own chat messages" ON chat_messages FOR ALL USING (auth.uid() = session_id);
CREATE POLICY "Users can view own portfolios" ON portfolios FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can view own trades" ON trades FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can view own strategies" ON saved_strategies FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can view own backtests" ON backtest_results FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can view own models" ON ml_models FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can view own predictions" ON model_predictions FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can view own optimizations" ON portfolio_optimizations FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can view own risk metrics" ON risk_metrics FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can view own crypto exchanges" ON crypto_exchanges FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can view own crypto trades" ON crypto_trades FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can view own charts" ON chart_configurations FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can view own alerts" ON price_alerts FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can view own likes" ON strategy_likes FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Users can view own comments" ON strategy_comments FOR ALL USING (auth.uid() = user_id);

-- Public strategies can be viewed by anyone
CREATE POLICY "Public strategies are viewable" ON saved_strategies FOR SELECT USING (is_public = true);
CREATE POLICY "Public strategies can be liked" ON strategy_likes FOR INSERT WITH CHECK (true);
CREATE POLICY "Public strategies can be commented" ON strategy_comments FOR INSERT WITH CHECK (true);
