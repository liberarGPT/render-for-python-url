-- Complete Supabase Database Schema
-- Run this in your Supabase SQL Editor to set up the full database

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create profiles table (extends auth.users)
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    avatar_url TEXT,
    subscription_tier TEXT DEFAULT 'free' CHECK (subscription_tier IN ('free', 'pro', 'enterprise')),
    subscription_end TIMESTAMPTZ,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create user API keys table
CREATE TABLE IF NOT EXISTS public.user_api_keys (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
    
    -- Alpaca API Keys
    alpaca_api_key_encrypted BYTEA,
    alpaca_secret_key_encrypted BYTEA,
    alpaca_paper_trading BOOLEAN DEFAULT TRUE,
    
    -- Other API Keys
    finnhub_api_key_encrypted BYTEA,
    cryptocompare_api_key_encrypted BYTEA,
    alpha_vantage_api_key_encrypted BYTEA,
    polygon_api_key_encrypted BYTEA,
    
    -- OpenBB API Keys
    openbb_pat_token_encrypted BYTEA,
    
    -- MCP Configuration
    mcp_enabled BOOLEAN DEFAULT FALSE,
    mcp_config JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create trading strategies table
CREATE TABLE IF NOT EXISTS public.trading_strategies (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    strategy_type TEXT CHECK (strategy_type IN ('momentum', 'mean_reversion', 'arbitrage', 'ml', 'custom')),
    code TEXT, -- Python code for the strategy
    parameters JSONB, -- Strategy parameters
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create backtests table
CREATE TABLE IF NOT EXISTS public.backtests (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
    strategy_id UUID REFERENCES public.trading_strategies(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(15,2) NOT NULL,
    final_capital DECIMAL(15,2),
    total_return DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    win_rate DECIMAL(5,4),
    results JSONB, -- Detailed backtest results
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create trades table
CREATE TABLE IF NOT EXISTS public.trades (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
    strategy_id UUID REFERENCES public.trading_strategies(id),
    symbol TEXT NOT NULL,
    side TEXT CHECK (side IN ('buy', 'sell')) NOT NULL,
    quantity DECIMAL(15,8) NOT NULL,
    price DECIMAL(15,4) NOT NULL,
    total_value DECIMAL(15,2) NOT NULL,
    commission DECIMAL(10,4) DEFAULT 0,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'filled', 'cancelled', 'rejected'))
);

-- Create AI chat sessions table
CREATE TABLE IF NOT EXISTS public.ai_chat_sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
    session_name TEXT,
    context TEXT, -- Chat context/type (strategy_building, market_analysis, etc.)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create AI chat messages table
CREATE TABLE IF NOT EXISTS public.ai_chat_messages (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES public.ai_chat_sessions(id) ON DELETE CASCADE NOT NULL,
    role TEXT CHECK (role IN ('user', 'assistant', 'system')) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB, -- Additional message metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create OpenBB data cache table
CREATE TABLE IF NOT EXISTS public.openbb_data_cache (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    data_type TEXT NOT NULL, -- 'equity', 'crypto', 'forex', 'etf', etc.
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL, -- '1d', '1h', '5m', etc.
    data JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create MCP sessions table
CREATE TABLE IF NOT EXISTS public.mcp_sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
    session_name TEXT,
    tools_used JSONB, -- List of MCP tools used in this session
    context JSONB, -- Session context and configuration
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create MCP tool usage table
CREATE TABLE IF NOT EXISTS public.mcp_tool_usage (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES public.mcp_sessions(id) ON DELETE CASCADE NOT NULL,
    tool_name TEXT NOT NULL,
    input_data JSONB,
    output_data JSONB,
    execution_time_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create encryption/decryption functions
CREATE OR REPLACE FUNCTION public.encrypt_api_key(key_text TEXT)
RETURNS BYTEA
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  encryption_key TEXT;
BEGIN
  -- Get encryption key from environment
  encryption_key := current_setting('app.encryption_key', true);
  
  -- If no key set, use a default for development
  IF encryption_key IS NULL OR encryption_key = '' THEN
    encryption_key := 'default-development-key-change-in-production';
  END IF;
  
  -- Use pgp_sym_encrypt with the key
  RETURN pgp_sym_encrypt(key_text, encryption_key);
END;
$$;

CREATE OR REPLACE FUNCTION public.decrypt_api_key(encrypted_data BYTEA)
RETURNS TEXT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  encryption_key TEXT;
BEGIN
  IF encrypted_data IS NULL THEN
    RETURN NULL;
  END IF;
  
  -- Get encryption key from environment
  encryption_key := current_setting('app.encryption_key', true);
  
  -- If no key set, use a default for development
  IF encryption_key IS NULL OR encryption_key = '' THEN
    encryption_key := 'default-development-key-change-in-production';
  END IF;
  
  -- Use pgp_sym_decrypt with the key
  RETURN pgp_sym_decrypt(encrypted_data, encryption_key);
END;
$$;

-- Create RLS policies
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.trading_strategies ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.backtests ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.openbb_data_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.mcp_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.mcp_tool_usage ENABLE ROW LEVEL SECURITY;

-- Profiles policies
CREATE POLICY "Users can view own profile" ON public.profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.profiles
    FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can delete own profile" ON public.profiles
    FOR DELETE USING (auth.uid() = id);

-- API Keys policies
CREATE POLICY "Users can manage own API keys" ON public.user_api_keys
    FOR ALL USING (auth.uid() = user_id);

-- Trading strategies policies
CREATE POLICY "Users can manage own strategies" ON public.trading_strategies
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can view public strategies" ON public.trading_strategies
    FOR SELECT USING (is_public = true);

-- Backtests policies
CREATE POLICY "Users can manage own backtests" ON public.backtests
    FOR ALL USING (auth.uid() = user_id);

-- Trades policies
CREATE POLICY "Users can manage own trades" ON public.trades
    FOR ALL USING (auth.uid() = user_id);

-- AI Chat policies
CREATE POLICY "Users can manage own chat sessions" ON public.ai_chat_sessions
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can manage own chat messages" ON public.ai_chat_messages
    FOR ALL USING (auth.uid() = (SELECT user_id FROM public.ai_chat_sessions WHERE id = session_id));

-- OpenBB data cache policies
CREATE POLICY "Users can manage own data cache" ON public.openbb_data_cache
    FOR ALL USING (auth.uid() = user_id OR user_id IS NULL);

-- MCP policies
CREATE POLICY "Users can manage own MCP sessions" ON public.mcp_sessions
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can manage own MCP tool usage" ON public.mcp_tool_usage
    FOR ALL USING (auth.uid() = (SELECT user_id FROM public.mcp_sessions WHERE id = session_id));

-- Grant permissions
GRANT EXECUTE ON FUNCTION public.encrypt_api_key(TEXT) TO authenticated;
GRANT EXECUTE ON FUNCTION public.decrypt_api_key(BYTEA) TO authenticated;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_user_api_keys_user_id ON public.user_api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_trading_strategies_user_id ON public.trading_strategies(user_id);
CREATE INDEX IF NOT EXISTS idx_backtests_user_id ON public.backtests(user_id);
CREATE INDEX IF NOT EXISTS idx_trades_user_id ON public.trades(user_id);
CREATE INDEX IF NOT EXISTS idx_ai_chat_sessions_user_id ON public.ai_chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_ai_chat_messages_session_id ON public.ai_chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_openbb_data_cache_expires ON public.openbb_data_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_mcp_sessions_user_id ON public.mcp_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_mcp_tool_usage_session_id ON public.mcp_tool_usage(session_id);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add updated_at triggers
CREATE TRIGGER handle_updated_at_profiles
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER handle_updated_at_user_api_keys
    BEFORE UPDATE ON public.user_api_keys
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER handle_updated_at_trading_strategies
    BEFORE UPDATE ON public.trading_strategies
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER handle_updated_at_ai_chat_sessions
    BEFORE UPDATE ON public.ai_chat_sessions
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER handle_updated_at_mcp_sessions
    BEFORE UPDATE ON public.mcp_sessions
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();
