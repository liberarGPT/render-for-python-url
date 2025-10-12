"""
Storage utility for persisting user settings in localStorage
"""

from typing import Dict, Any

# Default settings structure
DEFAULT_SETTINGS = {
    "ticker_input": "NVDA, AMD, TSLA",
    "analyst_market": True,
    "analyst_social": True,
    "analyst_news": True,
    "analyst_fundamentals": True,
    "analyst_macro": True,
    "research_depth": "Shallow",
    "allow_shorts": False,
    "loop_enabled": False,
    "loop_interval": 60,
    "market_hour_enabled": False,
    "market_hours_input": "",
    "trade_after_analyze": False,
    "trade_dollar_amount": 4500,
    "quick_llm": "gpt-5-nano",
    "deep_llm": "gpt-5-nano"
}

def get_default_settings() -> Dict[str, Any]:
    """Get the default settings structure"""
    return DEFAULT_SETTINGS.copy()

def create_storage_store_component():
    """Create a dcc.Store component for localStorage persistence"""
    from dash import dcc
    return dcc.Store(id='settings-store', storage_type='local', data=DEFAULT_SETTINGS)
