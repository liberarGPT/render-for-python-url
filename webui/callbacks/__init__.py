"""
Callbacks package for TradingAgents WebUI
Contains organized callback functions grouped by functionality
"""

from .status_callbacks import register_status_callbacks
from .chart_callbacks import register_chart_callbacks  
from .report_callbacks import register_report_callbacks
from .control_callbacks import register_control_callbacks
from .trading_callbacks import register_trading_callbacks
from .storage_callbacks import register_storage_callbacks
from .chatbot_callbacks import register_chatbot_callbacks # Import the new chatbot callbacks
from webui.components.rss_ticker_feed import register_rss_callbacks # Import RSS feed callbacks
from webui.components.trending_tickers import register_trending_callbacks # Import trending tickers callbacks

def register_all_callbacks(app):
    """Register all callback functions with the Dash app"""
    register_status_callbacks(app)
    register_chart_callbacks(app)
    register_report_callbacks(app)
    register_control_callbacks(app)
    register_trading_callbacks(app)
    register_storage_callbacks(app)
    register_chatbot_callbacks(app) # Register the new chatbot callbacks
    register_rss_callbacks(app) # Register RSS feed callbacks
    register_trending_callbacks(app) # Register trending tickers callbacks