"""
RSS Ticker Feed Component
Continuously monitors news feeds and suggests relevant tickers based on news content.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import feedparser
import requests
from datetime import datetime, timedelta
import time
import threading
import queue
import re
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RSSTickerFeed:
    """RSS feed monitor that suggests tickers based on news content."""
    
    def __init__(self):
        self.rss_feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://feeds.marketwatch.com/marketwatch/topstories/",
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "https://feeds.reuters.com/news/wealth",
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US",
        ]
        
        # Common ticker patterns and company keywords
        self.ticker_keywords = {
            'AAPL': ['apple', 'iphone', 'macbook', 'ipad', 'services', 'app store'],
            'MSFT': ['microsoft', 'azure', 'office', 'windows', 'xbox', 'linkedin'],
            'GOOGL': ['google', 'alphabet', 'search', 'youtube', 'android', 'chrome'],
            'AMZN': ['amazon', 'aws', 'prime', 'alexa', 'ecommerce', 'retail'],
            'TSLA': ['tesla', 'elon musk', 'electric vehicle', 'ev', 'autopilot', 'cybertruck'],
            'NVDA': ['nvidia', 'ai', 'gpu', 'gaming', 'data center', 'cuda'],
            'META': ['meta', 'facebook', 'instagram', 'whatsapp', 'vr', 'metaverse'],
            'NFLX': ['netflix', 'streaming', 'content', 'subscription'],
            'AMD': ['amd', 'processor', 'cpu', 'gpu', 'semiconductor'],
            'INTC': ['intel', 'processor', 'cpu', 'semiconductor', 'foundry'],
            'JPM': ['jpmorgan', 'jp morgan', 'banking', 'financial services'],
            'BAC': ['bank of america', 'bofa', 'banking', 'financial services'],
            'WMT': ['walmart', 'retail', 'grocery', 'supermarket'],
            'JNJ': ['johnson & johnson', 'pharmaceutical', 'healthcare', 'vaccine'],
            'PG': ['procter & gamble', 'consumer goods', 'household products'],
            'KO': ['coca cola', 'beverage', 'soft drink', 'soda'],
            'PFE': ['pfizer', 'pharmaceutical', 'vaccine', 'medicine'],
            'DIS': ['disney', 'entertainment', 'streaming', 'theme park'],
            'ADBE': ['adobe', 'creative software', 'photoshop', 'pdf'],
            'CRM': ['salesforce', 'crm', 'saas', 'cloud software'],
        }
        
        self.suggestion_queue = queue.Queue()
        self.suggestions_history = []  # Store persistent history
        self.is_running = True  # Auto-start
        self.start_monitoring()  # Auto-start monitoring
        
    def extract_tickers_from_text(self, text: str) -> List[str]:
        """Extract potential tickers from text based on keywords and patterns."""
        text_lower = text.lower()
        suggested_tickers = []
        
        # Check for direct ticker mentions (3-5 letter uppercase codes)
        ticker_pattern = r'\b[A-Z]{3,5}\b'
        potential_tickers = re.findall(ticker_pattern, text.upper())
        
        # Filter for known tickers
        for ticker in potential_tickers:
            if ticker in self.ticker_keywords:
                suggested_tickers.append(ticker)
        
        # Check for company keyword matches
        for ticker, keywords in self.ticker_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    suggested_tickers.append(ticker)
                    break
        
        return list(set(suggested_tickers))  # Remove duplicates
    
    def fetch_news_feeds(self):
        """Fetch news from RSS feeds and extract ticker suggestions."""
        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:10]:  # Limit to recent 10 entries
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    link = entry.get('link', '')
                    published = entry.get('published_parsed', None)
                    
                    # Combine title and summary for analysis
                    full_text = f"{title} {summary}"
                    
                    # Extract ticker suggestions
                    suggested_tickers = self.extract_tickers_from_text(full_text)
                    
                    if suggested_tickers:
                        suggestion = {
                            'timestamp': datetime.now(),
                            'title': title,
                            'summary': summary[:200] + "..." if len(summary) > 200 else summary,
                            'link': link,
                            'suggested_tickers': suggested_tickers,
                            'confidence': len(suggested_tickers) / len(self.ticker_keywords) * 100
                        }
                        
                        self.suggestion_queue.put(suggestion)
                        self.suggestions_history.append(suggestion)  # Keep persistent history
                        # Keep only last 100 suggestions
                        if len(self.suggestions_history) > 100:
                            self.suggestions_history = self.suggestions_history[-100:]
                        logger.info(f"Found tickers {suggested_tickers} in: {title[:50]}...")
                
            except Exception as e:
                logger.error(f"Error fetching feed {feed_url}: {e}")
        
        # Small delay between feed checks
        time.sleep(2)
    
    def start_monitoring(self):
        """Start the RSS monitoring in a separate thread."""
        if self.is_running:
            return
            
        self.is_running = True
        
        def monitor_loop():
            while self.is_running:
                try:
                    self.fetch_news_feeds()
                    time.sleep(30)  # Check feeds every 30 seconds
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(60)  # Wait longer on error
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("RSS ticker monitoring started")
    
    def stop_monitoring(self):
        """Stop the RSS monitoring."""
        self.is_running = False
        logger.info("RSS ticker monitoring stopped")
    
    def get_latest_suggestions(self, limit: int = 10) -> List[Dict]:
        """Get the latest ticker suggestions from the queue and history."""
        suggestions = []
        
        # Get new suggestions from queue
        while not self.suggestion_queue.empty():
            try:
                suggestion = self.suggestion_queue.get_nowait()
                suggestions.append(suggestion)
            except queue.Empty:
                break
        
        # If no new suggestions, show recent history
        if not suggestions and hasattr(self, 'suggestions_history') and self.suggestions_history:
            suggestions = self.suggestions_history[-limit:]
        
        # Sort by timestamp (newest first) and limit results
        suggestions.sort(key=lambda x: x['timestamp'], reverse=True)
        return suggestions[:limit]

# Global instance
rss_feed = RSSTickerFeed()

def create_rss_ticker_feed_layout():
    """Create the RSS ticker feed layout."""
    return dbc.Card([
        dbc.CardHeader([
            html.H4("📰 RSS Ticker Feed", className="mb-0"),
            dbc.Badge("Live", color="success", className="ms-2")
        ]),
        dbc.CardBody([
            # Status display only (auto-starting)
            dbc.Row([
                dbc.Col([
                    dbc.Badge("Status: Auto-Running", id="rss-status-badge", color="success")
                ], width=6),
                dbc.Col([
                    dbc.Button("Clear Feed", id="clear-rss-feed", color="warning", size="sm")
                ], width=6)
            ], className="mb-3"),
            
            # Auto-refresh interval
            dbc.Row([
                dbc.Col([
                    html.Label("Auto-refresh (seconds):"),
                    dbc.Input(
                        id="rss-refresh-interval",
                        type="number",
                        value=5,
                        min=1,
                        max=60,
                        size="sm"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Max suggestions:"),
                    dbc.Input(
                        id="rss-max-suggestions",
                        type="number",
                        value=10,
                        min=1,
                        max=50,
                        size="sm"
                    )
                ], width=6)
            ], className="mb-3"),
            
            # Feed display
            html.Div(id="rss-feed-display"),
            
            # Auto-refresh component
            dcc.Interval(
                id='rss-interval-component',
                interval=5000,  # 5 seconds
                n_intervals=0
            )
        ])
    ], className="mb-4")

def register_rss_callbacks(app):
    """Register RSS feed callbacks."""
    
    @app.callback(
        [Output('rss-feed-display', 'children'),
         Output('rss-status-badge', 'children'),
         Output('rss-status-badge', 'color')],
        [Input('rss-interval-component', 'n_intervals'),
         Input('clear-rss-feed', 'n_clicks')],
        [State('rss-refresh-interval', 'value'),
         State('rss-max-suggestions', 'value')]
    )
    def update_rss_feed(n_intervals, clear_clicks, refresh_interval, max_suggestions):
        """Update the RSS feed display."""
        ctx = dash.callback_context
        
        if not ctx.triggered:
            # Show persistent history on first load
            status_text = "Status: Auto-Running"
            status_color = "success"
        else:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # Handle button clicks
            if trigger_id == 'clear-rss-feed':
                # Clear the queue and history
                while not rss_feed.suggestion_queue.empty():
                    try:
                        rss_feed.suggestion_queue.get_nowait()
                    except queue.Empty:
                        break
                rss_feed.suggestions_history = []
                status_text = "Status: Cleared"
                status_color = "warning"
            else:
                # Auto-refresh
                status_text = "Status: Auto-Running"
                status_color = "success"
        
        # Get latest suggestions
        suggestions = rss_feed.get_latest_suggestions(max_suggestions or 10)
        
        if not suggestions:
            feed_content = dbc.Alert("No ticker suggestions yet. Feed is monitoring news sources...", 
                                   color="info")
        else:
            feed_items = []
            for suggestion in suggestions:
                ticker_badges = [
                    dbc.Badge(ticker, color="primary", className="me-1")
                    for ticker in suggestion['suggested_tickers']
                ]
                
                feed_items.append(
                    dbc.Card([
                        dbc.CardBody([
                            html.H6(suggestion['title'], className="card-title", style={"fontSize": "14px"}),
                            html.P(suggestion['summary'], className="card-text", style={"fontSize": "12px"}),
                            html.Div(ticker_badges, className="mb-2"),
                            html.Small([
                                f"Suggested: {suggestion['timestamp'].strftime('%H:%M:%S')} | ",
                                html.A("Read More", href=suggestion['link'], target="_blank")
                            ], className="text-muted")
                        ])
                    ], className="mb-2")
                )
            
            feed_content = html.Div(feed_items)
        
        return feed_content, status_text, status_color
