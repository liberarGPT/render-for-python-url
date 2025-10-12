"""
Trending Tickers Component
Fetches trending tickers from multiple sources including unusual options activity.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import requests
from datetime import datetime, timedelta
import time
import threading
import queue
from typing import List, Dict, Any
import logging
from bs4 import BeautifulSoup
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendingTickersFeed:
    """Trending tickers monitor that fetches from multiple sources."""
    
    def __init__(self):
        self.trending_queue = queue.Queue()
        self.trending_history = []  # Store persistent history
        self.is_running = True  # Auto-start
        self.start_monitoring()  # Auto-start monitoring
        
        # Headers to mimic a real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def fetch_finviz_unusual_volume(self) -> List[Dict]:
        """Fetch unusual volume data from Finviz."""
        try:
            # Finviz unusual volume page
            url = "https://finviz.com/screener.ashx?v=110&s=uo_volume"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            trending_data = []
            
            # Find the table with stock data
            table = soup.find('table', class_='table-light')
            if table:
                rows = table.find_all('tr')[1:11]  # Skip header, get top 10
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        ticker = cells[1].get_text(strip=True)
                        company = cells[2].get_text(strip=True)
                        volume = cells[5].get_text(strip=True) if len(cells) > 5 else "N/A"
                        
                        if ticker and ticker != 'Ticker':
                            trending_data.append({
                                'ticker': ticker,
                                'company': company,
                                'volume': volume,
                                'source': 'Finviz Unusual Volume',
                                'timestamp': datetime.now(),
                                'type': 'unusual_volume'
                            })
            
            logger.info(f"Found {len(trending_data)} unusual volume tickers from Finviz")
            return trending_data
            
        except Exception as e:
            logger.error(f"Error fetching Finviz data: {e}")
            return []
    
    def fetch_finviz_top_gainers(self) -> List[Dict]:
        """Fetch top gainers from Finviz."""
        try:
            # Finviz top gainers page
            url = "https://finviz.com/screener.ashx?v=110&s=ta_topgainers"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            trending_data = []
            
            # Find the table with stock data
            table = soup.find('table', class_='table-light')
            if table:
                rows = table.find_all('tr')[1:11]  # Skip header, get top 10
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        ticker = cells[1].get_text(strip=True)
                        company = cells[2].get_text(strip=True)
                        change = cells[8].get_text(strip=True) if len(cells) > 8 else "N/A"
                        
                        if ticker and ticker != 'Ticker':
                            trending_data.append({
                                'ticker': ticker,
                                'company': company,
                                'change': change,
                                'source': 'Finviz Top Gainers',
                                'timestamp': datetime.now(),
                                'type': 'top_gainer'
                            })
            
            logger.info(f"Found {len(trending_data)} top gainers from Finviz")
            return trending_data
            
        except Exception as e:
            logger.error(f"Error fetching Finviz gainers: {e}")
            return []
    
    def fetch_barchart_unusual_options(self) -> List[Dict]:
        """Fetch unusual options activity from Barchart."""
        try:
            # Barchart unusual options page
            url = "https://www.barchart.com/options/unusual-activity/stocks"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            trending_data = []
            
            # Look for options data (this might need adjustment based on actual page structure)
            options_rows = soup.find_all('tr', class_='bc-datatable-row')
            
            for row in options_rows[:10]:  # Limit to top 10
                cells = row.find_all('td')
                if len(cells) >= 4:
                    ticker_cell = cells[0].find('a')
                    if ticker_cell:
                        ticker = ticker_cell.get_text(strip=True)
                        volume = cells[3].get_text(strip=True) if len(cells) > 3 else "N/A"
                        
                        if ticker:
                            trending_data.append({
                                'ticker': ticker,
                                'company': ticker,  # Barchart might not show company names
                                'volume': volume,
                                'source': 'Barchart Unusual Options',
                                'timestamp': datetime.now(),
                                'type': 'unusual_options'
                            })
            
            logger.info(f"Found {len(trending_data)} unusual options from Barchart")
            return trending_data
            
        except Exception as e:
            logger.error(f"Error fetching Barchart data: {e}")
            return []
    
    def fetch_trending_data(self):
        """Fetch trending data from all sources."""
        all_trending = []
        
        # Fetch from different sources
        try:
            all_trending.extend(self.fetch_finviz_unusual_volume())
            time.sleep(2)  # Rate limiting
            
            all_trending.extend(self.fetch_finviz_top_gainers())
            time.sleep(2)  # Rate limiting
            
            all_trending.extend(self.fetch_barchart_unusual_options())
            
        except Exception as e:
            logger.error(f"Error in fetch_trending_data: {e}")
        
        # Add to queue and history
        for item in all_trending:
            self.trending_queue.put(item)
            self.trending_history.append(item)
            
        # Keep only last 100 items
        if len(self.trending_history) > 100:
            self.trending_history = self.trending_history[-100:]
        
        logger.info(f"Total trending items collected: {len(all_trending)}")
    
    def start_monitoring(self):
        """Start the trending monitoring in a separate thread."""
        if self.is_running:
            return
            
        self.is_running = True
        
        def monitor_loop():
            while self.is_running:
                try:
                    self.fetch_trending_data()
                    time.sleep(300)  # Check every 5 minutes (rate limiting)
                except Exception as e:
                    logger.error(f"Error in trending monitoring loop: {e}")
                    time.sleep(600)  # Wait 10 minutes on error
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("Trending tickers monitoring started")
    
    def stop_monitoring(self):
        """Stop the trending monitoring."""
        self.is_running = False
        logger.info("Trending tickers monitoring stopped")
    
    def get_latest_trending(self, limit: int = 15) -> List[Dict]:
        """Get the latest trending tickers from the queue and history."""
        trending = []
        
        # Get new items from queue
        while not self.trending_queue.empty():
            try:
                item = self.trending_queue.get_nowait()
                trending.append(item)
            except queue.Empty:
                break
        
        # If no new items, show recent history
        if not trending and hasattr(self, 'trending_history') and self.trending_history:
            trending = self.trending_history[-limit:]
        
        # Sort by timestamp (newest first) and limit results
        trending.sort(key=lambda x: x['timestamp'], reverse=True)
        return trending[:limit]

# Global instance
trending_feed = TrendingTickersFeed()

def create_trending_tickers_layout():
    """Create the trending tickers layout."""
    return dbc.Card([
        dbc.CardHeader([
            html.H4("📈 Trending Tickers", className="mb-0"),
            dbc.Badge("Live", color="info", className="ms-2")
        ]),
        dbc.CardBody([
            # Status display
            dbc.Row([
                dbc.Col([
                    dbc.Badge("Status: Auto-Running", id="trending-status-badge", color="info")
                ], width=6),
                dbc.Col([
                    dbc.Button("Clear Data", id="clear-trending-data", color="warning", size="sm")
                ], width=6)
            ], className="mb-3"),
            
            # Auto-refresh interval
            dbc.Row([
                dbc.Col([
                    html.Label("Auto-refresh (minutes):"),
                    dbc.Input(
                        id="trending-refresh-interval",
                        type="number",
                        value=5,
                        min=1,
                        max=60,
                        size="sm"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Max results:"),
                    dbc.Input(
                        id="trending-max-results",
                        type="number",
                        value=15,
                        min=5,
                        max=50,
                        size="sm"
                    )
                ], width=6)
            ], className="mb-3"),
            
            # Trending display
            html.Div(id="trending-tickers-display"),
            
            # Auto-refresh component
            dcc.Interval(
                id='trending-interval-component',
                interval=300000,  # 5 minutes
                n_intervals=0
            )
        ])
    ], className="mb-4")

def register_trending_callbacks(app):
    """Register trending tickers callbacks."""
    
    @app.callback(
        [Output('trending-tickers-display', 'children'),
         Output('trending-status-badge', 'children'),
         Output('trending-status-badge', 'color')],
        [Input('trending-interval-component', 'n_intervals'),
         Input('clear-trending-data', 'n_clicks')],
        [State('trending-refresh-interval', 'value'),
         State('trending-max-results', 'value')]
    )
    def update_trending_tickers(n_intervals, clear_clicks, refresh_interval, max_results):
        """Update the trending tickers display."""
        ctx = dash.callback_context
        
        if not ctx.triggered:
            # Show persistent history on first load
            status_text = "Status: Auto-Running"
            status_color = "info"
        else:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # Handle button clicks
            if trigger_id == 'clear-trending-data':
                # Clear the queue and history
                while not trending_feed.trending_queue.empty():
                    try:
                        trending_feed.trending_queue.get_nowait()
                    except queue.Empty:
                        break
                trending_feed.trending_history = []
                status_text = "Status: Cleared"
                status_color = "warning"
            else:
                # Auto-refresh
                status_text = "Status: Auto-Running"
                status_color = "info"
        
        # Get latest trending data
        trending_data = trending_feed.get_latest_trending(max_results or 15)
        
        if not trending_data:
            trending_content = dbc.Alert("No trending data yet. Sources are being monitored...", 
                                       color="info")
        else:
            trending_items = []
            for item in trending_data:
                # Determine badge color based on type
                if item['type'] == 'unusual_options':
                    badge_color = "danger"
                elif item['type'] == 'unusual_volume':
                    badge_color = "warning"
                elif item['type'] == 'top_gainer':
                    badge_color = "success"
                else:
                    badge_color = "primary"
                
                trending_items.append(
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H6(item['ticker'], className="mb-1"),
                                    html.Small(item['company'], className="text-muted")
                                ], width=6),
                                dbc.Col([
                                    dbc.Badge(item['type'].replace('_', ' ').title(), 
                                            color=badge_color, size="sm"),
                                    html.Br(),
                                    html.Small(item['source'], className="text-muted")
                                ], width=6)
                            ]),
                            html.Hr(className="my-2"),
                            html.Small([
                                f"Updated: {item['timestamp'].strftime('%H:%M:%S')}"
                            ], className="text-muted")
                        ])
                    ], className="mb-2")
                )
            
            trending_content = html.Div(trending_items)
        
        return trending_content, status_text, status_color
