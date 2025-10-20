"""
Alpaca Backtrader Integration

This module provides integration between Alpaca API and Backtrader for backtesting and live trading.
"""
import os
import backtrader as bt
from alpaca_backtrader_api import AlpacaStore
from datetime import datetime, timedelta

class AlpacaBacktraderClient:
    def __init__(self, api_key=None, api_secret=None, paper=True):
        """
        Initialize Alpaca Backtrader client
        
        Args:
            api_key (str): Alpaca API key
            api_secret (str): Alpaca API secret
            paper (bool): Whether to use paper trading (default: True)
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.api_secret = api_secret or os.getenv('ALPACA_SECRET_KEY')
        self.paper = paper
        self.store = None
        self.data = None
        self.cerebro = None
        
    def connect(self):
        """Connect to Alpaca API"""
        try:
            self.store = AlpacaStore(
                key_id=self.api_key,
                secret_key=self.api_secret,
                paper=self.paper,
                usePolygon=False  # Set to True if you have a Polygon subscription
            )
            return True
        except Exception as e:
            print(f"Error connecting to Alpaca: {str(e)}")
            return False
    
    def get_broker(self):
        """Get Alpaca broker instance"""
        if not self.store:
            self.connect()
        return self.store.getbroker()
    
    def get_data(self, symbol, timeframe=bt.TimeFrame.Days, fromdate=None, todate=None):
        """
        Get historical data from Alpaca
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            timeframe: Backtrader timeframe (e.g., bt.TimeFrame.Days)
            fromdate (datetime): Start date
            todate (datetime): End date
            
        Returns:
            Backtrader data feed or None if error
        """
        if not self.store:
            self.connect()
            
        if not fromdate:
            fromdate = datetime.utcnow() - timedelta(days=365)  # Default to 1 year of data
        if not todate:
            todate = datetime.utcnow()
            
        try:
            self.data = self.store.getdata(
                dataname=symbol,
                timeframe=timeframe,
                fromdate=fromdate,
                todate=todate,
                historical=True
            )
            return self.data
        except Exception as e:
            print(f"Error getting data: {str(e)}")
            return None
    
    def run_backtest(self, strategy, symbol, **kwargs):
        """
        Run a backtest with the given strategy
        
        Args:
            strategy (bt.Strategy): Backtrader strategy class
            symbol (str): Stock symbol to backtest
            **kwargs: Additional arguments to pass to the strategy
            
        Returns:
            dict: Backtest results
        """
        # Initialize Cerebro engine
        self.cerebro = bt.Cerebro()
        
        # Add the data feed
        data = self.get_data(symbol, **kwargs)
        if not data:
            return {"status": "error", "message": "Failed to get data"}
            
        self.cerebro.adddata(data)
        
        # Add the strategy
        self.cerebro.addstrategy(strategy, **kwargs)
        
        # Set the broker
        broker = self.get_broker()
        self.cerebro.setbroker(broker)
        
        # Run the backtest
        try:
            print("Starting Portfolio Value: %.2f" % self.cerebro.broker.getvalue())
            results = self.cerebro.run()
            print("Final Portfolio Value: %.2f" % self.cerebro.broker.getvalue())
            
            # Get strategy results
            strat = results[0]
            return {
                "status": "success",
                "initial_value": self.cerebro.broker.startingcash,
                "final_value": self.cerebro.broker.getvalue(),
                "return_pct": (self.cerebro.broker.getvalue() / self.cerebro.broker.startingcash - 1) * 100,
                "trades": len(strat.analyzers.trades.get_analysis() if hasattr(strat, 'analyzers') and 'trades' in strat.analyzers else [])
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Example Strategy
class SmaCross(bt.Strategy):
    params = (
        ('fast', 10),
        ('slow', 30),
        ('printlog', True),
    )
    
    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}, {txt}")
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        
        # Add indicators
        sma1 = bt.ind.SMA(period=self.p.fast)
        sma2 = bt.ind.SMA(period=self.p.slow)
        self.crossover = bt.ind.CrossOver(sma1, sma2)
        
    def next(self):
        if self.order:
            return
            
        if not self.position:
            if self.crossover > 0:  # Fast crosses above slow
                self.log(f"BUY CREATE, {self.dataclose[0]:.2f}")
                self.order = self.buy()
        else:
            if self.crossover < 0:  # Fast crosses below slow
                self.log(f"SELL CREATE, {self.dataclose[0]:.2f}")
                self.order = self.sell()
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
            elif order.issell():
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order Canceled/Margin/Rejected: {order.getstatusname()}")
            
        self.order = None

# Example usage
def run_example():
    # Initialize client
    client = AlpacaBacktraderClient()
    
    # Run backtest with example strategy
    results = client.run_backtest(
        strategy=SmaCross,
        symbol="AAPL",
        fast=10,
        slow=30,
        printlog=True
    )
    
    print("\nBacktest Results:")
    print(f"Initial Value: ${results['initial_value']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Return: {results['return_pct']:.2f}%")
    print(f"Trades: {results['trades']}")

if __name__ == "__main__":
    run_example()
