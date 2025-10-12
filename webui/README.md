# TradingAgents Web UI

This directory contains the web-based user interface for the TradingAgents framework, built with Dash and Flask.

## Structure

- `app_dash.py`: Main Dash application with UI components and callbacks
- `components/`: UI components and analysis functionality
  - `analysis.py`: Analysis runner and state management
  - `ui.py`: UI components and helpers
- `utils/`: Utility functions
  - `charts.py`: Chart creation utilities using Plotly
  - `state.py`: Application state management
  - `styles.py`: UI styling constants
- `assets/`: Static assets for the Dash application
  - `custom.css`: Custom CSS styles

## Running the Web UI

You can run the web UI using the helper script:

```bash
python run_webui_dash.py
```

Or directly from Python:

```python
from webui.app_dash import run_app

run_app(port=7860, debug=True)
```

## Features

- Interactive stock charts with technical indicators
- Real-time agent status updates
- Detailed analysis reports in a tabbed interface
- Configurable analysis parameters (ticker, date, analysts, LLMs)
- Dark mode UI optimized for financial data visualization

## Dependencies

- Dash: Web application framework (v3.0+)
- Flask: Backend server
- Plotly: Interactive charts
- Dash Bootstrap Components: UI components
- Pandas: Data manipulation
- yfinance: Financial data retrieval

## Customization

You can customize the UI by modifying the `app_dash.py` file and the CSS in the `assets/custom.css` file.

## Troubleshooting

- If you encounter an error related to `app.run_server`, make sure you're using `app.run` instead, as newer versions of Dash (3.0+) have deprecated the `run_server` method. 