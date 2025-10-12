import dash_bootstrap_components as dbc
from dash import html, dcc

def create_chatbot_panel():
    """
    Creates the chatbot panel component for the web UI.
    """
    return dbc.Card(
        dbc.CardBody([
            html.H4([
                html.I(className="fas fa-robot me-2"),
                "AI Chatbot (Gemini)"
            ], className="mb-3"),
            html.Hr(),
            html.Div(
                id="chatbot-conversation-container",
                className="mb-3",
                style={
                    "height": "400px",
                    "overflowY": "auto",
                    "border": "1px solid #334155",
                    "borderRadius": "8px",
                    "padding": "15px",
                    "backgroundColor": "#1E293B",
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "10px"
                }
            ),
            dbc.InputGroup(
                [
                    dbc.Input(
                        id="chatbot-input",
                        type="text",
                        placeholder="Ask Gemini about the market or tickers...",
                        className="form-control"
                    ),
                    dbc.Button(
                        [
                            html.I(className="fas fa-paper-plane me-2"),
                            "Send"
                        ],
                        id="chatbot-send-btn",
                        color="primary",
                        className="btn-primary"
                    ),
                ],
                className="mb-2"
            ),
            dbc.Button(
                [
                    html.I(className="fas fa-sync-alt me-2"),
                    "Clear Chat"
                ],
                id="chatbot-clear-btn",
                color="secondary",
                outline=True,
                size="sm",
                className="w-100"
            ),
            html.Div(id="chatbot-status-message", className="mt-2 text-muted small")
        ]),
        className="mb-4"
    )