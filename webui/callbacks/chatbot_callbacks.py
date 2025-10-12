from dash import Input, Output, State, html, callback_context as ctx
import dash_bootstrap_components as dbc
import dash
import uuid

from webui.utils.gemini_chatbot import get_gemini_chatbot
from webui.utils.state import app_state

def register_chatbot_callbacks(app):
    """
    Registers all chatbot-related callbacks.
    """

    @app.callback(
        [Output("chatbot-conversation-container", "children"),
         Output("chatbot-input", "value"),
         Output("chatbot-status-message", "children")],
        [Input("chatbot-send-btn", "n_clicks"),
         Input("chatbot-input", "n_submit"),
         Input("chatbot-clear-btn", "n_clicks")],
        [State("chatbot-input", "value"),
         State("chatbot-conversation-container", "children"),
         State("app-store", "data")],
        prevent_initial_call=True
    )
    def handle_chatbot_interaction(send_clicks, submit_clicks, clear_clicks,
                                   user_message, current_conversation, app_store_data):
        trigger_id = ctx.triggered_id if ctx.triggered_id else ""
        
        # Ensure current_conversation is a list
        if not isinstance(current_conversation, list):
            current_conversation = []

        session_id = app_store_data.get("session_id", str(uuid.uuid4()))
        
        if "chatbot-clear-btn" in trigger_id:
            get_gemini_chatbot().clear_history(session_id)
            return [], "", "Chat history cleared."

        if ("chatbot-send-btn" in trigger_id or "chatbot-input" in trigger_id) and user_message:
            # Add user message to conversation
            current_conversation.append(
                html.Div(
                    html.Div(
                        user_message,
                        style={
                            "backgroundColor": "#3B82F6",
                            "color": "white",
                            "padding": "10px 15px",
                            "borderRadius": "10px",
                            "marginLeft": "20%",
                            "wordWrap": "break-word",
                            "maxWidth": "80%",
                            "whiteSpace": "pre-wrap"
                        }
                    ),
                    className="mb-2"
                )
            )
            
            # Get current app context for Gemini
            context = ""
            current_symbol = app_state.current_symbol
            if current_symbol:
                symbol_state = app_state.get_state(current_symbol)
                if symbol_state:
                    context_parts = []
                    if symbol_state.get("analysis_complete"):
                        context_parts.append(f"Analysis for {current_symbol} is complete.")
                        if symbol_state.get("analysis_results", {}).get("decision"):
                            context_parts.append(f"Final decision: {symbol_state['analysis_results']['decision']}.")
                        if symbol_state.get("current_reports", {}).get("final_trade_decision"):
                            context_parts.append(f"Portfolio Manager's final trade decision: {symbol_state['current_reports']['final_trade_decision']}.")
                    else:
                        context_parts.append(f"Analysis for {current_symbol} is in progress.")
                        # Add partial reports if available
                        for report_type, content in symbol_state.get("current_reports", {}).items():
                            if content and content.strip() and len(content) > 50: # Only add substantial content
                                context_parts.append(f"Partial {report_type.replace('_', ' ').title()}: {content[:200]}...") # Truncate for brevity
                    
                    if context_parts:
                        context = " ".join(context_parts)
            
            # Send message to Gemini
            gemini_response = get_gemini_chatbot().send_message(session_id, user_message, context)
            
            # Add Gemini response to conversation
            current_conversation.append(
                html.Div(
                    html.Div(
                        gemini_response,
                        style={
                            "backgroundColor": "#6B7280",
                            "color": "white",
                            "padding": "10px 15px",
                            "borderRadius": "10px",
                            "marginRight": "20%",
                            "wordWrap": "break-word",
                            "maxWidth": "80%",
                            "whiteSpace": "pre-wrap"
                        }
                    ),
                    className="mb-2"
                )
            )
            
            return current_conversation, "", ""
        
        return dash.no_update, dash.no_update, ""

    @app.callback(
        Output("chatbot-conversation-container", "children", allow_duplicate=True),
        Input("app-store", "data"),
        prevent_initial_call=True
    )
    def restore_chatbot_history(app_store_data):
        """
        Restores chatbot conversation history on page load/refresh.
        """
        session_id = app_store_data.get("session_id")
        if not session_id:
            return []
        
        history = get_gemini_chatbot().get_history(session_id)
        conversation_elements = []
        for msg in history:
            if msg["role"] == "user":
                conversation_elements.append(
                    html.Div(
                        html.Div(
                            msg["text"],
                            style={
                                "backgroundColor": "#3B82F6",
                                "color": "white",
                                "padding": "10px 15px",
                                "borderRadius": "10px",
                                "marginLeft": "20%",
                                "wordWrap": "break-word",
                                "maxWidth": "80%",
                                "whiteSpace": "pre-wrap"
                            }
                        ),
                        className="mb-2"
                    )
                )
            else:
                conversation_elements.append(
                    html.Div(
                        html.Div(
                            msg["text"],
                            style={
                                "backgroundColor": "#6B7280",
                                "color": "white",
                                "padding": "10px 15px",
                                "borderRadius": "10px",
                                "marginRight": "20%",
                                "wordWrap": "break-word",
                                "maxWidth": "80%",
                                "whiteSpace": "pre-wrap"
                            }
                        ),
                        className="mb-2"
                    )
                )
        return conversation_elements