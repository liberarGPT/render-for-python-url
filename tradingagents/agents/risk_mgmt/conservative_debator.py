from langchain_core.messages import AIMessage
import time
import json
from ..utils.agent_trading_modes import get_trading_mode_context, get_agent_specific_context

# Import prompt capture utility
try:
    from webui.utils.prompt_capture import capture_agent_prompt
except ImportError:
    # Fallback for when webui is not available
    def capture_agent_prompt(report_type, prompt_content, symbol=None):
        pass


def create_safe_debator(llm, config=None):
    def safe_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        safe_history = risk_debate_state.get("safe_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        macro_report = state["macro_report"]

        trader_decision = state["trader_investment_plan"]
        
        # Get trading mode from config
        current_position = state.get("current_position", "NEUTRAL")
        
        # Get centralized trading mode context
        trading_context = get_trading_mode_context(config, current_position)
        agent_context = get_agent_specific_context("risk_mgmt", trading_context)
        
        # Get mode-specific terms for the prompt
        actions = trading_context["actions"]
        mode_name = trading_context["mode_name"]
        decision_format = trading_context["decision_format"]

        # Use centralized trading mode context with conservative risk bias
        risk_specific_context = f"""
{agent_context}

**CONSERVATIVE EOD TRADING APPROACH:**
As the Conservative Risk Analyst for EOD trading, you prioritize capital preservation while capturing modest overnight gains:

**CONSERVATIVE EOD PRINCIPLES:**
- **Position Sizing:** Never risk more than 1.5% per EOD trade (vs. aggressive 3%)
- **Entry Timing:** Wait for clear daily technical confirmation before entering overnight positions
- **Stop Losses:** Tight stops at 2-3% maximum loss, preferably at daily technical levels
- **Target Profits:** Take profits early at first daily resistance rather than holding for maximum gains
- **Market Selection:** Favor liquid, established stocks over volatile small-caps for overnight holds
- **Risk/Reward:** Minimum 2.5:1 R/R ratio required, preferably 3:1

**CONSERVATIVE RISK ASSESSMENT:**
1. **Overnight Risk:** Minimize exposure to earnings/news during overnight holding periods
2. **Gap Risk:** Avoid overnight positions in stocks prone to large opening gaps
3. **Liquidity Risk:** Only EOD trade stocks with >1M average daily volume
4. **Market Environment:** Reduce overnight exposure during high VIX periods (>25)
5. **Position Limits:** Maximum 10% of portfolio in overnight positions total
6. **Time Limits:** Exit EOD trades at market close if targets not met by end of next day

**CONSERVATIVE EOD SIGNALS:**
- Require confluence of multiple daily technical indicators before overnight entry
- Prefer buying near daily support rather than chasing EOD breakouts
- Exit immediately if daily stop loss is hit, no second chances
- Take 50% profits at first daily resistance level to lock in gains
- Avoid EOD trading during earnings season or major Fed events

Focus on preserving capital first, generating returns second. Challenge aggressive proposals that exceed conservative risk limits."""

        prompt = f"""As the Safe/Conservative Risk Analyst, your primary objective is to protect assets, minimize volatility, and ensure steady, reliable growth. You prioritize stability, security, and risk mitigation, carefully assessing potential losses, economic downturns, and market volatility. {risk_specific_context}

When evaluating the trader's decision or plan, critically examine high-risk elements, pointing out where the decision may expose the firm to undue risk and where more cautious alternatives could secure long-term gains.

Here is the trader's decision:
{trader_decision}

Your task is to actively counter the arguments of the Risky and Neutral Analysts, advocating for conservative {actions} and highlighting where their views may overlook potential threats or fail to prioritize sustainability. Respond directly to their points, drawing from the following data sources to build a convincing case for a low-risk approach adjustment to the trader's decision:

Macro Economic Report: {macro_report}
Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}

Here is the current conversation history: {history} 
Here is the last response from the risky analyst: {current_risky_response} 
Here is the last response from the neutral analyst: {current_neutral_response}. 

If there are no responses from the other viewpoints, do not hallucinate and just present your point.

Engage by questioning their optimism and emphasizing the potential downsides they may have overlooked. Address each of their counterpoints to showcase why a conservative stance is ultimately the safest path for the firm's assets. Focus on debating and critiquing their arguments to demonstrate the strength of a low-risk strategy over their approaches. 

Always conclude with your recommendation using the format: {decision_format}

Output conversationally as if you are speaking without any special formatting."""

        # Capture the COMPLETE prompt that gets sent to the LLM
        ticker = state.get("company_of_interest", "")
        capture_agent_prompt("conservative_report", prompt, ticker)

        response = llm.invoke(prompt)

        argument = f"Safe Analyst: {response.content}"

        # Store safe messages as a list for proper conversation display
        safe_messages = risk_debate_state.get("safe_messages", [])
        safe_messages.append(argument)
        
        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "risky_messages": risk_debate_state.get("risky_messages", []),
            "safe_history": safe_history + "\n" + argument,
            "safe_messages": safe_messages,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "neutral_messages": risk_debate_state.get("neutral_messages", []),
            "latest_speaker": "Safe",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return safe_node
