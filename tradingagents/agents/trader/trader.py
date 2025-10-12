import functools
import time
import json
from ..utils.agent_trading_modes import get_trading_mode_context, get_agent_specific_context, extract_recommendation, format_final_decision
from tradingagents.dataflows.alpaca_utils import AlpacaUtils

# Import prompt capture utility
try:
    from webui.utils.prompt_capture import capture_agent_prompt
except ImportError:
    # Fallback for when webui is not available
    def capture_agent_prompt(report_type, prompt_content, symbol=None):
        pass


def create_trader(llm, memory, config=None):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        macro_report = state["macro_report"]
        
        # Determine current position from live Alpaca account (fallback to state)
        current_position = AlpacaUtils.get_current_position_state(company_name)
        # Persist into state so downstream agents see an accurate picture
        state["current_position"] = current_position

        # ---------------------------------------------------------
        # NEW: Pull richer live account & position metrics from Alpaca
        # ---------------------------------------------------------
        positions_data = AlpacaUtils.get_positions_data()
        account_info = AlpacaUtils.get_account_info()

        # Build a user-friendly summary for the specific symbol the agent cares about
        position_stats_desc = ""
        symbol_key = company_name.upper().replace("/", "")
        for pos in positions_data:
            if pos["Symbol"].upper() == symbol_key:
                qty = pos["Qty"]
                avg_entry = pos["Avg Entry"]
                today_pl_dollars = pos["Today's P/L ($)"]
                today_pl_percent = pos["Today's P/L (%)"]
                total_pl_dollars = pos["Total P/L ($)"]
                total_pl_percent = pos["Total P/L (%)"]

                position_stats_desc = (
                    f"Position Details for {company_name}:\n"
                    f"- Quantity: {qty}\n"
                    f"- Average Entry Price: {avg_entry}\n"
                    f"- Today's P/L: {today_pl_dollars} ({today_pl_percent})\n"
                    f"- Total P/L: {total_pl_dollars} ({total_pl_percent})"
                )
                break
        if not position_stats_desc:
            position_stats_desc = "No open position details available for this symbol."

        buying_power = account_info.get("buying_power", 0.0)
        cash = account_info.get("cash", 0.0)
        daily_change_dollars = account_info.get("daily_change_dollars", 0.0)
        daily_change_percent = account_info.get("daily_change_percent", 0.0)
        account_status_desc = (
            "Account Status:\n"
            f"- Buying Power: ${buying_power:,.2f}\n"
            f"- Cash: ${cash:,.2f}\n"
            f"- Daily Change: ${daily_change_dollars:,.2f} ({daily_change_percent:.2f}%)"
        )
        # ---------------------------------------------------------
        # END NEW BLOCK
        # ---------------------------------------------------------

        # Human-readable description for the prompt
        open_pos_desc = (
            f"We currently have an open {current_position} position in {company_name}."
            if current_position != "NEUTRAL"
            else f"We do not have any open position in {company_name}."
        )
        
        # Get centralized trading mode context
        trading_context = get_trading_mode_context(config, current_position)
        agent_context = get_agent_specific_context("trader", trading_context)
        
        # Get mode-specific terms for the prompt
        actions = trading_context["actions"]
        mode_name = trading_context["mode_name"]
        decision_format = trading_context["decision_format"]
        final_format = trading_context["final_format"]

        curr_situation = f"{macro_report}\n\n{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        # Use centralized trading mode context for trader-specific instructions
        trader_context = f"""
{agent_context}

**EOD TRADER DECISION MAKING:**
As the EOD Trader, you specialize in making trading decisions at market close for overnight positions. You focus on:

**EOD TRADING METHODOLOGY:**
- **Decision Timing:** Make all trading decisions during the final 10 minutes of market hours (3:50-4:00 PM ET)
- **Entry Strategy:** Based on daily closing prices, EOD momentum, and after-hours setup
- **Exit Strategy:** Daily reassessment at market close, gap management at next day's open
- **Risk Management:** Risk 1-3% per trade, target 3-9% returns (2:1 to 3:1 R/R)
- **Position Sizing:** Based on daily volatility (ATR) and overnight gap risk

**EOD TRADING DECISION CRITERIA:**
1. **Daily Technical Setup:** Daily closing patterns, EOD momentum, and key level breaks
2. **End-of-Day Volume:** Volume confirmation from full trading session completion
3. **Risk/Reward:** Minimum 2:1 risk-reward ratio based on daily price ranges
4. **Overnight Catalysts:** News events, earnings, or announcements affecting next day
5. **Daily Market Context:** End-of-day market sentiment and overnight positioning
6. **Gap Risk Assessment:** Potential for overnight gaps and pre-market volatility

**POSITION MANAGEMENT:**
- Enter positions at market close or prepare for next day's open
- Daily stop loss and target reassessment at market close
- Manage overnight news and pre-market risk exposure
- Never risk more than 3% on any single EOD trade

Current Alpaca Position Status:
{open_pos_desc}

{position_stats_desc}

Alpaca Account Status:
{account_status_desc}

Your {decision_format} should be based on:
- **Entry Point:** Specific price level for EOD entry or next day's open
- **Target Price:** Realistic profit target based on daily ranges and resistance levels
- **Stop Loss:** Maximum acceptable loss point (below daily support levels)
- **Position Size:** Calculated based on daily volatility, not account size
- **Time Horizon:** Expected overnight hold with daily reassessment

Always conclude with: {final_format}

**CRITICAL:** Focus on EOD trading setups, not intraday scalping or long-term investments.

**ANALYSIS REQUIREMENT:** Provide comprehensive EOD trading analysis including:
1. **Daily Technical Setup** - End-of-day patterns, daily chart analysis, key levels
2. **Entry Strategy** - Specific EOD entry points and overnight positioning
3. **Risk Management** - Stop loss placement and daily volatility-based sizing
4. **Profit Targets** - Realistic targets based on daily ranges and technical levels
5. **Overnight Risk** - Assessment of news risk and pre-market factors
6. **Daily Context** - How market close conditions affect overnight positioning"""

        # Enhanced content validation for investment plan
        plan_content = investment_plan if investment_plan else ""
        
        # Check if investment plan is substantial enough
        if len(plan_content.strip()) < 150 or ("FINAL TRANSACTION PROPOSAL:" in plan_content and len(plan_content.replace("FINAL TRANSACTION PROPOSAL:", "").strip()) < 100):
            # Generate enhanced analysis prompt when investment plan is insufficient
            enhanced_prompt = f"""As a EOD Trader specializing in overnight positions, provide a comprehensive trading plan for {company_name}.

**AVAILABLE ANALYSIS:**
Market Analysis: {market_research_report[:500] if market_research_report else 'Limited data available'}
Sentiment Analysis: {sentiment_report[:300] if sentiment_report else 'Limited data available'}
News Analysis: {news_report[:300] if news_report else 'Limited data available'}
Fundamentals: {fundamentals_report[:300] if fundamentals_report else 'Limited data available'}
Macro Analysis: {macro_report[:300] if macro_report else 'Limited data available'}

**REQUIRED EOD TRADING PLAN:**
Provide a detailed analysis covering:
1. **Technical Setup Analysis** - Current chart patterns, support/resistance levels
2. **EOD Trading Entry Strategy** - Specific entry points and confirmation signals  
3. **Risk Management Plan** - Stop loss levels, position sizing methodology
4. **Profit Target Strategy** - Realistic targets based on technical levels
5. **Time Horizon Assessment** - Expected overnight hold rationale
6. **Market Context Integration** - How macro/news/sentiment affects the setup

**TRADING DECISION TABLE:**
Include a markdown table with:
| Aspect | Details |
|--------|---------|
| Entry Price | $X.XX (specific level) |
| Stop Loss | $X.XX (risk management) |
| Target 1 | $X.XX (first resistance) |
| Target 2 | $X.XX (extended target) |
| Risk/Reward | X:1 ratio |
| Position Size | X shares (based on stop distance) |
| Time Frame | X-X days expected hold |

Focus on actionable EOD trading insights with specific price levels and risk management."""

            context = {
                "role": "user", 
                "content": enhanced_prompt
            }
        else:
            # Use original context with valid investment plan
            context = {
                "role": "user",
                "content": f"Based on comprehensive EOD trading analysis by specialist analysts, here is a EOD trading plan for {company_name}. This plan incorporates technical patterns, momentum indicators, support/resistance levels, and volume analysis optimized for overnight positions.\n\nProposed EOD Trading Plan: {investment_plan}\n\nMake your EOD trading decision focusing on clear entry points, profit targets, and stop losses with proper risk management for overnight positions.",
            }

        messages = [
            {
                "role": "system",
                "content": f"""You are a trading agent analyzing market data to make investment decisions. {trader_context}

Do not forget to utilize lessons from past decisions to learn from your mistakes. Here is some reflections from similar situations you traded in and the lessons learned: {past_memory_str}""",
            },
            context,
        ]

        # Capture the COMPLETE prompt that gets sent to the LLM
        try:
            # Combine system and user messages into complete prompt
            system_message = messages[0]["content"]
            user_message = context["content"]
            complete_prompt = f"""SYSTEM MESSAGE:
{system_message}

USER MESSAGE:
{user_message}"""
            
            capture_agent_prompt("trader_investment_plan", complete_prompt, company_name)
        except Exception as e:
            print(f"[TRADER] Warning: Could not capture complete prompt: {e}")
            # Fallback to system message only
            capture_agent_prompt("trader_investment_plan", messages[0]["content"], company_name)

        result = llm.invoke(messages)

        # Enhanced validation and final proposal handling
        analysis_content = result.content if hasattr(result, 'content') else str(result)
        
        # Check if we have substantial analysis content
        if len(analysis_content.strip()) < 200 or ("FINAL TRANSACTION PROPOSAL:" in analysis_content and len(analysis_content.replace("FINAL TRANSACTION PROPOSAL:", "").strip()) < 150):
            # Generate fallback comprehensive analysis
            fallback_prompt = f"""As an expert EOD trader, create a comprehensive trading plan for {company_name} focusing on overnight positions.

**EOD TRADING ANALYSIS:**
1. **Technical Setup** - Analyze current price action and key levels
2. **Entry Strategy** - Define specific entry points and signals
3. **Risk Management** - Calculate stop losses and position size
4. **Profit Targets** - Set realistic price objectives
5. **Trading Timeline** - Establish expected holding period

Include detailed reasoning for EOD trading decisions and conclude with a clear recommendation.

Focus on actionable insights with specific price levels and risk parameters."""
            
            fallback_result = llm.invoke(fallback_prompt)
            analysis_content = fallback_result.content if hasattr(fallback_result, 'content') else str(fallback_result)
        
        # Ensure we have a final recommendation
        if "FINAL TRANSACTION PROPOSAL:" not in analysis_content:
            # Create final recommendation based on analysis
            final_prompt = f"""Based on the following EOD trading analysis for {company_name}, provide your final trading decision.

Analysis:
{analysis_content}

Provide a brief justification and conclude with: FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**"""
            
            final_result = llm.invoke(final_prompt)
            final_content = final_result.content if hasattr(final_result, 'content') else str(final_result)
            
            # Properly combine analysis with final proposal
            combined_content = analysis_content + "\n\n---\n\n## Final Trading Decision\n\n" + final_content
            result = type(result)(content=combined_content)
        else:
            # Analysis already contains final proposal
            result = type(result)(content=analysis_content)

        # Extract the recommendation from the response
        trading_mode = trading_context["mode"]
        extracted_recommendation = extract_recommendation(result.content, trading_mode)
        
        # Format the final decision if extraction was successful
        final_decision_content = result.content
        if extracted_recommendation:
            final_decision_content = format_final_decision(extracted_recommendation, trading_mode)

        return {
            "messages": [result],
            "trader_investment_plan": final_decision_content,
            "sender": name,
            "trading_mode": trading_mode,
            "current_position": current_position,
            "recommended_action": extracted_recommendation,
        }

    return functools.partial(trader_node, name="Trader")
