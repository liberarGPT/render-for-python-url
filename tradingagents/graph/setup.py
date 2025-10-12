# TradingAgents/graph/setup.py

import concurrent.futures
import threading
import copy
import time
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.analysts.macro_analyst import create_macro_analyst
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.utils.agent_utils import Toolkit

from .conditional_logic import ConditionalLogic


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: ChatOpenAI,
        deep_thinking_llm: ChatOpenAI,
        toolkit: Toolkit,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        risk_manager_memory,
        conditional_logic: ConditionalLogic,
        config: Dict[str, Any] = None,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.toolkit = toolkit
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.risk_manager_memory = risk_manager_memory
        self.conditional_logic = conditional_logic
        self.config = config

    def _create_parallel_analysts_coordinator(self, selected_analysts, analyst_nodes, tool_nodes, delete_nodes):
        """Create a coordinator that runs selected analysts in parallel"""
        
        def parallel_analysts_execution(state: AgentState):
            """Execute selected analysts in parallel"""
            print(f"[PARALLEL] Starting parallel execution of analysts: {selected_analysts}")
            print(f"[PARALLEL] State keys available: {list(state.keys())}")
            
            # Check if UI state management is available
            ui_available = False
            try:
                from webui.utils.state import app_state
                ui_available = True
            except ImportError:
                pass
            
            # Update UI status for all analysts as in_progress
            if ui_available:
                for analyst_type in selected_analysts:
                    analyst_name = f"{analyst_type.capitalize()} Analyst"
                    app_state.update_agent_status(analyst_name, "in_progress")
            
            def execute_single_analyst(analyst_info):
                """Execute a single analyst in a separate thread"""
                analyst_type, analyst_node = analyst_info
                
                # Create a deep copy of the state for this analyst
                analyst_state = copy.deepcopy(state)
                
                print(f"[PARALLEL] Starting {analyst_type} analyst")
                
                # Execute the analyst
                try:
                    # Add a small delay before starting analyst execution
                    time.sleep(0.1)  # 100ms delay before starting
                    
                    result_state = analyst_node(analyst_state)
                    
                    # Check if the analyst made tool calls
                    has_tool_calls = False
                    if result_state.get("messages") and len(result_state["messages"]) > 0:
                        last_message = result_state["messages"][-1]
                        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                            has_tool_calls = True
                    
                    if has_tool_calls:
                        print(f"[PARALLEL] {analyst_type} analyst making tool calls")
                        tool_result = tool_nodes[analyst_type].invoke(result_state)
                        
                        if tool_result and tool_result.get("messages"):
                            # Preserve original state fields when merging tool results
                            merged_state = analyst_state.copy()  # Start with original state
                            
                            # Update with tool result messages
                            merged_state["messages"] = tool_result["messages"]
                            
                            # Add a small delay before making the next LLM call
                            time.sleep(0.2)  # 200ms delay between tool result and next analyst call
                            
                            # Run analyst again with tool results
                            result_state = analyst_node(merged_state)
                    else:
                        print(f"[PARALLEL] {analyst_type} analyst completed without tool calls")
                    
                    # Clean up messages safely
                    if result_state.get("messages"):
                        # Check if all messages have valid IDs before cleaning
                        valid_messages = [m for m in result_state["messages"] if m is not None and hasattr(m, 'id') and m.id is not None]
                        if valid_messages:
                            # Create a temporary state with only valid messages for cleanup
                            temp_state = {"messages": valid_messages}
                            final_state = delete_nodes[analyst_type](temp_state)
                            # Preserve other fields from result_state
                            for key, value in result_state.items():
                                if key != "messages":
                                    final_state[key] = value
                        else:
                            # No valid messages to clean, use result_state as is
                            final_state = result_state
                    else:
                        final_state = result_state
                    
                    print(f"[PARALLEL] {analyst_type} analyst completed")
                    
                    # Update UI status to completed
                    if ui_available:
                        analyst_name = f"{analyst_type.capitalize()} Analyst"
                        app_state.update_agent_status(analyst_name, "completed")
                    
                    return analyst_type, final_state
                    
                except Exception as e:
                    print(f"[PARALLEL] Error in {analyst_type} analyst: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Update UI status to error (completed with issues)
                    if ui_available:
                        analyst_name = f"{analyst_type.capitalize()} Analyst"
                        app_state.update_agent_status(analyst_name, "completed")
                    
                    return analyst_type, analyst_state
            
            # Execute all analysts in parallel with staggered starts
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_analysts)) as executor:
                # Submit analyst tasks with delays to avoid API rate limits
                future_to_analyst = {}
                for i, analyst_type in enumerate(selected_analysts):
                    # Add a small delay between each analyst start (0.5 seconds apart)
                    if i > 0:
                        time.sleep(0.5)
                    
                    analyst_node = analyst_nodes[analyst_type]
                    future = executor.submit(execute_single_analyst, (analyst_type, analyst_node))
                    future_to_analyst[future] = analyst_type
                    print(f"[PARALLEL] Submitted {analyst_type} analyst (delay: {i * 0.5}s)")
                
                # Collect results as they complete
                completed_results = {}
                for future in concurrent.futures.as_completed(future_to_analyst):
                    analyst_type = future_to_analyst[future]
                    try:
                        result_analyst_type, result_state = future.result()
                        completed_results[result_analyst_type] = result_state
                        print(f"[PARALLEL] {result_analyst_type} analyst completed successfully")
                    except Exception as e:
                        print(f"[PARALLEL] {analyst_type} analyst failed: {e}")
                        completed_results[analyst_type] = state  # Use original state as fallback
            
            print(f"[PARALLEL] All analysts completed. Merging results...")
            
            # Merge all results into the final state
            final_state = copy.deepcopy(state)
            
            # Collect all analyst reports
            for analyst_type, result_state in completed_results.items():
                if result_state["messages"]:
                    # Extract the analyst's report from their final message
                    final_message = result_state["messages"][-1]
                    if hasattr(final_message, 'content') and final_message.content:
                        # Store the report in the appropriate field
                        report_field = f"{analyst_type}_report"
                        if analyst_type == "social":
                            report_field = "sentiment_report"
                        
                        # Add to state
                        final_state[report_field] = final_message.content
                        print(f"[PARALLEL] Stored {analyst_type} report")
                        
                        # Update report in UI state as well
                        if ui_available:
                            ticker = state.get("ticker", "")
                            if ticker:
                                ui_state = app_state.get_state(ticker)
                                if ui_state:
                                    ui_state["current_reports"][report_field] = final_message.content
            
            print(f"[PARALLEL] Parallel analyst execution completed")
            return final_state
        
        return parallel_analysts_execution

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals", "macro"]
    ):
        """Set up and compile the agent workflow graph with configurable parallel/sequential analyst execution.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst  
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")
        
        # Check if parallel execution is enabled
        parallel_mode = self.config.get("parallel_analysts", True)
        print(f"[SETUP] Using {'parallel' if parallel_mode else 'sequential'} analyst execution mode")

        # Create analyst nodes
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = create_market_analyst(
                self.quick_thinking_llm, self.toolkit
            )
            delete_nodes["market"] = create_msg_delete()
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            analyst_nodes["social"] = create_social_media_analyst(
                self.quick_thinking_llm, self.toolkit
            )
            delete_nodes["social"] = create_msg_delete()
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            analyst_nodes["news"] = create_news_analyst(
                self.quick_thinking_llm, self.toolkit
            )
            delete_nodes["news"] = create_msg_delete()
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = create_fundamentals_analyst(
                self.quick_thinking_llm, self.toolkit
            )
            delete_nodes["fundamentals"] = create_msg_delete()
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        if "macro" in selected_analysts:
            analyst_nodes["macro"] = create_macro_analyst(
                self.quick_thinking_llm, self.toolkit
            )
            delete_nodes["macro"] = create_msg_delete()
            tool_nodes["macro"] = self.tool_nodes["macro"]

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(
            self.quick_thinking_llm, self.bull_memory
        )
        bear_researcher_node = create_bear_researcher(
            self.quick_thinking_llm, self.bear_memory
        )
        research_manager_node = create_research_manager(
            self.deep_thinking_llm, self.invest_judge_memory
        )
        trader_node = create_trader(self.deep_thinking_llm, self.trader_memory, self.config)

        # Create risk analysis nodes
        risky_analyst = create_risky_debator(self.quick_thinking_llm, self.config)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm, self.config)
        safe_analyst = create_safe_debator(self.quick_thinking_llm, self.config)
        risk_manager_node = create_risk_manager(
            self.deep_thinking_llm, self.risk_manager_memory, self.config
        )

        # Create workflow
        workflow = StateGraph(AgentState)

        if parallel_mode:
            # Create parallel analysts coordinator
            parallel_analysts_node = self._create_parallel_analysts_coordinator(
                selected_analysts, analyst_nodes, tool_nodes, delete_nodes
            )
            
            # Add the parallel analysts node
            workflow.add_node("Parallel Analysts", parallel_analysts_node)
            
            # Define edges for parallel execution
            # Start with parallel analysts execution
            workflow.add_edge(START, "Parallel Analysts")
            
            # After parallel analysts complete, proceed to Bull Researcher
            workflow.add_edge("Parallel Analysts", "Bull Researcher")
        else:
            # Add individual analyst nodes for sequential execution
            for analyst_type, node in analyst_nodes.items():
                workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
                workflow.add_node(
                    f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type]
                )
                workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])
            
            # Define edges for sequential execution
            # Start with the first analyst
            first_analyst = selected_analysts[0]
            workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

            # Connect analysts in sequence
            for i, analyst_type in enumerate(selected_analysts):
                current_analyst = f"{analyst_type.capitalize()} Analyst"
                current_tools = f"tools_{analyst_type}"
                current_clear = f"Msg Clear {analyst_type.capitalize()}"

                # Add conditional edges for current analyst
                workflow.add_conditional_edges(
                    current_analyst,
                    getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                    [current_tools, current_clear],
                )
                workflow.add_edge(current_tools, current_analyst)

                # Connect to next analyst or to Bull Researcher if this is the last analyst
                if i < len(selected_analysts) - 1:
                    next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                    workflow.add_edge(current_clear, next_analyst)
                else:
                    workflow.add_edge(current_clear, "Bull Researcher")

        # Add other nodes (common to both modes)
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Risky Analyst", risky_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Safe Analyst", safe_analyst)
        workflow.add_node("Risk Judge", risk_manager_node)

        # Add remaining edges (unchanged from original)
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Risky Analyst")
        workflow.add_conditional_edges(
            "Risky Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Safe Analyst": "Safe Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Safe Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Risky Analyst": "Risky Analyst",
                "Risk Judge": "Risk Judge",
            },
        )
        workflow.add_edge("Risk Judge", END)

        return workflow.compile()
