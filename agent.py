"""Web Research Agent implementation with LangGraph."""

import logging
import os
from typing import Dict, Any, TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from state import ResearchState
import nodes
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_web_research_agent():
    """
    Create and configure the Web Research Agent graph.
    
    Returns:
        The configured agent graph.
    """
    graph = StateGraph(ResearchState)
    
    graph.add_node("analyze_query", nodes.analyze_query)
    graph.add_node("plan_research_strategy", nodes.plan_research_strategy)
    graph.add_node("execute_web_search", nodes.execute_web_search)
    graph.add_node("execute_news_search", nodes.execute_news_search)
    graph.add_node("evaluate_results_and_select_urls", nodes.evaluate_results_and_select_urls)
    graph.add_node("scrape_websites", nodes.scrape_websites)
    graph.add_node("extract_and_synthesize_information", nodes.extract_and_synthesize_information)
    graph.add_node("compile_final_report", nodes.compile_final_report)
    
    
    graph.set_entry_point("analyze_query")
    
    graph.add_edge("analyze_query", "plan_research_strategy")
    
    def route_to_search_strategy(state: ResearchState) -> str | Sequence[str]:
        news_search_disabled = os.environ.get("DISABLE_NEWS_SEARCH") == "true"
        
        search_approach = state.get("research_plan", {}).get("search_approach", "web_search")
        
        if news_search_disabled:
            if search_approach == "news_search":
                logger.info("News search is disabled, using web search instead")
                return "execute_web_search"
            elif search_approach == "parallel_search":
                logger.info("News search is disabled, using web search only (no parallel search)")
                return "execute_web_search"
        
        if search_approach == "parallel_search":
            return ["execute_web_search", "execute_news_search"]
        
        return search_approach  
    
    graph.add_conditional_edges(
        "plan_research_strategy",
        route_to_search_strategy,
        {
            "web_search": "execute_web_search", 
            "news_search": "execute_news_search",
            "extract_and_synthesize_information": "extract_and_synthesize_information",
            "execute_web_search": "execute_web_search",
            "execute_news_search": "execute_news_search"
        }
    )
    
 
    graph.add_edge(["execute_web_search", "execute_news_search"], "evaluate_results_and_select_urls")
    graph.add_edge("execute_web_search", "evaluate_results_and_select_urls")
    graph.add_edge("execute_news_search", "evaluate_results_and_select_urls")
    
    def route_after_evaluation(state: ResearchState) -> str:
        next_node = state.get("next_node", "extract_and_synthesize_information")
        
        if next_node == "plan_research_strategy":
            retry_count = state["iteration_count"]["total_research"]
            max_retries = state["max_iterations"]["total_research"]
            logger.info(f"Checking retry limits: current={retry_count}, max={max_retries}")
            if retry_count >= max_retries:
                logger.warning(f"Max retries ({max_retries}) exceeded, giving up and moving to synthesis")
                return "extract_and_synthesize_information"  
        
        return next_node
    
    graph.add_conditional_edges(
        "evaluate_results_and_select_urls",
        route_after_evaluation,
        {
            "scrape_websites": "scrape_websites",
            "extract_and_synthesize_information": "extract_and_synthesize_information",
            "plan_research_strategy": "plan_research_strategy"
        }
    )
    
    graph.add_edge("scrape_websites", "extract_and_synthesize_information")
    graph.add_edge("extract_and_synthesize_information", "compile_final_report")
    graph.add_edge("compile_final_report", END)
    return graph.compile()


def run_web_research_agent(query: str) -> Dict[str, Any]:
    """
    Run the Web Research Agent with a user query.
    
    Args:
        query: The user's research query
        
    Returns:
        The final state containing the research report
    """
    agent = create_web_research_agent()
    
    initial_state = {"original_query": query}
    
    logger.info(f"Starting research for query: {query}")
    
    try:
        result = agent.invoke(initial_state)
        logger.info("Research complete")
        return result
    except Exception as e:
        logger.error(f"Error running research agent: {str(e)}")
        return {
            "original_query": query,
            "final_report": f"Error conducting research: {str(e)}",
            "error_log": [{"type": "agent_error", "message": str(e)}]
        }
