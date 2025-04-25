from typing import Dict, List, Optional, TypedDict, Any, Annotated
import operator
from typing import TypeVar, Dict, Any, Callable

# Define a type variable for dict values
T = TypeVar('T')

# Custom reducer function to merge dictionaries
def dict_merge(dict1: Dict[str, T], dict2: Dict[str, T]) -> Dict[str, T]:
    """Merge two dictionaries and return the combined dictionary."""
    result = dict1.copy()
    result.update(dict2)
    return result

class ResearchState(TypedDict, total=False):
    """State maintained throughout the research process."""
    
    # Input and analysis
    original_query: str  # This should not be modified by parallel nodes
    analyzed_query: Dict[str, Any]  # Intent, keywords, type of info needed
    
    # Search and planning
    search_queries: List[str]  # Generated search terms
    research_plan: Dict[str, Any]  # Strategy with steps
    
    # Results from tools - Use Annotated types with reducers for parallel node updates
    web_results: Annotated[List[Dict[str, Any]], operator.add]  # [{'url': str, 'snippet': str, 'title': str}]
    news_results: Annotated[List[Dict[str, Any]], operator.add]  # [{'url': str, 'title': str, 'summary': str, 'date': str}]
    urls_to_scrape: Annotated[List[str], operator.add]
    scraped_content: Annotated[Dict[str, str], dict_merge]  # {url: content}
    
    # Processed information
    analyzed_content: Annotated[Dict[str, Dict[str, Any]], dict_merge]  # {url: {'relevance': float, 'key_points': List[str]}}
    synthesized_information: Annotated[List[Dict[str, Any]], operator.add]  # Structured extracted data
    
    # Output
    final_report: str
    
    # Control and error handling
    error_log: Annotated[List[Dict[str, Any]], operator.add]
    iteration_count: Dict[str, int]  # {task_type: count} to prevent infinite loops
    max_iterations: Dict[str, int]  # Maximum iterations allowed per task type
    next_node: str  # The next node to route to in the graph 