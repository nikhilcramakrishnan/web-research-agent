import re
from typing import Dict, List, Tuple, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from state import ResearchState
from tools import TavilySearchTool, NewsAggregatorTool
import config
import os
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
import logging
import json
import asyncio

logger = logging.getLogger(__name__)

web_search_tool = TavilySearchTool()
news_tool = NewsAggregatorTool()

google_api_key = os.environ.get("GOOGLE_API_KEY") or config.GOOGLE_API_KEY
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is required for the research agent")

llm = ChatGoogleGenerativeAI(
    model=config.LLM_MODEL,
    temperature=config.LLM_TEMPERATURE,
    google_api_key=google_api_key,
    convert_system_message_to_human=True  
)

def analyze_query(state: ResearchState) -> ResearchState:
    """Analyze user query to extract research requirements.

    Args:
        state (ResearchState): current state with 'original_query'.

    Returns:
        ResearchState: updated state with 'analyzed_query', 'search_queries',
            'iteration_count', 'max_iterations', and empty 'error_log'.
    """
    logger.info(f"Analyzing query: {state['original_query']}")
    
    prompt = ChatPromptTemplate.from_template("""
        You are a meticulous research assistant AI. Your task is to thoroughly analyze user queries to understand the core intent, necessary information, and optimal approach for research.

        Analyze the following research query:

        QUERY: {query}

        Break down the query by determining the following components:

        1.  **main_topic**: The primary subject or domain of the query.
        2.  **specific_request**: The precise question being asked or the specific information the user seeks within the main topic.
        3.  **info_type**: The nature of the information required (e.g., factual definitions, technical explanations, user opinions/reviews, news updates, comparative analysis, historical context, pros/cons).
        4.  **time_sensitive**: Is the query about current, rapidly changing information (True) or established knowledge (False)?
        5.  **key_entities**: List the crucial nouns, concepts, products, organizations, or people central to the query.
        6.  **subjective_criteria**: Identify any terms requiring judgment or assessment (e.g., "best", "reliable", "struggle", "effective", "minimal", "complex").
        7.  **depth_required**: Assess the level of detail needed:
            * "low": Simple, discrete facts (e.g., dates, definitions).
            * "medium": Requires some context, explanation, or summarization found in detailed articles or sections.
            * "high": Needs comprehensive analysis, synthesis of multiple sources, understanding complex mechanisms, or detailed comparisons (common for subjective or "why/how" questions).
        8.  **regional_context**: Note any specified geographical area, language, or domain focus (e.g., "in Canada", "for Python developers"). Use null if not applicable.
        9.  **search_queries**: Generate a list of **{max_search_queries}** distinct and effective search engine queries. These queries should explore different aspects and angles of the original query to find comprehensive information. Each query should be a phrase or question you would realistically use in Google. Ensure the queries combine the `specific_request`, `key_entities`, and `subjective_criteria` in meaningful ways.
         **Absolutely avoid:**
                 * Simply repeating the exact original query word-for-word.
                 * Overly short, vague, or non-descriptive queries (e.g., "specific generative", "advancements").
                 * Queries that are trivial variations of each other (e.g., just adding a single word like "latest" or a question mark).
                 * Queries that are just single words or isolated entities.
                 * Queries that are too similar to each other.
                                                
         Include variations, synonyms, and technical terms where appropriate.
         Example for query "Recent advancements in edible robotics":
         ["latest research edible robotics", "developments in food safe robots", "applications of ingestible robotics in medicine", "challenges designing edible robots", "companies developing edible robot technology", "edible robotics current state review", "future of edible robots"]
         10. **requires_web_scraping**: Set to true if the query involves subjective assessment, requires comparisons, needs in-depth analysis/synthesis from potentially long-form content, or tackles complex topics where simple search snippets are insufficient. Set to false for straightforward factual queries.

        Respond ONLY with a valid JSON object adhering to this structure:
        {{
            "main_topic": "topic",
            "specific_request": "what is being asked",
            "info_type": "facts/opinions/news/analysis/comparison/etc",
            "time_sensitive": true/false,
            "key_entities": ["entity1", "entity2", ...],
            "subjective_criteria": ["criterion1", "criterion2", ...],
            "depth_required": "low/medium/high",
            "regional_context": "region or null",
            "search_queries": ["search query 1", "search query 2", ...],
            "requires_web_scraping": true/false
        }}
        """)
    
    response = llm.invoke(prompt.format(query=state['original_query'], max_search_queries=config.MAX_SEARCH_QUERIES))

    try:
        response_text = response.content
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].strip()
        else:
            json_text = response_text.strip()
            
        analyzed_query = json.loads(json_text)
        logger.info(f"Query analysis complete: {analyzed_query}")
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        analyzed_query = {
            "main_topic": state['original_query'],
            "specific_request": state['original_query'],
            "info_type": "facts",
            "time_sensitive": False,
            "key_entities": [state['original_query']],
            "subjective_criteria": [],
            "depth_required": "medium",
            "regional_context": None,
            "search_queries": [state['original_query']],
            "requires_web_scraping": False
        }
    
    return {
        **state,
        "analyzed_query": analyzed_query,
        "search_queries": analyzed_query.get("search_queries", [state['original_query']]),
        "iteration_count": {"search_refinement": 0, "total_research": 0},
        "max_iterations": config.MAX_ITERATIONS,
        "error_log": []
    }


def plan_research_strategy(state: ResearchState) -> ResearchState:
    """Plan research steps based on the analyzed query.

    Args:
        state (ResearchState): current state with 'analyzed_query'.

    Returns:
        ResearchState: state updated with 'research_plan' and incremented 'iteration_count'.
    """
    logger.info("Planning research strategy")
    
    state["iteration_count"]["total_research"] = state["iteration_count"].get("total_research", 0) + 1
    
    if state["iteration_count"]["total_research"] > state["max_iterations"]["total_research"]:
        logger.warning("Exceeded maximum total research iterations")
        return {
            **state,
            "research_plan": {"search_approach": "extract_and_synthesize_information"}
        }
    
    prompt = ChatPromptTemplate.from_template("""
    You are a research strategist. Based on the analyzed query, create a research plan.
    
    QUERY ANALYSIS: {analyzed_query}
    
    Please create a step-by-step plan for researching this topic:
    
    1. What type of search should be performed first (web search or news search or both in parallel)?
    2. What specific information should we look for in the search results?
    3. Should we prioritize certain types of sources?
    
    Respond in a structured JSON format:
    {{
        "search_approach": "web_search" or "news_search" or "parallel_search",
        "priority_info": ["info1", "info2", ...],
        "source_priorities": ["academic", "news", "general", etc],
        "search_refinement_needed": true/false
    }}
    """)
    
    response = llm.invoke(prompt.format(analyzed_query=state['analyzed_query']))
    
    try:
        response_text = response.content
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].strip()
        else:
            json_text = response_text.strip()
            
        research_plan = json.loads(json_text)
        logger.info(f"Research plan created: {research_plan}")
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        research_plan = {"search_approach": "web_search", "priority_info": [], "source_priorities": []}
    
    return {
        **state,
        "research_plan": research_plan
    }


def execute_web_search(state: ResearchState) -> Dict[str, Any]:
    """Perform web searches for the given queries.

    Args:
        state (ResearchState): current state with 'search_queries'.

    Returns:
        Dict[str, Any]: state fragment containing 'web_results'.
    """
    logger.info(f"Executing web search (limited to {config.MAX_SEARCH_QUERIES} API calls)")
    
    search_queries = state.get("search_queries", [state.get("original_query", "")])
    if not search_queries:
        search_queries = [state.get("original_query", "")]
    
    search_queries = search_queries[:config.MAX_SEARCH_QUERIES]
    
    all_results = []
    for i, query in enumerate(search_queries):
        logger.info(f"Executing Tavily search {i+1}/{len(search_queries)}: {query}")
        results = web_search_tool.search(query)
        all_results.extend(results)
    
    logger.info(f"Retrieved a total of {len(all_results)} search results from {len(search_queries)} queries")
    return {
        "web_results": all_results
    }


def execute_news_search(state: ResearchState) -> Dict[str, Any]:
    """Perform news searches for the given queries.

    Args:
        state (ResearchState): current state with 'search_queries' and time sensitivity info.

    Returns:
        Dict[str, Any]: state fragment containing 'news_results'.
    """
    logger.info("Executing news search")
    
    time_sensitive = state.get("analyzed_query", {}).get("time_sensitive", False)
    days_back = 3 if time_sensitive else 30
    
    search_queries = state.get("search_queries", [state.get("original_query", "")])
    if not search_queries:
        search_queries = [state.get("original_query", "")]
    
    all_results = []
    for query in search_queries:
        results = news_tool.search_news(query, days_back=days_back)
        all_results.extend(results)
    
    return {
        "news_results": all_results
    }


def evaluate_results_and_select_urls(state: ResearchState) -> ResearchState:
    """Assess search snippets and determine if further scraping is needed.

    Args:
        state (ResearchState): state containing 'web_results' and 'news_results'.

    Returns:
        ResearchState: state updated with 'urls_to_scrape' and 'next_node'.
    """
    logger.info("Evaluating search results and selecting URLs")
    
    web_results = state.get("web_results", [])
    news_results = state.get("news_results", [])
    
    web_snippets = [
        {"source": "web", "title": r.get("title", ""), "snippet": r.get("snippet", ""), "url": r.get("url", "")}
        for r in web_results
    ]
    
    news_snippets = [
        {"source": "news", "title": r.get("title", ""), "snippet": r.get("summary", ""), 
         "url": r.get("url", ""), "date": r.get("date", ""), "publisher": r.get("source", "")}
        for r in news_results
    ]
    
    all_snippets = web_snippets + news_snippets
    
    if not all_snippets:
        logger.warning("No search results to evaluate, refinement needed")
        return {
            **state,
            "error_log": state.get("error_log", []) + [{"type": "search_error", "message": "No search results found"}],
            "next_node": "plan_research_strategy",
            "urls_to_scrape": []
        }
    
    original_query = state.get("original_query", "")
    analyzed_query = state.get("analyzed_query", {})
    
    query_has_subjective_elements = any(term in original_query.lower() for term in 
        ["best", "good", "great", "better", "worst", "minimal", "excellent", "quality", 
         "experience", "reliable", "recommended", "should", "worth", "comparison"])
    
    info_type = analyzed_query.get("info_type", "")
    requires_detailed_analysis = "opinion" in info_type.lower() or "comparison" in info_type.lower() or "analysis" in info_type.lower()
    
    prompt_template = """
    You are a research analyst evaluating search results to determine which are most relevant to a query.
    
    ORIGINAL QUERY: {original_query}
    ANALYZED QUERY: {analyzed_query}
    RESEARCH PLAN: {research_plan}
    
    Here are the search result snippets:
    {snippets}
    
    EVALUATION GUIDELINES:
    1. Identify ALL information requirements from the query.
    2. Assess whether snippets provide DETAILED information for each requirement.
    3. For subjective assessments (e.g., "best", "good quality", "worth it") or comparisons, 
       snippets rarely contain enough analysis - full content is typically needed.
    4. For specific facts, snippets may be sufficient.
    5. Be SKEPTICAL about snippet sufficiency - if in doubt, recommend scraping deeper content.
    
    Please evaluate these results and determine:
    1. Are the snippets alone sufficient to answer ALL parts of the query comprehensively? If yes, no web scraping is needed.
    2. If not, which specific URLs should be scraped for more in-depth information? (maximum {max_urls})
    3. If these results aren't helpful at all, should we refine our search strategy?
    
    Make sure to consider:
    - If the query asks for opinions, reasons, or comparisons, deeper content is usually needed
    - If the query requires recent or time-sensitive information, check if snippets have sufficient date context
    - If the query requires specific details that may only be available in full content
    
    Respond in the following JSON format:
    {{
        "requirements": [
            {{"aspect": "requirement1", "sufficient": true/false, "reason": "explanation"}}
        ],
        "snippets_sufficient": true/false,  // Should be false if ANY requirement is insufficient
        "urls_to_scrape": ["url1", "url2", ...] if snippets_sufficient is false, else [],
        "refine_search": true/false,
        "reasoning": "Brief explanation of your evaluation"
    }}
    """
    
    if query_has_subjective_elements or requires_detailed_analysis:
        prompt_template += """
        
        IMPORTANT NOTE: This query contains subjective components or requires detailed analysis.
        For such queries, search snippets are rarely sufficient to provide a comprehensive answer.
        Full content typically provides the justifications, comparisons, and context needed for 
        proper analysis. You should have a very high bar for declaring snippets "sufficient" in this case.
        """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    response = llm.invoke(
        prompt.format(
            original_query=original_query,
            analyzed_query=analyzed_query,
            research_plan=state.get("research_plan", {}),
            snippets=json.dumps(all_snippets, indent=2),
            max_urls=config.MAX_URLS_TO_SCRAPE
        )
    )
    
    try:
        response_text = response.content
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].strip()
        else:
            json_text = response_text.strip()
            
        evaluation = json.loads(json_text)
        logger.info(f"Evaluation complete: {evaluation}")
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        evaluation = {"snippets_sufficient": False, "urls_to_scrape": [], "refine_search": True}
    
    urls_to_scrape = evaluation.get("urls_to_scrape", [])
    snippets_sufficient = evaluation.get("snippets_sufficient", False)
    refine_search = evaluation.get("refine_search", False)
    
    if (query_has_subjective_elements or requires_detailed_analysis) and snippets_sufficient and not urls_to_scrape:
        substantive_snippets = [s for s in all_snippets if len(s.get("snippet", "")) > 100]
        if len(substantive_snippets) < 3: 
            logger.warning("Query has subjective elements but LLM marked snippets as sufficient without substantial evidence")
            snippets_sufficient = False
            urls_to_scrape = [s["url"] for s in all_snippets[:min(3, len(all_snippets))]]
    
    next_node = "extract_and_synthesize_information"
    
    if refine_search:
        logger.info("Search refinement needed")
        next_node = "plan_research_strategy"
    elif snippets_sufficient:
        logger.info("Snippets are sufficient, proceeding to synthesis")
        next_node = "extract_and_synthesize_information"
    elif urls_to_scrape:
        logger.info(f"Selected {len(urls_to_scrape)} URLs for scraping")
        next_node = "scrape_websites"
    else:
        if not snippets_sufficient:
            logger.warning("Snippets insufficient but no URLs selected, selecting some automatically")
            urls_to_scrape = [s["url"] for s in all_snippets[:min(3, len(all_snippets))]]
            if urls_to_scrape:
                next_node = "scrape_websites"
            else:
                logger.warning("No URLs available, defaulting to synthesis")
                next_node = "extract_and_synthesize_information"
        else:
            logger.warning("No clear evaluation direction, defaulting to synthesis")
            next_node = "extract_and_synthesize_information"
    
    return {
        **state,
        "urls_to_scrape": urls_to_scrape[:config.MAX_URLS_TO_SCRAPE],
        "next_node": next_node
    }


def scrape_websites(state: ResearchState) -> ResearchState:
    """Crawl and extract content from selected URLs.

    Args:
        state (ResearchState): state containing 'urls_to_scrape'.

    Returns:
        ResearchState: state updated with 'scraped_content' and 'error_log'.
    """
    logger.info("Scraping websites with Crawl4AI")
    
    import asyncio

    original_query = state.get("original_query", "")
    strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(provider='gemini/gemini-1.5-flash', api_token=google_api_key),
        instruction=f"Extract key insights relevant to the query: {original_query}",
        extraction_type="block"
    )
    urls_to_scrape = state.get("urls_to_scrape", [])
    scraped_content = state.get("scraped_content", {})
    async def scrape_urls():
        async with AsyncWebCrawler() as crawler:
            for url in urls_to_scrape:
                try:
                    logger.info(f"Crawl4AI scraping URL: {url}")
                    result = await crawler.arun(
                        url=url,
                        config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS),
                        extraction_strategy=strategy
                    )
                    if getattr(result, 'extracted_content', None):
                        scraped_content[url] = result.extracted_content
                        logger.info(f"Stored extracted_content for {url}")
                    elif result.markdown:
                        scraped_content[url] = result.markdown
                        logger.info(f"Successfully scraped {url} with Crawl4AI")
                    else:
                        logger.warning(f"No content retrieved from {url}")
                        error = {"type": "scrape_error", "url": url, "message": "Failed to extract content"}
                        state["error_log"] = state.get("error_log", []) + [error]
                except Exception as e:
                    logger.error(f"Failed to scrape {url} with Crawl4AI: {str(e)}")
                    error = {"type": "scrape_error", "url": url, "message": str(e)}
                    state["error_log"] = state.get("error_log", []) + [error]
    
    try:
        asyncio.run(scrape_urls())
    except Exception as e:
        logger.error(f"AsyncWebCrawler failed: {e}")
        state["error_log"] = state.get("error_log", []) + [{"type": "scrape_error", "message": str(e)}]

    return {
        **state,
        "scraped_content": scraped_content
    }


def extract_and_synthesize_information(state: ResearchState) -> Dict[str, Any]:
    """Extract data points from sources and combine into a coherent answer.

    Args:
        state (ResearchState): state with 'web_results', 'news_results', and 'scraped_content'.

    Returns:
        Dict[str, Any]: state fragment containing 'synthesized_information'.
    """
    logger.info("Extracting and synthesizing information")
    
    original_query = state.get("original_query", "")
    analyzed_query = state.get("analyzed_query", {})
    
    depth_required = analyzed_query.get("depth_required", "medium")
    subjective_criteria = analyzed_query.get("subjective_criteria", [])
    requires_web_scraping = analyzed_query.get("requires_web_scraping", False)
    
    is_fallback = state["iteration_count"].get("total_research", 0) >= state["max_iterations"].get("total_research", 0)
    
    if is_fallback:
        logger.warning("Synthesizing with limited information after exhausting search iterations")
    
    web_results = state.get("web_results", [])
    news_results = state.get("news_results", [])
    scraped_content = state.get("scraped_content", {})

    has_scraped_content = bool(scraped_content)
    has_substantial_info = has_scraped_content or len(web_results) >= 5 or len(news_results) >= 5
    
    if (depth_required == "high" or subjective_criteria) and not has_scraped_content and not is_fallback:
        logger.warning("Query requires high depth or has subjective criteria, but no scraped content is available")
    
    if not web_results and not news_results and not scraped_content:
        logger.warning("No information available to synthesize")
        return {
            "synthesized_information": [{
                "topic": original_query,
                "key_findings": ["No relevant information found after exhaustive search."],
                "confidence": "low",
                "sources": []
            }]
        }
    
    context = []
    
    if web_results:
        context.append("WEB SEARCH RESULTS:")
        for i, result in enumerate(web_results):
            context.append(f"[Web {i+1}] Title: {result.get('title', 'No title')}")
            context.append(f"    URL: {result.get('url', 'No URL')}")
            context.append(f"    Snippet: {result.get('snippet', 'No snippet')}")
            context.append("")
    
    if news_results:
        context.append("NEWS SEARCH RESULTS:")
        for i, result in enumerate(news_results):
            context.append(f"[News {i+1}] Title: {result.get('title', 'No title')}")
            context.append(f"    URL: {result.get('url', 'No URL')}")
            context.append(f"    Date: {result.get('date', 'No date')}")
            context.append(f"    Source: {result.get('source', 'Unknown source')}")
            context.append(f"    Summary: {result.get('summary', 'No summary')}")
            context.append("")
    
    if scraped_content:
        context.append("SCRAPED WEB CONTENT:")
        for i, (url, content) in enumerate(scraped_content.items()):
            truncated_content = content[:10000] + "..." if len(content) > 10000 else content
            context.append(f"[Scraped {i+1}] URL: {url}")
            context.append(f"Content: {truncated_content}")
            context.append("")
    
    limitations = []
    if requires_web_scraping and not has_scraped_content:
        limitations.append("WARNING: This query required in-depth content analysis, but no full web content was available.")
    
    if subjective_criteria and not has_scraped_content:
        criteria_list = ", ".join(subjective_criteria)
        limitations.append(f"WARNING: This query involves subjective assessment of [{criteria_list}], which typically requires detailed content analysis.")
    
    if depth_required == "high" and not has_substantial_info:
        limitations.append("WARNING: This query requires high-depth information, but the available data may be insufficient.")
    
    limitations_text = "\n".join(limitations)
    
    base_prompt = """
    You are a research analyst synthesizing information for a query.
    
    QUERY: {original_query}
    
    ANALYZED QUERY: {analyzed_query}
    
    INFORMATION CONTEXT:
    {context}
    
    {limitations_text}
    
    {fallback_notice}
    
    Based on the information provided, please synthesize the key findings and insights.
    Include only facts that are directly supported by the sources provided.
    
    """
    
    if subjective_criteria:
        subjective_prompt = f"""
        This query involves subjective assessments of: {', '.join(subjective_criteria)}
        
        When addressing these subjective elements:
        1. Clearly indicate confidence levels for each assessment
        2. Note when different sources have conflicting opinions
        3. Explain the criteria used for making subjective judgments
        4. Acknowledge limitations in making definitive statements
        """
        base_prompt += subjective_prompt
    
    base_prompt += """
    Respond in a structured JSON format:
    {{
        "key_topics": [
            {{
                "topic": "Main topic/subtopic name",
                "key_findings": [
                    "Finding 1 with source reference [Web 3]",
                    "Finding 2 with source reference [News 1]",
                    ...
                ],
                "confidence": "high/medium/low",
                "supporting_evidence": "Brief explanation of the evidence"
            }},
            ...
        ],
        "information_gaps": [
            "Description of any significant missing information",
            ...
        ],
        "source_assessment": "Assessment of the quality/reliability of sources",
        "recommendations": [
            "Recommendation 1",
            ...
        ],
        "confidence_summary": {{
            "overall": "high/medium/low",
            "reasoning": "Explanation for the confidence level"
        }}
    }}
    """
    
    prompt = ChatPromptTemplate.from_template(base_prompt)
    
    fallback_notice = ""
    if is_fallback:
        fallback_notice = "NOTE: This synthesis is being performed with limited information after exhausting search attempts. The results may be incomplete or less reliable."

    response = llm.invoke(
        prompt.format(
            original_query=original_query,
            analyzed_query=analyzed_query,
            context="\n".join(context),
            fallback_notice=fallback_notice,
            limitations_text=limitations_text
        )
    )
    
    try:
        response_text = response.content
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].strip()
        else:
            json_text = response_text.strip()
        
        json_text = re.sub(r'\s+', ' ', json_text)
        
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)
        
        try:
            synthesized_info = json.loads(json_text)
        except json.JSONDecodeError as json_error:
            logger.warning(f"Initial JSON parsing failed: {json_error}, attempting more aggressive cleanup")
            
            if json_text.startswith('{') and '}' in json_text:
                clean_json = json_text[json_text.find('{'):json_text.rfind('}')+1]
                synthesized_info = json.loads(clean_json)
            else:
                raise json.JSONDecodeError("Cannot parse JSON", json_text, 0)
        
        logger.info("Information synthesis complete")
        
        if "key_topics" not in synthesized_info:
            synthesized_info["key_topics"] = [{
                "topic": original_query,
                "key_findings": ["Information extracted but not properly formatted."],
                "confidence": "low",
                "supporting_evidence": "Data structure error in synthesis process."
            }]
        
        if "information_gaps" not in synthesized_info:
            synthesized_info["information_gaps"] = ["Information gaps not identified due to formatting issues."]
        
        if "source_assessment" not in synthesized_info:
            synthesized_info["source_assessment"] = "Source assessment unavailable due to data formatting issues."
        
        if "recommendations" not in synthesized_info:
            synthesized_info["recommendations"] = []
        
        if "confidence_summary" not in synthesized_info:
            synthesized_info["confidence_summary"] = {
                "overall": "low",
                "reasoning": "Confidence assessment unavailable due to data formatting issues."
            }
        elif not isinstance(synthesized_info["confidence_summary"], dict):
            synthesized_info["confidence_summary"] = {
                "overall": "low",
                "reasoning": f"Original confidence data was malformed: {synthesized_info['confidence_summary']}"
            }
    
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Raw response text: {response_text[:500]}...")
        synthesized_info = {
            "key_topics": [{
                "topic": original_query,
                "key_findings": ["Error synthesizing information."],
                "confidence": "low",
                "supporting_evidence": "Processing error occurred during synthesis."
            }],
            "information_gaps": ["Complete synthesis unavailable due to processing error."],
            "source_assessment": "Unable to assess sources due to processing error.",
            "recommendations": [],
            "confidence_summary": {
                "overall": "low",
                "reasoning": f"Error during information processing: {str(e)}"
            }
        }
    
    return {
        "synthesized_information": [synthesized_info]
    }


def compile_final_report(state: ResearchState) -> Dict[str, Any]:
    """Generate a formatted Markdown report from synthesized research data.

    Args:
        state (ResearchState): state containing 'synthesized_information'.

    Returns:
        Dict[str, Any]: state fragment containing 'final_report'.
    """
    logger.info("Compiling final report")
    
    original_query = state.get("original_query", "")
    analyzed_query = state.get("analyzed_query", {})
    
    depth_required = analyzed_query.get("depth_required", "medium")
    has_subjective_criteria = bool(analyzed_query.get("subjective_criteria", []))
    requires_web_scraping = analyzed_query.get("requires_web_scraping", False)
    
    synthesized_info = state.get("synthesized_information", [{}])[0]
    
    reached_max_retries = state["iteration_count"].get("total_research", 0) >= state["max_iterations"].get("total_research", 0)
    
    scraped_content = state.get("scraped_content", {})
    has_scraped_content = bool(scraped_content)
    
    confidence_summary = synthesized_info.get("confidence_summary", {"overall": "medium", "reasoning": ""})
    
    low_confidence_flags = []
    if reached_max_retries:
        low_confidence_flags.append("Research was limited by maximum iteration constraints")
    
    if (depth_required == "high" or has_subjective_criteria or requires_web_scraping) and not has_scraped_content:
        low_confidence_flags.append("Research depth may be insufficient for this type of query")
    
    prompt = ChatPromptTemplate.from_template("""
    You are a professional research analyst creating a final report based on research findings.
    
    ORIGINAL QUERY: {original_query}
    
    ANALYZED QUERY: {analyzed_query}
    
    SYNTHESIZED INFORMATION:
    {synthesized_info}
    
    {limitations_notice}
    
    Create a comprehensive research report addressing the original query.
    
    The report should:
    1. Use a professional, objective tone
    2. Be well-structured with clear sections
    3. Present information factually with appropriate confidence levels
    4. Acknowledge information gaps or limitations
    5. Provide recommendations when appropriate
    6. Include a methodology section describing how the research was conducted
    7. Organize information logically by topics and subtopics
    
    Format the report as Markdown with appropriate headings, bullet points, and formatting.
    Start with a title, then include an introduction summarizing the query and approach.
    """)
    
    limitations_notice = ""
    if low_confidence_flags:
        limitations_notice = "IMPORTANT RESEARCH LIMITATIONS:\n- " + "\n- ".join(low_confidence_flags)
        limitations_notice += "\n\nPlease clearly acknowledge these limitations in your report."
    
    response = llm.invoke(
        prompt.format(
            original_query=original_query,
            analyzed_query=analyzed_query,
            synthesized_info=json.dumps(synthesized_info, indent=2),
            limitations_notice=limitations_notice
        )
    )
    
    final_report = "RESEARCH REPORT\n"
    final_report += f"Query: {original_query}\n"
    final_report += "==================================================\n\n"
    final_report += response.content
    
    return {
        "final_report": final_report
    }
