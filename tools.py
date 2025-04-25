"""Tool implementations for the Web Research Agent."""

import json
import logging
import requests
import time
import os
from typing import Dict, List, Any, Optional
import config
import datetime
import re
import random

logger = logging.getLogger(__name__)

def redact_api_key_from_url(url):
    """
    Redact API keys from URLs to prevent them from being logged.
    """
    redacted_url = re.sub(r'(api_?[kK]ey=)[^&]+', r'\1[REDACTED]', url)
    return redacted_url

class TavilySearchTool:
    """Tool for performing web searches using Tavily's Search API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY") or config.TAVILY_API_KEY
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is required for TavilySearchTool")
        
        self.base_url = "https://api.tavily.com/search"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def search(self, query: str, num_results: int = config.MAX_SEARCH_RESULTS) -> List[Dict[str, Any]]:
        """
        Perform a web search using Tavily's Search API.
        
        Args:
            query: The search query
            num_results: Maximum number of results to return
            
        Returns:
            List of search results with url, title, and snippet
        """
        logger.info(f"Performing Tavily web search for: {query}")
        
        try:
            payload = {
                "query": query,
                "max_results": min(num_results, 20),  
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
                "topic": "general"
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            if "results" in data:
                for result in data["results"][:num_results]:
                    results.append({
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "snippet": result.get("content", "")
                    })
            
            logger.info(f"Tavily web search returned {len(results)} results")
            
            if logger.level <= logging.DEBUG:
                for i, result in enumerate(results):
                    logger.debug(f"Result {i+1}:")
                    logger.debug(f"  Title: {result.get('title', 'No title')[:100]}...")
                    logger.debug(f"  URL: {result.get('url', 'No URL')}")
                    snippet = result.get('snippet', 'No snippet')
                    logger.debug(f"  Snippet: {snippet[:150]}..." if len(snippet) > 150 else f"  Snippet: {snippet}")
            
            return results
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error in Tavily web search: {str(e)}")
            results = []
            for i in range(min(3, num_results)):
                results.append({
                    "url": f"https://example.com/result{i}",
                    "title": f"Example Result {i} for {query}",
                    "snippet": f"This is a fallback snippet related to {query}. Tavily API call failed: {str(e)}",
                })
            return results

class WebSearchTool:
    """
    DEPRECATED: This tool is kept for backward compatibility.
    Please use TavilySearchTool instead for web searches.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        logger.warning("WebSearchTool is deprecated. Please use TavilySearchTool instead.")
        self.tavily_tool = TavilySearchTool(api_key)
    
    def search(self, query: str, num_results: int = config.MAX_SEARCH_RESULTS) -> List[Dict[str, Any]]:
        """
        Forwards the search to TavilySearchTool.
        
        Args:
            query: The search query
            num_results: Maximum number of results to return
            
        Returns:
            List of search results with url, title, and snippet
        """
        return self.tavily_tool.search(query, num_results)

class NewsAggregatorTool:
    """Tool for aggregating news from NewsAPI.org."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("NEWS_API_KEY") or config.NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2/everything"
        if not self.api_key:
            logger.warning("NEWS_API_KEY not provided. News search will return mock data.")
    
    def search_news(self, query: str, days_back: int = 7,
                   num_results: int = config.MAX_NEWS_RESULTS) -> List[Dict[str, Any]]:
        """
        Search for recent news articles related to the query.

        Args:
            query: News search query
            days_back: How many days back to search
            num_results: Maximum number of results to return

        Returns:
            List of news results with url, title, summary, date, and source
        """
        logger.info(f"Searching news for: {query}")
        results: List[Dict[str, Any]] = []
        # If no API key, return mock data
        if not self.api_key:
            logger.info("NewsAPI key missing, returning mock data")
            current_time = time.time()
            for i in range(min(3, num_results)):
                days_ago = i % days_back
                timestamp = time.strftime("%Y-%m-%d", time.localtime(current_time - days_ago * 86400))
                results.append({
                    "url": f"https://news-example.com/article{i}",
                    "title": f"[MOCK] News Article {i} about {query}",
                    "summary": "NewsAPI key missing. This is mock data.",
                    "date": timestamp,
                    "source": f"Mock News Source {i}",
                })
            logger.info(f"News search (MOCK) returned {len(results)} results")
            return results
        # Perform real NewsAPI request
        try:
            # Calculate the 'from' date based on days_back
            from_date = (datetime.datetime.utcnow() - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
            params = {
                "q": query,
                "from": from_date,
                "sortBy": "relevancy",
                "pageSize": num_results,
                "apiKey": self.api_key
            }
            response = requests.get(self.base_url, params=params, timeout=config.SEARCH_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            articles = data.get("articles", [])
            for art in articles[:num_results]:
                results.append({
                    "url": art.get("url", ""),
                    "title": art.get("title", ""),
                    "summary": art.get("description") or art.get("content") or "",
                    "date": art.get("publishedAt", "")[:10],
                    "source": art.get("source", {}).get("name", ""),
                })
            logger.info(f"News search returned {len(results)} articles")
            return results
        except requests.exceptions.RequestException as e:
            logger.error(f"Error in NewsAPI search: {str(e)}")
            # Fallback to mock data on error
            current_time = time.time()
            for i in range(min(3, num_results)):
                days_ago = i % days_back
                timestamp = time.strftime("%Y-%m-%d", time.localtime(current_time - days_ago * 86400))
                results.append({
                    "url": f"https://news-example.com/article{i}",
                    "title": f"[MOCK] News Article {i} about {query}",
                    "summary": f"NewsAPI search failed: {str(e)}. This is mock data.",
                    "date": timestamp,
                    "source": f"Mock News Source {i}",
                })
            logger.info(f"News search (fallback) returned {len(results)} mock results")
            return results