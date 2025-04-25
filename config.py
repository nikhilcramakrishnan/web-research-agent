"""Configuration for the Web Research Agent."""

import os
from typing import TypedDict, Dict, Any, Annotated
import operator
from dotenv import load_dotenv

load_dotenv()

# API keys for external services
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Language model settings
LLM_MODEL = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.1

# Search and scraping parameters
MAX_SEARCH_RESULTS = 10
MAX_NEWS_RESULTS = 5
MAX_URLS_TO_SCRAPE = 5
MAX_SEARCH_QUERIES = 5

# Retry and iteration limits
MAX_ITERATIONS = {"search_refinement": 3, "scraping_attempts": 2, "total_research": 8}

# Timeout for network requests (seconds)
SEARCH_TIMEOUT = 30

