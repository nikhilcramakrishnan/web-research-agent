import pytest
from nodes import evaluate_results_and_select_urls
import nodes

# Dummy response to simulate LLM behavior
class DummyResponse:
    def __init__(self, content):
        self.content = content


def test_evaluate_no_results(monkeypatch):
    state = {
        "web_results": [],
        "news_results": [],
        "error_log": [],
        "analyzed_query": {"info_type": ""},
        "original_query": "test",
        "research_plan": {}
    }
    updated = evaluate_results_and_select_urls(state)
    assert "error_log" in updated
    assert any(e.get("type") == "search_error" for e in updated["error_log"]), "Expected search_error in error_log"
    assert updated.get("next_node") == "plan_research_strategy"
    assert updated.get("urls_to_scrape") == []


def test_evaluate_invalid_json(monkeypatch):
    monkeypatch.setattr(nodes.ChatGoogleGenerativeAI, "invoke", lambda self, *args, **kwargs: DummyResponse("not a json"))
    state = {
        "web_results": [{"url": "http://example.com", "snippet": "snippet", "title": "title"}],
        "news_results": [],
        "error_log": [],
        "analyzed_query": {"info_type": "facts"},
        "original_query": "test",
        "research_plan": {}
    }
    updated = evaluate_results_and_select_urls(state)
    assert updated.get("urls_to_scrape") == []
    assert updated.get("next_node") == "plan_research_strategy"


def test_evaluate_snippets_sufficient(monkeypatch):
    content = '{"snippets_sufficient": true, "urls_to_scrape": [], "refine_search": false}'
    monkeypatch.setattr(nodes.ChatGoogleGenerativeAI, "invoke", lambda self, *args, **kwargs: DummyResponse(content))
    state = {
        "web_results": [{"url": "http://example.com", "snippet": "snippet", "title": "title"}],
        "news_results": [],
        "error_log": [],
        "analyzed_query": {"info_type": "facts"},
        "original_query": "simple query",
        "research_plan": {}
    }
    updated = evaluate_results_and_select_urls(state)
    assert updated.get("urls_to_scrape") == []
    assert updated.get("next_node") == "extract_and_synthesize_information" 