import importlib
import pytest

# Test missing API key scenario

def test_missing_google_api_key(monkeypatch):
    # Prevent loading .env file and remove cached modules
    import dotenv, sys
    import importlib
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    sys.modules.pop("config", None)
    sys.modules.pop("nodes", None)
    with pytest.raises(ValueError) as excinfo:
        import nodes 
    assert "GOOGLE_API_KEY is required" in str(excinfo.value)

class DummyAgent:
    def invoke(self, state):
        raise Exception("fail")


def test_run_web_research_agent_error(monkeypatch):
    # Ensure API key is present for agent initialization
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy_key")
    import sys, importlib
    sys.modules.pop("config", None)
    sys.modules.pop("nodes", None)
    sys.modules.pop("agent", None)
    import agent
    importlib.reload(agent)
    monkeypatch.setattr(agent, "create_web_research_agent", lambda: DummyAgent())
    result = agent.run_web_research_agent("test query")
    assert "error_log" in result
    err = result["error_log"][0]
    assert err["type"] == "agent_error"
    assert "fail" in err["message"]


def test_plan_research_strategy_exceeds_max():
    from nodes import plan_research_strategy
    state = {
        "analyzed_query": {},
        "iteration_count": {"total_research": 5},
        "max_iterations": {"total_research": 3}
    }
    updated = plan_research_strategy(state)
    # It should increment iteration count
    assert updated["iteration_count"]["total_research"] == 6
    assert updated["research_plan"]["search_approach"] == "extract_and_synthesize_information" 