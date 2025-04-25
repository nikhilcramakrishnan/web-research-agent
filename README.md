# Web Research Agent

A LangGraph-based autonomous agent powered by Google's Gemini that can search the web, find relevant information, extract data from websites, and compile comprehensive research reports based on user queries.

## Overview

This agent functions as a digital research assistant, performing these tasks with minimal human input:

- Analyze complex research queries to understand user intent
- Plan and execute appropriate search strategies
- Find relevant information from web searches and news sources
- Extract and process data from multiple websites
- Synthesize information
- Generate comprehensive research reports

## More information

The agent uses a LangGraph-based architecture with a state-driven approach.
For more information read [documentation](./documentation.pdf)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nikhilcramakrishnan/web-research-agent.git
   cd web-research-agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (create a `.env` file):
   ```
   GOOGLE_API_KEY=your_google_api_key
   NEWS_API_KEY=your_news_api_key  #  Required for news search (NEWSAPI.org) 
   TAVILY_API_KEY=your_tavily_api_key  # Required for web search
   ```

4. Install frontend dependencies and start the React development server:
   ```bash
   cd frontend
   npm install
   npm start
   ```
## How to run ?
The app can be run both directly and using API mode. A React frontend is available in the `frontend` folder to improve experience.

### Running directly
Run a single research query:
```bash
python main.py "What are the latest developments in quantum computing?"
```

Save the report to a file:
```bash
python main.py "What are the latest developments in quantum computing?" -o quantum_report.txt
```

### API Usage

Run the FastAPI server:

```bash
uvicorn api:app --reload
```

Send a POST request to `/research`:

```bash
curl -X POST "http://localhost:8000/research" \
     -H "Content-Type: application/json" \
     -d '{"query":"What are the latest developments in quantum computing?"}'
```



You can also use the Python API directly:

```python
from agent import run_web_research_agent

result = run_web_research_agent("Your query here")
print(result.get("final_report"))


```

Also you can use the react frontend for a better user experience.
NB: The deployed model on vercel, would have less performance as our scrapping tool can't be installed on free tier limitations. Please understand that web scrapping won't work in that scenerio.