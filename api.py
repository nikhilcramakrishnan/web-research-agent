""" Functionality for the API. Endpoints for the API are defined here."""

import os
from dotenv import load_dotenv
load_dotenv()
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from agent import run_web_research_agent

app = FastAPI(
    title="Web Research Agent API",
    description="API for running the Web Research Agent programmatically",
    version="1.0"
)
# Allow all origins for development purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    query: str

class ResearchResponse(BaseModel):
    query: str
    report: str

@app.get("/", summary="Health check")
def root():
    return {"message": "Web Research Agent API is running"}

@app.post("/research", response_model=ResearchResponse, summary="Run research agent")
def run_research(req: ResearchRequest):
    """
    Run the web research agent on the given query and return the final report.
    """
    try:
        result = run_web_research_agent(req.query)
        report = result.get("final_report", "")
        return ResearchResponse(query=req.query, report=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
    

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)