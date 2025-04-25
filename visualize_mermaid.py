"""Visualize the Web Research Agent LangGraph using Mermaid diagram."""

from agent import create_web_research_agent

def main():
    # Create the agent graph
    graph = create_web_research_agent()
    
    # Print the Mermaid diagram representation
    mermaid_code = graph.get_graph().draw_mermaid()
    print(mermaid_code)
    
    print("\n")
    print("Copy the above code into a Markdown file with ```mermaid tags")
    print("or paste it into the Mermaid Live Editor: https://mermaid.live/")

if __name__ == "__main__":
    main() 