"""Main entry point for the Web Research Agent."""

import argparse
import logging
import os
import sys
import json
from typing import Optional
from agent import run_web_research_agent
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Check and set up required environment variables."""
    missing_vars = []

    # Check for required environment variables and ask user whether to proceed without them
    
    if not os.environ.get("GOOGLE_API_KEY"):
        missing_vars.append("GOOGLE_API_KEY")
    
    if not os.environ.get("TAVILY_API_KEY_BACKUP"):
        missing_vars.append("TAVILY_API_KEY_BACKUP")
    
    if not os.environ.get("NEWS_API_KEY"):
        missing_vars.append("NEWS_API_KEY")
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.warning("Some functionality may be limited or use mock responses.")
        
        if len(missing_vars) > 0 and not sys.argv[0].endswith('pytest'):
            proceed = input("Continue without all required API keys? (y/n): ")
            if proceed.lower() != 'y':
                logger.error("Exiting due to missing API keys")
                sys.exit(1)


def save_report(report: str, query: str, output_file: Optional[str] = None):
    """
    Save the research report to a file.
    
    Args:
        report: The research report text
        query: The original query
        output_file: Optional filename to save to. If None, generates a file name.
    """
    if not output_file:
        safe_query = ''.join(c if c.isalnum() else '_' for c in query[:30])
        output_file = f'research_report_{safe_query}.txt'
    
    # Ensure test_reports directory exists
    reports_dir = 'test_reports'
    os.makedirs(reports_dir, exist_ok=True)
    filename = os.path.basename(output_file)
    output_file = os.path.join(reports_dir, filename)
    
    try:
        with open(output_file, "w") as f:
            f.write(f"RESEARCH REPORT\n")
            f.write(f"Query: {query}\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(report)
        
        logger.info(f"Report saved to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error saving report: {str(e)}")
        return None


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Web Research Agent")
    parser.add_argument(
        "query", nargs="?", 
        help="Research query to investigate"
    )
    parser.add_argument(
        "-o", "--output", 
        help="Output file for the research report"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "-d", "--debug-search", action="store_true",
        help="Enable detailed DEBUG logging for search results"
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "-n", "--no-news", action="store_true",
        help="Disable news search functionality (web search only)"
    )
    
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.debug_search:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("tools").setLevel(logging.DEBUG)
    

    if args.no_news:
        os.environ["DISABLE_NEWS_SEARCH"] = "true"
        logger.info("News search functionality disabled")
    
    setup_environment()
    
    if args.interactive:
        run_interactive_mode(args.output)
    elif args.query:
        run_single_query(args.query, args.output)
    else:
        parser.print_help()


def run_single_query(query: str, output_file: Optional[str] = None):
    """
    Run a single research query.
    
    Args:
        query: The research query
        output_file: Optional file to save the report to
    """
    print(f"Researching: {query}")
    print("This may take a few minutes...")
    
    try:
        result = run_web_research_agent(query)
        
        report = result.get("final_report", "Error: No report generated")
        
        print("\n" + "=" * 50)
        print("RESEARCH REPORT")
        print("=" * 50)
        print(report)
        print("=" * 50 + "\n")
        
        if output_file or input("Save report to file? (y/n): ").lower() == 'y':
            save_report(report, query, output_file)
    
    except Exception as e:
        logger.error(f"Error during research: {str(e)}")
        print(f"An error occurred: {str(e)}")


def run_interactive_mode(default_output_file: Optional[str] = None):
    """
    Run the agent in interactive mode, allowing multiple queries.
    
    Args:
        default_output_file: Default file pattern to save reports
    """
    print("Web Research Agent - Interactive Mode")
    print("Type 'exit' or 'quit' to end the session")
    
    while True:
        query = input("\nEnter research query: ")
        
        if query.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        
        if not query.strip():
            continue
        
        try:
            print(f"Researching: {query}")
            print("This may take a few minutes...")
            
            result = run_web_research_agent(query)
            
            report = result.get("final_report", "Error: No report generated")
            
            print("\n" + "=" * 50)
            print("RESEARCH REPORT")
            print("=" * 50)
            print(report)
            print("=" * 50 + "\n")
            
            if input("Save report to file? (y/n): ").lower() == 'y':
                output_file = input(f"Enter filename (or press Enter for auto-generated): ")
                save_report(report, query, output_file or default_output_file)
        
        except Exception as e:
            logger.error(f"Error during research: {str(e)}")
            print(f"An error occurred: {str(e)}")
            
        print("\nReady for next query.")


if __name__ == "__main__":
    main() 