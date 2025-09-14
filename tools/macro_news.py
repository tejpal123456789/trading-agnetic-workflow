

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
# Initialize the Tavily search tool once. We can reuse this instance for multiple specialized tools.
from dotenv import load_dotenv
load_dotenv()
tavily_tool = TavilySearchResults(max_results=3)

@tool
def get_macroeconomic_news(trade_date: str) -> str:
    """Performs a live web search for macroeconomic news relevant to the stock market."""
    query = f"macroeconomic news and market trends affecting the stock market on {trade_date}"
    return tavily_tool.invoke({"query": query})