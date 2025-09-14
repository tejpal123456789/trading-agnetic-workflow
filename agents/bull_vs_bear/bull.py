from .base_bb import create_researcher_node
from llm import quick_thinking_llm
from memory.longterm_memory import bull_memory

from langgraph.prebuilt import create_react_agent
from tools.toolkit import toolkit


def create_bull_agent(llm, toolkit, state):
    """Create Bull researcher agent with state context"""
    
    # Prepare context from state
    situation_summary = f"""
    Market Report: {state['market_report']}
    Sentiment Report: {state['sentiment_report']}
    News Report: {state['news_report']}
    Fundamentals Report: {state['fundamentals_report']}
    
    Current debate history: {state['investment_debate_state']['history']}
    Bear's last argument: {state['investment_debate_state']['current_response']}
    Company: {state['company_of_interest']}
    Analysis Date: {state['trade_date']}
    """
    
    # Get memories from past similar situations
    past_memories = bull_memory.get_memories(situation_summary)
    past_memory_str = "\n".join([mem['recommendation'] for mem in past_memories])
    
    # Create system prompt with full context
    system_prompt = f"""You are a Bull Analyst. 
    Your goal is to argue for investing in the stock. 
    Focus on growth potential, competitive advantages, and positive indicators from the reports.
      Counter the bear's arguments effectively.
    
    CURRENT ANALYSIS CONTEXT:
    {situation_summary}
    
    PAST SIMILAR EXPERIENCES:
    {past_memory_str or 'No past memories found.'}
    
    Present compelling arguments for why this is a good investment opportunity based on the above context."""
    
    all_tools = [getattr(toolkit, name) for name in dir(toolkit) if callable(getattr(toolkit, name)) and not name.startswith("__")]
    
    return create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=system_prompt
    )