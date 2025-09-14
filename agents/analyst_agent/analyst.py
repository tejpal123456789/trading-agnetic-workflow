# Market Analyst: Focuses on technical indicators and price action.
from .base import create_analyst_node
from llm import quick_thinking_llm
from tools.toolkit import toolkit
from langgraph.prebuilt import create_react_agent


def create_market_agent(llm, toolkit, state=None):
    """Create the market analyst agent with state update capability"""
    system_prompt = f"""you are a trading assistant specialized in analyzing financial markets. 
your task is to perform a comprehensive technical market analysis for {state['company_of_interest']} as of {state['trade_date']}.
You need historical data spanning at least 3 months prior to {state['trade_date']} to calculate meaningful technical indicators."""
    all_tools_in_toolkit = [getattr(toolkit, name) for name in dir(toolkit) if callable(getattr(toolkit, name)) and not name.startswith("__")]
    
    base_agent = create_react_agent(
        model=llm,
        tools=all_tools_in_toolkit,
        prompt=system_prompt
    )
    
    # Wrapper function that updates state
    def agent_with_state_update(input_data):
        """Agent wrapper that automatically updates state"""
        
        # Run the base agent
        result = base_agent.invoke(input_data)
        
        # If state is provided, update it
        if state is not None:
            # Extract final report
            market_report = ""
            if result and "messages" in result and result["messages"]:
                final_message = result["messages"][-1]
                market_report = final_message.content if hasattr(final_message, 'content') else ""
            
            # Update state
            state["market_report"] = market_report
            state["messages"].extend(result["messages"])
            state["sender"] = "market_analyst"
            
            print(f"✅ State updated - Market report: {len(market_report)} characters")
        
        return result
    
    # Return the wrapper if state provided, otherwise return base agent
    return agent_with_state_update if state is not None else base_agent

def create_social_agent(llm, toolkit, state=None):
    """Create the social analyst agent with state update capability"""
    system_prompt = f"""You are a social media sentiment analyst specializing in financial markets. Analyze social sentiment around stocks from various platforms.
                    your task is task to analyze social sentiment for a given stock : {state['company_of_interest']} on a given date: {state['trade_date']}."""
    
    all_tools_in_toolkit = [getattr(toolkit, name) for name in dir(toolkit) if callable(getattr(toolkit, name)) and not name.startswith("__")]
    
    base_agent = create_react_agent(
        model=llm,
        tools=all_tools_in_toolkit,
        prompt=system_prompt
    )
    
    def agent_with_state_update(input_data):
        result = base_agent.invoke(input_data)
        
        if state is not None:
            sentiment_report = ""
            if result and "messages" in result and result["messages"]:
                final_message = result["messages"][-1]
                sentiment_report = final_message.content if hasattr(final_message, 'content') else ""
            
            state["sentiment_report"] = sentiment_report
            state["messages"].extend(result["messages"])
            state["sender"] = "social_analyst"
            
            print(f"✅ State updated - Sentiment report: {len(sentiment_report)} characters")
        
        return result
    
    return agent_with_state_update if state is not None else base_agent

def create_news_agent(llm, toolkit, state=None):
    """Create the news analyst agent with state update capability"""
    system_prompt = f"""You are a financial news analyst. Analyze recent news and its impact on stock performance.
                   your task is task to analyze recent news and its impact on stock performance for a given stock : {state['company_of_interest']} on a given date: {state['trade_date']}."""
    
    all_tools_in_toolkit = [getattr(toolkit, name) for name in dir(toolkit) if callable(getattr(toolkit, name)) and not name.startswith("__")]
    
    base_agent = create_react_agent(
        model=llm,
        tools=all_tools_in_toolkit,
        prompt=system_prompt
    )
    
    def agent_with_state_update(input_data):
        result = base_agent.invoke(input_data)
        
        if state is not None:
            news_report = ""
            if result and "messages" in result and result["messages"]:
                final_message = result["messages"][-1]
                news_report = final_message.content if hasattr(final_message, 'content') else ""
            
            state["news_report"] = news_report
            state["messages"].extend(result["messages"])
            state["sender"] = "news_analyst"
            
            print(f"✅ State updated - News report: {len(news_report)} characters")
        
        return result
    
    return agent_with_state_update if state is not None else base_agent

def create_fundamentals_agent(llm, toolkit, state=None):
    """Create the fundamentals analyst agent with state update capability"""
    system_prompt = f"""You are a fundamental analyst specializing in company financial analysis. Analyze financial statements and company metrics.
                   your task is task to analyze financial statements and company metrics for a given stock : {state['company_of_interest']} on a given date: {state['trade_date']}."""
    
    all_tools_in_toolkit = [getattr(toolkit, name) for name in dir(toolkit) if callable(getattr(toolkit, name)) and not name.startswith("__")]
    
    base_agent = create_react_agent(
        model=llm,
        tools=all_tools_in_toolkit,
        prompt=system_prompt
    )
    
    def agent_with_state_update(input_data):
        result = base_agent.invoke(input_data)
        
        if state is not None:
            fundamentals_report = ""
            if result and "messages" in result and result["messages"]:
                final_message = result["messages"][-1]
                fundamentals_report = final_message.content if hasattr(final_message, 'content') else ""
            
            state["fundamentals_report"] = fundamentals_report
            state["messages"].extend(result["messages"])
            state["sender"] = "fundamentals_analyst"
            
            print(f"✅ State updated - Fundamentals report: {len(fundamentals_report)} characters")
        
        return result
    
    return agent_with_state_update if state is not None else base_agent