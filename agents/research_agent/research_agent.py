# This function creates the Research Manager node.
from llm import deep_thinking_llm
from memory.longterm_memory import invest_judge_memory
from langgraph.prebuilt import create_react_agent
from tools.toolkit import toolkit

def create_research_manager_agent(llm, toolkit, state):
    """Create Research Manager agent with full context"""
    
    # Prepare comprehensive context from state
    full_context = f"""
    ANALYSIS REPORTS:
    Market Report: {state['market_report']}
    Sentiment Report: {state['sentiment_report']}
    News Report: {state['news_report']}
    Fundamentals Report: {state['fundamentals_report']}
    
    BULL VS BEAR DEBATE:
    Full Debate History: {state['investment_debate_state']['history']}
    Bull Arguments: {state['investment_debate_state']['bull_history']}
    Bear Arguments: {state['investment_debate_state']['bear_history']}
    Debate Rounds: {state['investment_debate_state']['count']}
    
    COMPANY DETAILS:
    Company: {state['company_of_interest']}
    Analysis Date: {state['trade_date']}
    """
    
    # Get memories from past similar investment decisions
    past_memories = invest_judge_memory.get_memories(full_context)
    past_memory_str = "\n".join([mem['recommendation'] for mem in past_memories])
    
    # Create comprehensive system prompt
    system_prompt = f"""You are a Research Manager using create_react_agent approach.
    Your role is to make final investment decisions based on comprehensive analysis.
    
    RESPONSIBILITIES:
    - Critically evaluate the debate between Bull and Bear analysts
    - Synthesize all available information (reports + debate arguments)
    - Make a definitive investment decision: BUY, SELL, or HOLD
    - Develop a detailed investment plan with clear rationale
    - Assess risks and provide mitigation strategies
    - Use tools if needed to gather additional market context or validation
    
    CURRENT COMPREHENSIVE CONTEXT:
    {full_context}
    
    PAST SIMILAR INVESTMENT DECISIONS:
    {past_memory_str or 'No past investment decisions found.'}
    
    Based on all available information, provide a clear, actionable investment recommendation with detailed reasoning."""
    
    all_tools = [getattr(toolkit, name) for name in dir(toolkit) if callable(getattr(toolkit, name)) and not name.startswith("__")]
    
    return create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=system_prompt
    )