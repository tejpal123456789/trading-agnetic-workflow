from langgraph.prebuilt import create_react_agent
from tools.toolkit import toolkit
from memory.longterm_memory import risk_manager_memory

def create_portfolio_manager_agent(llm, toolkit, state):
        """Create portfolio manager agent for final decision"""
        
        # Get past portfolio manager memories
        full_context = f"{state['trader_investment_plan']} Risk Debate: {state['risk_debate_state']['history']}"
        past_memories = risk_manager_memory.get_memories(full_context)
        past_memory_str = "\n".join([mem['recommendation'] for mem in past_memories])
        
        system_prompt = f"""You are the Portfolio Manager using create_react_agent approach.
        Your decision is FINAL and BINDING. You have ultimate authority over trading decisions.
        
        TRADER'S PROPOSAL:
        {state['trader_investment_plan']}
        
        COMPLETE RISK DEBATE:
        {state['risk_debate_state']['history']}
        
        INVESTMENT CONTEXT:
        Company: {state['company_of_interest']}
        Analysis Date: {state['trade_date']}
        Original Investment Plan: {state['investment_plan'][:300]}...
        
        PAST PORTFOLIO DECISIONS:
        {past_memory_str or 'No past portfolio decisions found.'}
        
        Your responsibilities:
        - Make the final, binding trading decision: BUY, SELL, or HOLD
        - Provide clear justification based on all available information
        - Use tools if needed for final market validation
        - Consider risk-adjusted returns and portfolio impact
        - Your decision will be executed immediately
        
        Provide authoritative, final decision with clear rationale."""
        
        all_tools = [getattr(toolkit, name) for name in dir(toolkit) if callable(getattr(toolkit, name)) and not name.startswith("__")]
        
        return create_react_agent(
            model=llm,
            tools=all_tools,
            prompt=system_prompt
        )
    