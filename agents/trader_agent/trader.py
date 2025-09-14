from langgraph.prebuilt import create_react_agent
from tools.toolkit import toolkit
from memory.longterm_memory import trader_memory

def create_trader_agent(llm, toolkit, state):
        """Create trader agent using create_react_agent"""
        
        # Get past trader memories
        past_memories = trader_memory.get_memories(state['investment_plan'])
        past_memory_str = "\n".join([mem['recommendation'] for mem in past_memories])
        
        system_prompt = f"""You are a Professional Trader using create_react_agent approach.
        Your role is to convert investment plans into concrete, executable trading proposals.
        
        CURRENT INVESTMENT PLAN:
        {state['investment_plan']}
        
        COMPANY CONTEXT:
        Company: {state['company_of_interest']}
        Analysis Date: {state['trade_date']}
        Market Report: {state['market_report']}

        
        PAST TRADING EXPERIENCES:
        {past_memory_str or 'No past trading experiences found.'}
        
        Your responsibilities:
        - Create specific, actionable trading proposals
        - Include position sizing, entry/exit points, and risk management
        - Use tools if needed to get current market data for execution planning
        - Your response MUST end with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'
        
        Make your proposal practical and executable."""
        
        all_tools = [getattr(toolkit, name) for name in dir(toolkit) if callable(getattr(toolkit, name)) and not name.startswith("__")]
        
        return create_react_agent(
            model=llm,
            tools=all_tools,
            prompt=system_prompt
        )