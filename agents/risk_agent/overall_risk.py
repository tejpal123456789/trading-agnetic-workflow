from langgraph.prebuilt import create_react_agent
from tools.toolkit import toolkit


def create_risk_analyst_agent(llm, toolkit, state, risk_perspective):
        """Create risk analyst agent with specific perspective"""
        
        risk_prompts = {
            "risky": "You are the Risky Risk Analyst. You advocate for high-reward opportunities, bold strategies, and maximum position sizes. You believe in taking calculated risks for superior returns.",
            "safe": "You are the Safe/Conservative Risk Analyst. You prioritize capital preservation, risk minimization, and defensive strategies. You prefer smaller positions and tighter stop-losses.",
            "neutral": "You are the Neutral Risk Analyst. You provide balanced perspectives, weighing both opportunities and risks. You seek optimal risk-adjusted returns."
        }
        
        system_prompt = f"""{risk_prompts[risk_perspective]}
        
        Your role is to evaluate trading proposals from your risk perspective using create_react_agent.
        
        TRADER'S PROPOSAL:
        {state['trader_investment_plan']}
        
        CURRENT RISK DEBATE:
        {state['risk_debate_state']['history']}
        
        COMPANY CONTEXT:
        Company: {state['company_of_interest']}
        Analysis Date: {state['trade_date']}
        
        Your responsibilities:
        - Critique or support the trading proposal from your risk perspective
        - Use tools if needed to gather additional risk-related data
        - Present compelling arguments based on your risk philosophy
        - Counter other risk analysts' arguments effectively
        
        Provide thorough risk analysis from your unique perspective."""
        
        all_tools = [getattr(toolkit, name) for name in dir(toolkit) if callable(getattr(toolkit, name)) and not name.startswith("__")]
        
        return create_react_agent(
            model=llm,
            tools=all_tools,
            prompt=system_prompt
        )