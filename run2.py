from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from agent_state import AgentState, InvestDebateState, RiskDebateState
from langchain_core.messages import HumanMessage, AIMessage
from agents.analyst_agent.analyst import create_market_agent, create_social_agent, create_news_agent, create_fundamentals_agent
from tools.toolkit import toolkit
from rich.console import Console
from rich.markdown import Markdown
from llm import quick_thinking_llm, deep_thinking_llm
from memory.longterm_memory import bull_memory, bear_memory, invest_judge_memory, trader_memory, risk_manager_memory
import datetime
from concurrent.futures import ThreadPoolExecutor
from agents.bull_vs_bear.bull import create_bull_agent
from agents.bull_vs_bear.bear import create_bear_agent
from agents.research_agent.research_agent import create_research_manager_agent
from agents.trader_agent.trader import create_trader_agent
from agents.risk_agent.overall_risk import create_risk_analyst_agent
from agents.portfolio_manager_agent.portfolio_agent import create_portfolio_manager_agent
import json
import functools

console = Console()

class CompleteTradingWorkflow:
    def __init__(self):
        self.graph = self._build_graph()
        self.shared_state = None
    
    def _build_graph(self):
        """Build the complete LangGraph workflow with trader and risk management"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("initialization", self.initialization_node)
        workflow.add_node("parallel_analysis", self.parallel_analysis_node)
        workflow.add_node("bull_researcher", self.bull_researcher_node)
        workflow.add_node("bear_researcher", self.bear_researcher_node)
        workflow.add_node("research_manager", self.research_manager_node)
        
        # NEW NODES: Trader and Risk Management
        workflow.add_node("trader", self.trader_node)
        workflow.add_node("risky_analyst", self.risky_analyst_node)
        workflow.add_node("safe_analyst", self.safe_analyst_node)
        workflow.add_node("neutral_analyst", self.neutral_analyst_node)
        workflow.add_node("portfolio_manager", self.portfolio_manager_node)
        
        workflow.add_node("consolidation", self.consolidation_node)
        
        # Define edges - Extended flow
        workflow.add_edge(START, "initialization")
        workflow.add_edge("initialization", "parallel_analysis")
        
        # Investment debate cycle
        workflow.add_edge("parallel_analysis", "bull_researcher")
        workflow.add_edge("bull_researcher", "bear_researcher")
        
        workflow.add_conditional_edges(
            "bear_researcher",
            self.should_continue_debate,
            {
                "continue": "bull_researcher",
                "end": "research_manager"
            }
        )
        
        # NEW FLOW: Research Manager → Trader → Risk Management
        workflow.add_edge("research_manager", "trader")
        workflow.add_edge("trader", "risky_analyst")
        workflow.add_edge("risky_analyst", "safe_analyst")
        workflow.add_edge("safe_analyst", "neutral_analyst")
        
        # Risk management debate cycle
        workflow.add_conditional_edges(
            "neutral_analyst",
            self.should_continue_risk_debate,
            {
                "continue": "risky_analyst",
                "end": "portfolio_manager"
            }
        )
        
        workflow.add_edge("portfolio_manager", "consolidation")
        workflow.add_edge("consolidation", END)
        
        return workflow.compile()
    
    def initialization_node(self, state: AgentState) -> AgentState:
        """Initialize the workflow"""
        console.print("[bold blue]🚀 Initializing Complete Trading Analysis Workflow[/bold blue]")
        console.print(f"[green]Company:[/green] {state['company_of_interest']}")
        console.print(f"[green]Trade Date:[/green] {state['trade_date']}")
        
        init_message = AIMessage(
            content=f"Starting comprehensive analysis, debate, and trading execution for {state['company_of_interest']} on {state['trade_date']}"
        )
        
        return {
            **state,
            "messages": state["messages"] + [init_message],
            "sender": "initialization"
        }
    
    def parallel_analysis_node(self, state: AgentState) -> AgentState:
        """Execute all analysts in parallel"""
        console.print("[bold yellow]📊 Running Parallel Analysis...[/bold yellow]")
        
        # Create agents with shared state reference
        market_agent = create_market_agent(quick_thinking_llm, toolkit, state)
        social_agent = create_social_agent(quick_thinking_llm, toolkit, state)
        news_agent = create_news_agent(quick_thinking_llm, toolkit, state)
        fundamentals_agent = create_fundamentals_agent(quick_thinking_llm, toolkit, state)
        
        def run_market():
            console.print("[cyan]📈 Market Analyst starting...[/cyan]")
            return market_agent({
                "messages": [HumanMessage(content=f"Perform comprehensive technical market analysis for {state['company_of_interest']} on {state['trade_date']}")]
            })
        
        def run_social():
            console.print("[cyan]💬 Social Analyst starting...[/cyan]")
            return social_agent({
                "messages": [HumanMessage(content=f"Analyze social media sentiment for {state['company_of_interest']} on {state['trade_date']}")]
            })
        
        def run_news():
            console.print("[cyan]📰 News Analyst starting...[/cyan]")
            return news_agent({
                "messages": [HumanMessage(content=f"Analyze recent news impact for {state['company_of_interest']} on {state['trade_date']}")]
            })
        
        def run_fundamentals():
            console.print("[cyan]🏗️ Fundamentals Analyst starting...[/cyan]")
            return fundamentals_agent({
                "messages": [HumanMessage(content=f"Perform fundamental analysis for {state['company_of_interest']} on {state['trade_date']}")]
            })
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(run_market),
                executor.submit(run_social),
                executor.submit(run_news),
                executor.submit(run_fundamentals)
            ]
            
            for i, future in enumerate(futures):
                try:
                    future.result()
                    analyst_names = ["Market", "Social", "News", "Fundamentals"]
                    console.print(f"[green]✅ {analyst_names[i]} Analyst completed[/green]")
                except Exception as e:
                    analyst_names = ["Market", "Social", "News", "Fundamentals"]
                    console.print(f"[red]❌ {analyst_names[i]} Analyst failed: {str(e)}[/red]")
        
        console.print("[bold green]🎉 Parallel analysis completed![/bold green]")
        return {**state, "sender": "parallel_analysis"}
    
    def bull_researcher_node(self, state: AgentState) -> AgentState:
        """Bull researcher node using create_react_agent"""
        console.print(f"[bold green]🐂 Bull Researcher - Round {state['investment_debate_state']['count'] + 1}[/bold green]")
        
        bull_agent = create_bull_agent(quick_thinking_llm, toolkit, state)
        prompt = f"Present your strongest bull case for {state['company_of_interest']}. Make compelling arguments for why this stock should be bought."
        
        result = bull_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        
        bull_argument = ""
        if result and "messages" in result and result["messages"]:
            final_message = result["messages"][-1]
            bull_argument = f"Bull Analyst: {final_message.content}"
        
        debate_state = state['investment_debate_state'].copy()
        debate_state['history'] += "\n" + bull_argument
        debate_state['bull_history'] += "\n" + bull_argument
        debate_state['current_response'] = bull_argument
        debate_state['count'] += 1
        
        console.print("[green]🐂 Bull's Argument:[/green]")
        console.print(Markdown(bull_argument.replace('Bull Analyst: ', '')))
        
        return {
            **state,
            "investment_debate_state": debate_state,
            "messages": state["messages"] + result["messages"] if result and "messages" in result else state["messages"],
            "sender": "bull_researcher"
        }
    
    def bear_researcher_node(self, state: AgentState) -> AgentState:
        """Bear researcher node using create_react_agent"""
        console.print(f"[bold red]🐻 Bear Researcher - Round {state['investment_debate_state']['count']}[/bold red]")
        
        bear_agent = create_bear_agent(quick_thinking_llm, toolkit, state)
        prompt = f"Present your strongest bear case for {state['company_of_interest']}. Make compelling arguments for why this stock should be avoided or sold."
        
        result = bear_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        
        bear_argument = ""
        if result and "messages" in result and result["messages"]:
            final_message = result["messages"][-1]
            bear_argument = f"Bear Analyst: {final_message.content}"
        
        debate_state = state['investment_debate_state'].copy()
        debate_state['history'] += "\n" + bear_argument
        debate_state['bear_history'] += "\n" + bear_argument
        debate_state['current_response'] = bear_argument
        
        situation_context = f"{state['market_report'][:200]}... Company: {state['company_of_interest']}"
        bear_memory.add_situations([(situation_context, bear_argument)])
        
        console.print("[red]🐻 Bear's Rebuttal:[/red]")
        console.print(Markdown(bear_argument.replace('Bear Analyst: ', '')))
        
        return {
            **state,
            "investment_debate_state": debate_state,
            "messages": state["messages"] + result["messages"] if result and "messages" in result else state["messages"],
            "sender": "bear_researcher"
        }
    
    def should_continue_debate(self, state: AgentState) -> str:
        """Decide whether to continue the investment debate"""
        max_rounds = 1
        current_round = state['investment_debate_state']['count']
        
        if current_round >= max_rounds:
            console.print(f"[yellow]📊 Investment debate completed after {current_round} rounds[/yellow]")
            return "end"
        else:
            console.print(f"[yellow]🔄 Continuing investment debate - Round {current_round + 1}[/yellow]")
            return "continue"
    
    def research_manager_node(self, state: AgentState) -> AgentState:
        """Research manager node using create_react_agent"""
        console.print("[bold purple]👨‍💼 Research Manager - Making Investment Decision[/bold purple]")
        
        manager_agent = create_research_manager_agent(deep_thinking_llm, toolkit, state)
        
        prompt = f"""As Research Manager, evaluate all information and make your investment decision for {state['company_of_interest']}.
        
        Provide:
        1. Executive Summary of key bull and bear points
        2. Your decision: BUY, SELL, or HOLD (be definitive)
        3. Detailed investment plan with specific actions and timeline
        4. Risk assessment and mitigation strategies
        5. Price targets and exit conditions if applicable
        
        Make this decision actionable for traders."""
        
        result = manager_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        
        investment_plan = ""
        if result and "messages" in result and result["messages"]:
            final_message = result["messages"][-1]
            investment_plan = final_message.content
        
        debate_state = state['investment_debate_state'].copy()
        debate_state['judge_decision'] = investment_plan
        
        decision_context = f"""
        Company: {state['company_of_interest']}
        Date: {state['trade_date']}
        Market Report Summary: {state['market_report'][:300]}...
        Debate Summary: Bull vs Bear had {state['investment_debate_state']['count']} rounds
        Final Decision: {investment_plan[:200]}...
        """
        invest_judge_memory.add_situations([(decision_context, investment_plan)])
        
        console.print("[bold purple]👨‍💼 Research Manager Decision:[/bold purple]")
        console.print(Markdown(investment_plan))
        
        return {
            **state,
            "investment_debate_state": debate_state,
            "investment_plan": investment_plan,
            "messages": state["messages"] + result["messages"] if result and "messages" in result else state["messages"],
            "sender": "research_manager"
        }
    
    # NEW NODES: Trader and Risk Management
    
    def trader_node(self, state: AgentState) -> AgentState:
        """Trader node - creates executable trading proposal"""
        console.print("[bold blue]💼 Trader - Creating Trading Proposal[/bold blue]")
        
        trader_agent = create_trader_agent(quick_thinking_llm, toolkit, state)
        
        prompt = f"Based on the investment plan, create a specific trading proposal for {state['company_of_interest']}. Include position sizing, entry points, stop losses, and execution strategy."
        
        result = trader_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        
        trader_investment_plan = ""
        if result and "messages" in result and result["messages"]:
            final_message = result["messages"][-1]
            trader_investment_plan = final_message.content
        
        # Save trader experience to memory
        trading_context = f"Investment Plan: {state['investment_plan'][:200]}... Company: {state['company_of_interest']}"
        trader_memory.add_situations([(trading_context, trader_investment_plan)])
        
        console.print("[bold blue]💼 Trader's Proposal:[/bold blue]")
        console.print(Markdown(trader_investment_plan))
        
        return {
            **state,
            "trader_investment_plan": trader_investment_plan,
            "messages": state["messages"] + result["messages"] if result and "messages" in result else state["messages"],
            "sender": "trader"
        }
    

    def risky_analyst_node(self, state: AgentState) -> AgentState:
        """Risky risk analyst node"""
        console.print(f"[bold red]🎲 Risky Analyst - Risk Round {state['risk_debate_state']['count'] + 1}[/bold red]")
        
        risky_agent = create_risk_analyst_agent(quick_thinking_llm, toolkit, state, "risky")
        
        prompt = f"Evaluate the trader's proposal from an aggressive, high-reward perspective. Argue for taking maximum advantage of this opportunity."
        
        result = risky_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        
        risky_response = ""
        if result and "messages" in result and result["messages"]:
            final_message = result["messages"][-1]
            risky_response = final_message.content
        
        risk_state = state['risk_debate_state'].copy()
        risk_state['history'] += f"\nRisky Analyst: {risky_response}"
        risk_state['current_risky_response'] = risky_response
        risk_state['risky_history'] += f"\nRisky Analyst: {risky_response}"
        risk_state['latest_speaker'] = "Risky Analyst"
        risk_state['count'] += 1
        
        console.print("[red]🎲 Risky Analyst's View:[/red]")
        console.print(Markdown(risky_response))
        
        return {
            **state,
            "risk_debate_state": risk_state,
            "messages": state["messages"] + result["messages"] if result and "messages" in result else state["messages"],
            "sender": "risky_analyst"
        }
    
    def safe_analyst_node(self, state: AgentState) -> AgentState:
        """Safe risk analyst node"""
        console.print(f"[bold green]🛡️ Safe Analyst - Risk Round {state['risk_debate_state']['count']}[/bold green]")
        
        safe_agent = create_risk_analyst_agent(quick_thinking_llm, toolkit, state, "safe")
        
        prompt = f"Evaluate the trader's proposal from a conservative, risk-averse perspective. Focus on capital preservation and downside protection."
        
        result = safe_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        
        safe_response = ""
        if result and "messages" in result and result["messages"]:
            final_message = result["messages"][-1]
            safe_response = final_message.content
        
        risk_state = state['risk_debate_state'].copy()
        risk_state['history'] += f"\nSafe Analyst: {safe_response}"
        risk_state['current_safe_response'] = safe_response
        risk_state['safe_history'] += f"\nSafe Analyst: {safe_response}"
        risk_state['latest_speaker'] = "Safe Analyst"
        
        console.print("[green]🛡️ Safe Analyst's View:[/green]")
        console.print(Markdown(safe_response))
        
        return {
            **state,
            "risk_debate_state": risk_state,
            "messages": state["messages"] + result["messages"] if result and "messages" in result else state["messages"],
            "sender": "safe_analyst"
        }
    
    def neutral_analyst_node(self, state: AgentState) -> AgentState:
        """Neutral risk analyst node"""
        console.print(f"[bold yellow]⚖️ Neutral Analyst - Risk Round {state['risk_debate_state']['count']}[/bold yellow]")
        
        neutral_agent = create_risk_analyst_agent(quick_thinking_llm, toolkit, state, "neutral")
        
        prompt = f"Evaluate the trader's proposal from a balanced perspective. Weigh both the opportunities and risks objectively."
        
        result = neutral_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        
        neutral_response = ""
        if result and "messages" in result and result["messages"]:
            final_message = result["messages"][-1]
            neutral_response = final_message.content
        
        risk_state = state['risk_debate_state'].copy()
        risk_state['history'] += f"\nNeutral Analyst: {neutral_response}"
        risk_state['current_neutral_response'] = neutral_response
        risk_state['neutral_history'] += f"\nNeutral Analyst: {neutral_response}"
        risk_state['latest_speaker'] = "Neutral Analyst"
        
        console.print("[yellow]⚖️ Neutral Analyst's View:[/yellow]")
        console.print(Markdown(neutral_response))
        
        return {
            **state,
            "risk_debate_state": risk_state,
            "messages": state["messages"] + result["messages"] if result and "messages" in result else state["messages"],
            "sender": "neutral_analyst"
        }
    
    def should_continue_risk_debate(self, state: AgentState) -> str:
        """Decide whether to continue the risk debate"""
        max_rounds = 1  # Risk debate rounds
        current_round = state['risk_debate_state']['count']
        
        if current_round >= max_rounds * 3:  # 3 analysts per round
            console.print(f"[yellow]🛡️ Risk debate completed after {current_round // 3} rounds[/yellow]")
            return "end"
        else:
            console.print(f"[yellow]🔄 Continuing risk debate - Round {(current_round // 3) + 1}[/yellow]")
            return "continue"
    
    
    def portfolio_manager_node(self, state: AgentState) -> AgentState:
        """Portfolio manager node - makes final binding decision"""
        console.print("[bold magenta]👑 Portfolio Manager - Final Decision[/bold magenta]")
        
        portfolio_manager_agent = create_portfolio_manager_agent(deep_thinking_llm, toolkit, state)
        
        prompt = f"""As Portfolio Manager, review the trader's proposal and complete risk debate. Make your final, binding decision for {state['company_of_interest']}.
        
        Provide:
        1. Summary of key risk points from all three analysts
        2. Your final decision: BUY, SELL, or HOLD (be definitive)
        3. Justification based on risk-adjusted returns
        4. Final execution instructions
        
        Your decision will be implemented immediately."""
        
        result = portfolio_manager_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        
        final_trade_decision = ""
        if result and "messages" in result and result["messages"]:
            final_message = result["messages"][-1]
            final_trade_decision = final_message.content
        
        # Update risk debate state with final decision
        risk_state = state['risk_debate_state'].copy()
        risk_state['judge_decision'] = final_trade_decision
        
        # Save portfolio manager decision to memory
        portfolio_context = f"""
        Company: {state['company_of_interest']}
        Date: {state['trade_date']}
        Trader Proposal: {state['trader_investment_plan'][:200]}...
        Risk Debate: {state['risk_debate_state']['history'][:300]}...
        Final Decision: {final_trade_decision[:200]}...
        """
        risk_manager_memory.add_situations([(portfolio_context, final_trade_decision)])
        
        console.print("[bold magenta]👑 Portfolio Manager Final Decision:[/bold magenta]")
        console.print(Markdown(final_trade_decision))
        
        return {
            **state,
            "risk_debate_state": risk_state,
            "final_trade_decision": final_trade_decision,
            "messages": state["messages"] + result["messages"] if result and "messages" in result else state["messages"],
            "sender": "portfolio_manager"
        }
    
    def consolidation_node(self, state: AgentState) -> AgentState:
        """Final consolidation of all results"""
        console.print("[bold green]📋 Final Consolidation[/bold green]")
        
        summary_parts = [
            f"Complete analysis for {state['company_of_interest']}",
            f"Investment debate rounds: {state['investment_debate_state']['count']}",
            f"Risk debate rounds: {state['risk_debate_state']['count'] // 3}",
            f"Final trade decision: {'Available' if state.get('final_trade_decision') else 'Not available'}",
            f"Trader proposal: {'Available' if state.get('trader_investment_plan') else 'Not available'}"
        ]
        
        summary_message = AIMessage(
            content="Complete trading analysis workflow finished:\n" + "\n".join(summary_parts)
        )
        
        self._display_comprehensive_results(state)
        
        return {
            **state,
            "messages": state["messages"] + [summary_message],
            "sender": "consolidation"
        }
    
    def _display_comprehensive_results(self, state):
        """Display complete results including all phases"""
        console.print("\n" + "="*100)
        console.print("[bold green]🎉 COMPLETE TRADING WORKFLOW RESULTS[/bold green]")
        console.print("="*100)
        
        # Basic info
        console.print(f"[bold cyan]Company:[/bold cyan] {state['company_of_interest']}")
        console.print(f"[bold cyan]Analysis Date:[/bold cyan] {state['trade_date']}")
        console.print(f"[bold cyan]Total Messages:[/bold cyan] {len(state['messages'])}")
        
        # Analysis reports
        console.print(f"\n[bold yellow]📊 Analysis Reports:[/bold yellow]")
        reports = ["market_report", "sentiment_report", "news_report", "fundamentals_report"]
        for report in reports:
            status = f"{len(state.get(report, ''))} chars" if state.get(report) else "❌ Missing"
            console.print(f"  {report.replace('_', ' ').title()}: {status}")
        
        # Investment debate summary
        console.print(f"\n[bold magenta]🥊 Investment Debate:[/bold magenta]")
        console.print(f"  Rounds: {state['investment_debate_state']['count']}")
        console.print(f"  Bull arguments: {len(state['investment_debate_state']['bull_history'].split('Bull Analyst:')) - 1}")
        console.print(f"  Bear arguments: {len(state['investment_debate_state']['bear_history'].split('Bear Analyst:')) - 1}")
        
        # Investment plan
        if state.get('investment_plan'):
            console.print(f"\n[bold purple]👨‍💼 Investment Plan:[/bold purple]")
            console.print(Markdown(state['investment_plan'][:300] + "..."))
        
        # Trading proposal
        if state.get('trader_investment_plan'):
            console.print(f"\n[bold blue]💼 Trading Proposal:[/bold blue]")
            console.print(Markdown(state['trader_investment_plan']))
        
        # Risk debate summary
        console.print(f"\n[bold orange]🛡️ Risk Management Debate:[/bold orange]")
        console.print(f"  Risk rounds: {state['risk_debate_state']['count'] // 3}")
        console.print(f"  Risky arguments: {len(state['risk_debate_state']['risky_history'].split('Risky Analyst:')) - 1}")
        console.print(f"  Safe arguments: {len(state['risk_debate_state']['safe_history'].split('Safe Analyst:')) - 1}")
        console.print(f"  Neutral arguments: {len(state['risk_debate_state']['neutral_history'].split('Neutral Analyst:')) - 1}")
        
        # Final trade decision
        if state.get('final_trade_decision'):
            console.print(f"\n[bold magenta]👑 Final Trade Decision:[/bold magenta]")
            console.print(Markdown(state['final_trade_decision']))
        
        console.print("\n" + "="*100)
    
    def run_analysis(self, ticker: str, trade_date: str = None):
        """Run the complete trading workflow"""
        if trade_date is None:
            trade_date = (datetime.date.today() - datetime.timedelta(days=3)).strftime('%Y-%m-%d')
        
        console.print("[bold blue]🔍 STARTING COMPLETE TRADING WORKFLOW[/bold blue]")
        console.print("="*100)
        
        # Create initial state with RiskDebateState
        initial_state = AgentState({
            "messages": [HumanMessage(content=f"Complete trading analysis for {ticker} on {trade_date}")],
            "company_of_interest": ticker,
            "trade_date": trade_date,
            "market_report": "",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
            "investment_plan": "",
            "trader_investment_plan": "",
            "final_trade_decision": "",
            "sender": "user",
            "investment_debate_state": InvestDebateState({
                'history': '',
                'current_response': '',
                'count': 0,
                'bull_history': '',
                'bear_history': '',
                'judge_decision': ''
            }),
            "risk_debate_state": RiskDebateState({
                'history': '',
                'latest_speaker': '',
                'current_risky_response': '',
                'current_safe_response': '',
                'current_neutral_response': '',
                'count': 0,
                'risky_history': '',
                'safe_history': '',
                'neutral_history': '',
                'judge_decision': ''
            })
        })
        
        # Execute workflow
        final_state = self.graph.invoke(initial_state)
        
        console.print("\n[bold green]🏁 COMPLETE TRADING WORKFLOW FINISHED![/bold green]")
        return final_state

# Usage function
def main():
    """Main function to run complete trading workflow"""
    workflow = CompleteTradingWorkflow()
    
    # Use past date to avoid data issues
    TRADE_DATE = (datetime.date.today() - datetime.timedelta(days=0)).strftime('%Y-%m-%d')
    final_state = workflow.run_analysis("META", TRADE_DATE)
    
    # Serialize state for saving
    serializable_state = final_state.copy()
    if 'messages' in serializable_state:
        serializable_state['messages'] = [
            {
                'type': msg.__class__.__name__,
                'content': msg.content
            } for msg in serializable_state['messages']
        ]

    with open("complete_final_state.json", "w") as f:
        json.dump(serializable_state, f, indent=2)
    
    console.print(f"\n[bold green]📊 WORKFLOW SUMMARY:[/bold green]")
    console.print(f"Company Analyzed: {final_state['company_of_interest']}")
    console.print(f"Analysis Date: {final_state['trade_date']}")
    console.print(f"Investment Plan: {'✅ Available' if final_state.get('investment_plan') else '❌ Missing'}")
    console.print(f"Trading Proposal: {'✅ Available' if final_state.get('trader_investment_plan') else '❌ Missing'}")
    console.print(f"Final Decision: {'✅ Available' if final_state.get('final_trade_decision') else '❌ Missing'}")
    console.print(f"Investment Debate Rounds: {final_state['investment_debate_state']['count']}")
    console.print(f"Risk Debate Rounds: {final_state['risk_debate_state']['count'] // 3}")
    
    return final_state

if __name__ == "__main__":
    final_state = main()