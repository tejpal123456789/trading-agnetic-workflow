from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from agent_state import AgentState, InvestDebateState, RiskDebateState
from langchain_core.messages import HumanMessage, AIMessage
from agents.analyst_agent.analyst import create_market_agent, create_social_agent, create_news_agent, create_fundamentals_agent
from tools.toolkit import toolkit
from rich.console import Console
from rich.markdown import Markdown
from llm import quick_thinking_llm, deep_thinking_llm
from memory.longterm_memory import bull_memory, bear_memory, invest_judge_memory
import datetime
from concurrent.futures import ThreadPoolExecutor
from agents.bull_vs_bear.bull import create_bull_agent
from agents.bull_vs_bear.bear import create_bear_agent
from agents.research_agent import create_research_manager_agent
import json
console = Console()

class ExtendedTradingWorkflow:
    def __init__(self):
        self.graph = self._build_graph()
        self.shared_state = None
    
    def _build_graph(self):
        """Build the extended LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("initialization", self.initialization_node)
        workflow.add_node("parallel_analysis", self.parallel_analysis_node)
        workflow.add_node("bull_researcher", self.bull_researcher_node)
        workflow.add_node("bear_researcher", self.bear_researcher_node)
        workflow.add_node("research_manager", self.research_manager_node)
        workflow.add_node("consolidation", self.consolidation_node)
        
        # Define edges
        workflow.add_edge(START, "initialization")
        workflow.add_edge("initialization", "parallel_analysis")
        
        # After parallel analysis, start the debate cycle
        workflow.add_edge("parallel_analysis", "bull_researcher")
        workflow.add_edge("bull_researcher", "bear_researcher")
        
        # Add conditional logic for debate rounds
        workflow.add_conditional_edges(
            "bear_researcher",
            self.should_continue_debate,
            {
                "continue": "bull_researcher",  # Continue debate
                "end": "research_manager"       # End debate, go to manager
            }
        )
        
        workflow.add_edge("research_manager", "consolidation")
        workflow.add_edge("consolidation", END)
        
        return workflow.compile()
    
    def initialization_node(self, state: AgentState) -> AgentState:
        """Initialize the workflow"""
        console.print("[bold blue]🚀 Initializing Extended Trading Analysis Workflow[/bold blue]")
        console.print(f"[green]Company:[/green] {state['company_of_interest']}")
        console.print(f"[green]Trade Date:[/green] {state['trade_date']}")
        
        init_message = AIMessage(
            content=f"Starting comprehensive analysis and debate for {state['company_of_interest']} on {state['trade_date']}"
        )
        
        return {
            **state,
            "messages": state["messages"] + [init_message],
            "sender": "initialization"
        }
    
    def parallel_analysis_node(self, state: AgentState) -> AgentState:
        """Execute all analysts in parallel (same as before)"""
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
        
        # Create bull agent with state context
        bull_agent = create_bull_agent(quick_thinking_llm, toolkit, state)
        
        # Simple prompt since context is already in system prompt
        prompt = f"Present your strongest bull case for {state['company_of_interest']}. Make compelling arguments for why this stock should be bought."
        
        result = bull_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        
        # Extract the argument
        bull_argument = ""
        if result and "messages" in result and result["messages"]:
            final_message = result["messages"][-1]
            bull_argument = f"Bull Analyst: {final_message.content}"
        
        # Update debate state
        debate_state = state['investment_debate_state'].copy()
        debate_state['history'] += "\n" + bull_argument
        debate_state['bull_history'] += "\n" + bull_argument
        debate_state['current_response'] = bull_argument
        debate_state['count'] += 1
        
        # ADD MEMORY UPDATE (this was missing)
        # situation_context = f"{state['market_report'][:200]}... Company: {state['company_of_interest']}"
        # bull_memory.add_situations([(situation_context, bull_argument)])
        
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
            
            # Create bear agent with state context
            bear_agent = create_bear_agent(quick_thinking_llm, toolkit, state)
            
            # Simple prompt since context is already in system prompt
            prompt = f"Present your strongest bear case for {state['company_of_interest']}. Make compelling arguments for why this stock should be avoided or sold."
            
            result = bear_agent.invoke({"messages": [HumanMessage(content=prompt)]})
            
            # Extract the argument
            bear_argument = ""
            if result and "messages" in result and result["messages"]:
                final_message = result["messages"][-1]
                bear_argument = f"Bear Analyst: {final_message.content}"
            
            # Update debate state
            debate_state = state['investment_debate_state'].copy()
            debate_state['history'] += "\n" + bear_argument
            debate_state['bear_history'] += "\n" + bear_argument
            debate_state['current_response'] = bear_argument
            
            # ADD MEMORY UPDATE (this was missing)
            situation_context = f"{state['market_report'][:200]}... Company: {state['company_of_interest']}"
            bear_memory.add_situations([(situation_context, bear_argument)])
            
            console.print("[red]🐻 Bear's Rebuttal:[/red]")
            console.print(Markdown(bear_argument.replace('Bear Analyst: ', '')))
            
            return {
                **state,
                "investment_debate_state": debate_state,
                "messages": state["messages"] + result["messages"] if result and "messages" in result else state["messages"],
                "sender": "bear_researcher"}
    
    def should_continue_debate(self, state: AgentState) -> str:
        """Decide whether to continue the debate or end it"""
        max_rounds = 3  # Configure maximum debate rounds
        current_round = state['investment_debate_state']['count']
        
        if current_round >= max_rounds:
            console.print(f"[yellow]📊 Debate completed after {current_round} rounds[/yellow]")
            return "end"
        else:
            console.print(f"[yellow]🔄 Continuing debate - Round {current_round + 1}[/yellow]")
            return "continue"
    
    def research_manager_node(self, state: AgentState) -> AgentState:
        """Research manager node using create_react_agent"""
        console.print("[bold purple]👨‍💼 Research Manager - Making Final Investment Decision[/bold purple]")
        
        # Create research manager agent with full context
        manager_agent = create_research_manager_agent(deep_thinking_llm, toolkit, state)
        
        # Focused prompt for final decision (context already in system prompt)
        prompt = f"""As Research Manager, evaluate all information and make your final investment decision for {state['company_of_interest']}.
        
        Provide:
        1. Executive Summary of key bull and bear points
        2. Your final decision: BUY, SELL, or HOLD (be definitive)
        3. Detailed investment plan with specific actions and timeline
        4. Risk assessment and mitigation strategies
        5. Price targets and exit conditions if applicable
        
        Make this decision actionable for traders and investors."""
        
        result = manager_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        
        # Extract investment plan
        investment_plan = ""
        if result and "messages" in result and result["messages"]:
            final_message = result["messages"][-1]
            investment_plan = final_message.content
        
        # Update debate state with manager decision
        debate_state = state['investment_debate_state'].copy()
        debate_state['judge_decision'] = investment_plan
        
        # ADD MEMORY UPDATE - Save this decision for future reference
        decision_context = f"""
        Company: {state['company_of_interest']}
        Date: {state['trade_date']}
        Market Report Summary: {state['market_report'][:300]}...
        Debate Summary: Bull vs Bear had {state['investment_debate_state']['count']} rounds
        Final Decision: {investment_plan[:200]}...
        """
        invest_judge_memory.add_situations([(decision_context, investment_plan)])
        
        console.print("[bold purple]👨‍💼 Research Manager Final Decision:[/bold purple]")
        console.print(Markdown(investment_plan))
        
        return {
            **state,
            "investment_debate_state": debate_state,
            "investment_plan": investment_plan,
            "messages": state["messages"] + result["messages"] if result and "messages" in result else state["messages"],
            "sender": "research_manager"
        }
    
    def consolidation_node(self, state: AgentState) -> AgentState:
        """Final consolidation of all results"""
        console.print("[bold green]📋 Final Consolidation[/bold green]")
        
        # Create comprehensive summary
        summary_parts = [
            f"Analysis completed for {state['company_of_interest']}",
            f"Debate rounds: {state['investment_debate_state']['count']}",
            f"Final decision available: {'Yes' if state.get('investment_plan') else 'No'}",
        ]
        
        summary_message = AIMessage(
            content="Extended trading analysis workflow completed:\n" + "\n".join(summary_parts)
        )
        
        # Display final comprehensive results
        self._display_comprehensive_results(state)
        
        return {
            **state,
            "messages": state["messages"] + [summary_message],
            "sender": "consolidation"
        }
    
    def _display_comprehensive_results(self, state):
        """Display all results including debate and final decision"""
        console.print("\n" + "="*100)
        console.print("[bold green]🎉 COMPREHENSIVE TRADING ANALYSIS RESULTS[/bold green]")
        console.print("="*100)
        
        # Basic info
        console.print(f"[bold cyan]Company:[/bold cyan] {state['company_of_interest']}")
        console.print(f"[bold cyan]Analysis Date:[/bold cyan] {state['trade_date']}")
        console.print(f"[bold cyan]Total Messages:[/bold cyan] {len(state['messages'])}")
        
        # Analysis reports status
        console.print(f"\n[bold yellow]📊 Analysis Reports:[/bold yellow]")
        reports = ["market_report", "sentiment_report", "news_report", "fundamentals_report"]
        for report in reports:
            status = f"{len(state.get(report, ''))} chars" if state.get(report) else "❌ Missing"
            console.print(f"  {report.replace('_', ' ').title()}: {status}")
        
        # Debate summary
        console.print(f"\n[bold magenta]🥊 Debate Summary:[/bold magenta]")
        console.print(f"  Rounds completed: {state['investment_debate_state']['count']}")
        console.print(f"  Bull arguments: {len(state['investment_debate_state']['bull_history'].split('Bull Analyst:')) - 1}")
        console.print(f"  Bear arguments: {len(state['investment_debate_state']['bear_history'].split('Bear Analyst:')) - 1}")
        
        # Final decision
        if state.get('investment_plan'):
            console.print(f"\n[bold purple]👨‍💼 Final Investment Decision:[/bold purple]")
            console.print(Markdown(state['investment_plan']))
        
        console.print("\n" + "="*100)
    
    def run_analysis(self, ticker: str, trade_date: str = None):
        """Run the complete extended workflow"""
        if trade_date is None:
            trade_date = (datetime.date.today() - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
        
        console.print("[bold blue]🔍 STARTING EXTENDED LANGGRAPH WORKFLOW[/bold blue]")
        console.print("="*100)
        
        # Create initial state
        initial_state = AgentState({
            "messages": [HumanMessage(content=f"Comprehensive analysis for {ticker} on {trade_date}")],
            "company_of_interest": ticker,
            "trade_date": trade_date,
            "market_report": "",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
            "investment_plan": "",
            "sender": "user",
            "investment_debate_state": InvestDebateState({
                'history': '',
                'current_response': '',
                'count': 2,
                'bull_history': '',
                'bear_history': '',
                'judge_decision': ''
            })
        })
        
        # Execute workflow
        final_state = self.graph.invoke(initial_state)
        
        console.print("\n[bold green]🏁 EXTENDED WORKFLOW COMPLETED![/bold green]")
        return final_state

# Usage
def main():
    """Main function to run extended workflow"""
    workflow = ExtendedTradingWorkflow()
    TRADE_DATE = (datetime.date.today() - datetime.timedelta(days=0)).strftime('%Y-%m-%d')
    final_state = workflow.run_analysis("META", TRADE_DATE)
    serializable_state = final_state.copy()
    if 'messages' in serializable_state:
        serializable_state['messages'] = [
            {
                'type': msg.__class__.__name__,
                'content': msg.content
            } for msg in serializable_state['messages']
        ]

    with open("final_state.json", "w") as f:
        json.dump(serializable_state, f, indent=2)
    # with open("final_state.json", "w") as f:
    #     json.dump(final_state, f)
    
    console.print(f"\n[bold green]📊 FINAL RESULTS:[/bold green]")
    console.print(f"Investment Plan: {'Available' if final_state.get('investment_plan') else 'Not generated'}")
    console.print(f"Debate Rounds: {final_state['investment_debate_state']['count']}")
    
    return final_state

if __name__ == "__main__":
    final_state = main()