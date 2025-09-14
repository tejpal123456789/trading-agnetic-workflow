from langgraph.graph import StateGraph, START, END
from agent_state import AgentState, InvestDebateState, RiskDebateState
from langchain_core.messages import HumanMessage, AIMessage
from agents.analyst_agent.analyst import create_market_agent, create_social_agent, create_news_agent, create_fundamentals_agent
from tools.toolkit import toolkit
from rich.console import Console
from rich.markdown import Markdown
from llm import quick_thinking_llm
import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

console = Console()

class TradingAnalysisWorkflow:
    def __init__(self):
        self.graph = self._build_graph()
        self.shared_state = None
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("initialization", self.initialization_node)
        workflow.add_node("parallel_analysis", self.parallel_analysis_node)
        workflow.add_node("consolidation", self.consolidation_node)
        
        # Define edges
        workflow.add_edge(START, "initialization")
        workflow.add_edge("initialization", "parallel_analysis")
        workflow.add_edge("parallel_analysis", "consolidation")
        workflow.add_edge("consolidation", END)
        
        return workflow.compile()
    
    def initialization_node(self, state: AgentState) -> AgentState:
        """Initialize the workflow and create agents with shared state"""
        console.print("[bold blue]🚀 Initializing Trading Analysis Workflow[/bold blue]")
        console.print(f"[green]Company:[/green] {state['company_of_interest']}")
        console.print(f"[green]Trade Date:[/green] {state['trade_date']}")
        
        # Store state reference for agents to use
        self.shared_state = state
        
        # Add initialization message
        init_message = AIMessage(
            content=f"Starting comprehensive parallel analysis for {state['company_of_interest']} on {state['trade_date']}"
        )
        
        return {
            **state,
            "messages": state["messages"] + [init_message],
            "sender": "initialization"
        }
    
    def parallel_analysis_node(self, state: AgentState) -> AgentState:
        """Execute all analysts in parallel"""
        console.print("[bold yellow]📊 Running Parallel Analysis with LangGraph...[/bold yellow]")
        
        # Create agents with shared state reference
        market_agent = create_market_agent(quick_thinking_llm, toolkit, state)
        social_agent = create_social_agent(quick_thinking_llm, toolkit, state)
        news_agent = create_news_agent(quick_thinking_llm, toolkit, state)
        fundamentals_agent = create_fundamentals_agent(quick_thinking_llm, toolkit, state)
        
        # Define analysis functions
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
        
        # Execute all analysts in parallel
        console.print("[yellow]🔄 Executing parallel analysis...[/yellow]")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            futures = [
                executor.submit(run_market),
                executor.submit(run_social),
                executor.submit(run_news),
                executor.submit(run_fundamentals)
            ]
            
            # Wait for all to complete
            results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    results.append(result)
                    analyst_names = ["Market", "Social", "News", "Fundamentals"]
                    console.print(f"[green]✅ {analyst_names[i]} Analyst completed[/green]")
                except Exception as e:
                    results.append(None)
                    analyst_names = ["Market", "Social", "News", "Fundamentals"]
                    console.print(f"[red]❌ {analyst_names[i]} Analyst failed: {str(e)}[/red]")
        
        console.print("[bold green]🎉 All parallel analysis completed![/bold green]")
        
        # State was updated by individual agents, return the current state
        return {
            **state,
            "sender": "parallel_analysis"
        }
    
    def consolidation_node(self, state: AgentState) -> AgentState:
        """Consolidate and display all analysis results"""
        console.print("[bold green]📋 Consolidating Analysis Results...[/bold green]")
        
        # Create summary message
        summary_parts = []
        reports = {
            "Market Analysis": state.get('market_report', ''),
            "Social Sentiment": state.get('sentiment_report', ''),
            "News Analysis": state.get('news_report', ''),
            "Fundamentals Analysis": state.get('fundamentals_report', '')
        }
        
        for report_name, report_content in reports.items():
            if report_content and len(report_content) > 0:
                summary_parts.append(f"✅ {report_name}: {len(report_content)} characters")
            else:
                summary_parts.append(f"❌ {report_name}: Not generated")
        
        summary_message = AIMessage(
            content=f"Parallel analysis completed for {state['company_of_interest']}:\n" + "\n".join(summary_parts)
        )
        
        # Display results
        self._display_final_reports(state)
        
        return {
            **state,
            "messages": state["messages"] + [summary_message],
            "sender": "consolidation"
        }
    
    def _display_final_reports(self, state):
        """Display all individual reports"""
        console.print("\n" + "="*80)
        console.print("[bold green]🎉 FINAL LANGGRAPH ANALYSIS RESULTS[/bold green]")
        console.print("="*80)
        
        # Summary
        console.print(f"[bold cyan]Analysis Summary for {state['company_of_interest']}[/bold cyan]")
        console.print(f"[green]Analysis Date:[/green] {state['trade_date']}")
        console.print(f"[green]Total Messages:[/green] {len(state['messages'])}")
        console.print(f"[green]Final Sender:[/green] {state.get('sender', 'Unknown')}")
        
        # Individual Reports
        reports = {
            "📈 Market Technical Analysis": state.get("market_report", ""),
            "💬 Social Media Sentiment": state.get("sentiment_report", ""), 
            "📰 News Impact Analysis": state.get("news_report", ""),
            "🏗️ Fundamental Analysis": state.get("fundamentals_report", "")
        }
        
        console.print(f"\n[bold yellow]📊 Report Status:[/bold yellow]")
        for report_name, report_content in reports.items():
            status = f"{len(report_content)} characters" if report_content else "❌ Not generated"
            console.print(f"  {report_name}: {status}")
        
        console.print(f"\n[bold magenta]📋 DETAILED REPORTS:[/bold magenta]")
        
        for report_name, report_content in reports.items():
            if report_content and len(report_content) > 0:
                console.print(f"\n{'-'*60}")
                console.print(f"[bold cyan]{report_name}[/bold cyan]")
                console.print(f"{'-'*60}")
                console.print(Markdown(report_content))
            else:
                console.print(f"\n{'-'*60}")
                console.print(f"[bold red]{report_name}[/bold red]")
                console.print(f"{'-'*60}")
                console.print("[red]❌ Report not generated or empty[/red]")
    
    def run_analysis(self, ticker: str, trade_date: str = None):
        """Run the complete LangGraph workflow"""
        if trade_date is None:
            trade_date = (datetime.date.today() - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
        
        console.print("[bold blue]🔍 STARTING LANGGRAPH PARALLEL WORKFLOW[/bold blue]")
        console.print("="*80)
        
        # Create initial state
        initial_state = AgentState({
            "messages": [HumanMessage(content=f"Analyze {ticker} for trading on {trade_date}")],
            "company_of_interest": ticker,
            "trade_date": trade_date,
            "market_report": "",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
            "sender": "user",
            "investment_debate_state": InvestDebateState({
                'history': '', 'current_response': '', 'count': 0, 
                'bull_history': '', 'bear_history': '', 'judge_decision': ''
            }),
            "risk_debate_state": RiskDebateState({
                'history': '', 'latest_speaker': '', 'current_risky_response': '', 
                'current_safe_response': '', 'current_neutral_response': '', 'count': 0, 
                'risky_history': '', 'safe_history': '', 'neutral_history': '', 'judge_decision': ''
            })
        })
        
        # Execute the LangGraph workflow
        final_state = self.graph.invoke(initial_state)
        
        console.print("\n[bold green]🏁 LANGGRAPH WORKFLOW COMPLETED![/bold green]")
        console.print(f"Final state keys: {list(final_state.keys())}")
        
        return final_state
    
    async def run_analysis_async(self, ticker: str, trade_date: str = None):
        """Async version of the workflow"""
        if trade_date is None:
            trade_date = (datetime.date.today() - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
        
        # Create initial state
        initial_state = AgentState({
            "messages": [HumanMessage(content=f"Analyze {ticker} for trading on {trade_date}")],
            "company_of_interest": ticker,
            "trade_date": trade_date,
            "market_report": "",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
            "sender": "user",
            "investment_debate_state": InvestDebateState({
                'history': '', 'current_response': '', 'count': 0, 
                'bull_history': '', 'bear_history': '', 'judge_decision': ''
            }),
            "risk_debate_state": RiskDebateState({
                'history': '', 'latest_speaker': '', 'current_risky_response': '', 
                'current_safe_response': '', 'current_neutral_response': '', 'count': 0, 
                'risky_history': '', 'safe_history': '', 'neutral_history': '', 'judge_decision': ''
            })
        })
        
        # Execute async
        final_state = await self.graph.ainvoke(initial_state)
        return final_state

# Usage functions
def main():
    """Main function to run the LangGraph workflow"""
    workflow = TradingAnalysisWorkflow()
    final_state = workflow.run_analysis("GILLETTE")
    
    # Access individual reports
    console.print(f"\n[bold green]📊 WORKFLOW RESULTS SUMMARY:[/bold green]")
    console.print(f"Market report: {len(final_state.get('market_report', ''))} chars")
    console.print(f"Sentiment report: {len(final_state.get('sentiment_report', ''))} chars")
    console.print(f"News report: {len(final_state.get('news_report', ''))} chars")
    console.print(f"Fundamentals report: {len(final_state.get('fundamentals_report', ''))} chars")
    
    return final_state

async def main_async():
    """Async version of main"""
    workflow = TradingAnalysisWorkflow()
    final_state = await workflow.run_analysis_async("NVDA")
    return final_state

def run_async_workflow():
    """Run the async version"""
    return asyncio.run(main_async())

if __name__ == "__main__":
    # Run LangGraph workflow
    final_state = main()
    
    # Or run async version
    # final_state = run_async_workflow()