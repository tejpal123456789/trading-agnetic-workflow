# File: reflection_learning_system.py

from typing import Dict, Any, Callable, Optional
from rich.console import Console
from rich.markdown import Markdown
from memory.longterm_memory import bull_memory, bear_memory, trader_memory, risk_manager_memory, invest_judge_memory
from llm import quick_thinking_llm, deep_thinking_llm
import json
import datetime

console = Console()

class SignalProcessor:
    """Extract clean BUY/SELL/HOLD signals from natural language decisions"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def process_signal(self, full_signal: str) -> str:
        """Extract clean trading signal from decision text"""
        try:
            messages = [
                {
                    "role": "system", 
                    "content": "You are an assistant designed to extract the final investment decision: SELL, BUY, or HOLD from a financial report. Respond with only the single-word decision."
                },
                {
                    "role": "user", 
                    "content": full_signal
                }
            ]
            
            result = self.llm.invoke(messages).content.strip().upper()
            
            # Validate signal
            if result in ["BUY", "SELL", "HOLD"]:
                return result
            elif "BUY" in result:
                return "BUY"
            elif "SELL" in result:
                return "SELL"
            elif "HOLD" in result:
                return "HOLD"
            else:
                return "ERROR_UNPARSABLE_SIGNAL"
                
        except Exception as e:
            console.print(f"[red]Error processing signal: {str(e)}[/red]")
            return "ERROR_PROCESSING_FAILED"

class Reflector:
    """Core learning engine that analyzes outcomes and updates agent memories"""
    
    def __init__(self, llm):
        self.llm = llm
        self.reflection_prompt = """You are an expert financial analyst conducting a post-trade analysis. 
        Review the trading decision, market context, and actual outcome to extract learning insights.
        
        Your analysis should:
        1. Determine if the decision was correct based on the outcome
        2. Identify the 2-3 most critical factors that contributed to success/failure
        3. Formulate a concise, actionable lesson for future similar situations
        4. Rate the decision quality on a scale of 1-10
        
        Market Context & Analysis: {situation}
        Original Decision: {decision}
        Actual Outcome: {outcome}
        Financial Result: {returns_losses}
        
        Provide a structured reflection with specific lessons learned."""
    
    def reflect(self, 
               final_state: Dict[str, Any], 
               returns_losses: float, 
               outcome_description: str,
               memory, 
               component_key_func: Callable,
               agent_name: str) -> str:
        """
        Conduct reflection for a specific agent
        
        Args:
            final_state: Complete workflow state
            returns_losses: Actual financial outcome (positive for profit, negative for loss)
            outcome_description: Text description of what happened
            memory: Memory instance for this agent
            component_key_func: Function to extract relevant text for this agent
            agent_name: Name of the agent for logging
            
        Returns:
            Generated reflection text
        """
        try:
            # Extract relevant context for this agent
            agent_content = component_key_func(final_state)
            
            # Build comprehensive situation context
            situation = f"""
            COMPANY: {final_state.get('company_of_interest', 'Unknown')}
            DATE: {final_state.get('trade_date', 'Unknown')}
            
            ANALYSIS REPORTS:
            Market Report: {final_state.get('market_report', 'N/A')[:200]}...
            Sentiment Report: {final_state.get('sentiment_report', 'N/A')[:200]}...
            News Report: {final_state.get('news_report', 'N/A')[:200]}...
            Fundamentals Report: {final_state.get('fundamentals_report', 'N/A')[:200]}...
            
            {agent_name.upper()} SPECIFIC CONTENT:
            {agent_content[:500]}...
            """
            
            prompt = self.reflection_prompt.format(
                situation=situation,
                decision=final_state.get('final_trade_decision', 'No decision recorded'),
                outcome=outcome_description,
                returns_losses=f"${returns_losses:,.2f}" if returns_losses != 0 else "Break-even"
            )
            
            # Generate reflection
            reflection_result = self.llm.invoke(prompt).content
            
            # Store in memory
            memory.add_situations([(situation, reflection_result)])
            
            console.print(f"[green]Reflection completed for {agent_name}[/green]")
            return reflection_result
            
        except Exception as e:
            console.print(f"[red]Error reflecting for {agent_name}: {str(e)}[/red]")
            return f"Error during reflection: {str(e)}"

class TradingReflectionSystem:
    """Complete reflection system for trading workflow"""
    
    def __init__(self, llm=None):
        self.llm = llm or deep_thinking_llm
        self.signal_processor = SignalProcessor(quick_thinking_llm)
        self.reflector = Reflector(self.llm)
    
    def process_trading_outcome(self, 
                              final_state: Dict[str, Any],
                              actual_returns: float,
                              outcome_description: str,
                              save_reflection: bool = True) -> Dict[str, Any]:
        """
        Complete post-trade analysis and learning
        
        Args:
            final_state: Final state from workflow
            actual_returns: Actual profit/loss in dollars
            outcome_description: Description of what happened
            save_reflection: Whether to save reflection results
            
        Returns:
            Dictionary with signal, reflections, and analysis
        """
        console.print("[bold blue]Starting Post-Trade Reflection Process[/bold blue]")
        console.print("=" * 80)
        
        # Step 1: Extract clean signal
        raw_decision = final_state.get('final_trade_decision', '')
        clean_signal = self.signal_processor.process_signal(raw_decision)
        
        console.print(f"[cyan]Extracted Trading Signal:[/cyan] {clean_signal}")
        console.print(f"[cyan]Actual Outcome:[/cyan] ${actual_returns:,.2f}")
        
        # Step 2: Determine if decision was correct
        decision_correctness = self._evaluate_decision_correctness(clean_signal, actual_returns)
        
        # Step 3: Run reflections for each agent
        reflections = {}
        
        # Reflection configurations
        reflection_configs = [
            {
                'name': 'Bull Researcher',
                'memory': bull_memory,
                'extractor': lambda s: s.get('investment_debate_state', {}).get('bull_history', ''),
            },
            {
                'name': 'Bear Researcher', 
                'memory': bear_memory,
                'extractor': lambda s: s.get('investment_debate_state', {}).get('bear_history', ''),
            },
            {
                'name': 'Research Manager',
                'memory': invest_judge_memory,
                'extractor': lambda s: s.get('investment_plan', ''),
            },
            {
                'name': 'Trader',
                'memory': trader_memory,
                'extractor': lambda s: s.get('trader_investment_plan', ''),
            },
            {
                'name': 'Risk Manager',
                'memory': risk_manager_memory,
                'extractor': lambda s: s.get('final_trade_decision', ''),
            }
        ]
        
        for config in reflection_configs:
            console.print(f"[yellow]Reflecting for {config['name']}...[/yellow]")
            
            reflection_text = self.reflector.reflect(
                final_state=final_state,
                returns_losses=actual_returns,
                outcome_description=outcome_description,
                memory=config['memory'],
                component_key_func=config['extractor'],
                agent_name=config['name']
            )
            
            reflections[config['name']] = reflection_text
        
        # Step 4: Generate overall system reflection
        system_reflection = self._generate_system_reflection(
            final_state, actual_returns, outcome_description, clean_signal, decision_correctness
        )
        
        # Step 5: Compile results
        reflection_results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'company': final_state.get('company_of_interest', 'Unknown'),
            'trade_date': final_state.get('trade_date', 'Unknown'),
            'extracted_signal': clean_signal,
            'actual_returns': actual_returns,
            'outcome_description': outcome_description,
            'decision_correctness': decision_correctness,
            'agent_reflections': reflections,
            'system_reflection': system_reflection,
            'raw_decision': raw_decision
        }
        
        # Step 6: Display results
        self._display_reflection_results(reflection_results)
        
        # Step 7: Save if requested
        if save_reflection:
            self._save_reflection_results(reflection_results)
        
        console.print("[bold green]Post-Trade Reflection Completed[/bold green]")
        return reflection_results
    
    def _evaluate_decision_correctness(self, signal: str, returns: float) -> str:
        """Determine if the trading decision was correct"""
        if signal == "BUY" and returns > 0:
            return "CORRECT - Bought and profited"
        elif signal == "SELL" and returns < 0:
            return "CORRECT - Sold and avoided loss"
        elif signal == "HOLD" and abs(returns) < 100:  # Small movements
            return "CORRECT - Held during sideways movement"
        elif signal == "BUY" and returns < 0:
            return "INCORRECT - Bought but lost money"
        elif signal == "SELL" and returns > 0:
            return "INCORRECT - Sold but missed gains"
        elif signal == "HOLD" and abs(returns) > 500:
            return "INCORRECT - Held during significant movement"
        else:
            return "NEUTRAL - Outcome unclear or break-even"
    
    def _generate_system_reflection(self, final_state, returns, outcome, signal, correctness):
        """Generate overall system-level reflection"""
        system_prompt = f"""Analyze the overall system performance for this trading decision:
        
        Decision: {signal}
        Outcome: {correctness}
        Returns: ${returns:,.2f}
        
        Provide insights on:
        1. How well the multi-agent system worked together
        2. Which components contributed most to success/failure
        3. System-level improvements for future trades
        4. Overall confidence in the decision-making process
        """
        
        try:
            return self.llm.invoke(system_prompt).content
        except:
            return "System reflection generation failed"
    
    def _display_reflection_results(self, results):
        """Display reflection results in formatted output"""
        console.print("\n" + "=" * 100)
        console.print("[bold green]POST-TRADE REFLECTION RESULTS[/bold green]")
        console.print("=" * 100)
        
        # Basic info
        console.print(f"[bold cyan]Company:[/bold cyan] {results['company']}")
        console.print(f"[bold cyan]Trade Date:[/bold cyan] {results['trade_date']}")
        console.print(f"[bold cyan]Signal:[/bold cyan] {results['extracted_signal']}")
        console.print(f"[bold cyan]Returns:[/bold cyan] ${results['actual_returns']:,.2f}")
        console.print(f"[bold cyan]Decision:[/bold cyan] {results['decision_correctness']}")
        
        # Agent reflections
        console.print(f"\n[bold yellow]AGENT REFLECTIONS:[/bold yellow]")
        for agent_name, reflection in results['agent_reflections'].items():
            console.print(f"\n[bold]{agent_name}:[/bold]")
            console.print(Markdown(reflection[:300] + "..." if len(reflection) > 300 else reflection))
        
        # System reflection
        console.print(f"\n[bold purple]SYSTEM REFLECTION:[/bold purple]")
        console.print(Markdown(results['system_reflection']))
    
    def _save_reflection_results(self, results):
        """Save reflection results to file"""
        filename = f"reflection_{results['company']}_{results['trade_date']}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            console.print(f"[blue]Reflection results saved to: {filename}[/blue]")
        except Exception as e:
            console.print(f"[red]Error saving reflection: {str(e)}[/red]")

# Convenience functions
def simulate_trading_outcome(final_state: Dict[str, Any], 
                           hypothetical_returns: float = None,
                           outcome_description: str = None) -> Dict[str, Any]:
    """Simulate a trading outcome for testing/demo purposes"""
    
    if hypothetical_returns is None:
        # Generate random outcome for demo
        import random
        hypothetical_returns = random.uniform(-1000, 2000)
    
    if outcome_description is None:
        if hypothetical_returns > 0:
            outcome_description = f"Stock moved favorably, generating ${hypothetical_returns:,.2f} profit"
        else:
            outcome_description = f"Stock moved unfavorably, resulting in ${abs(hypothetical_returns):,.2f} loss"
    
    # Create reflection system and process outcome
    reflection_system = TradingReflectionSystem()
    return reflection_system.process_trading_outcome(
        final_state, hypothetical_returns, outcome_description
    )

def quick_reflection(final_state: Dict[str, Any], 
                    returns: float, 
                    description: str) -> None:
    """Quick reflection without detailed setup"""
    system = TradingReflectionSystem()
    system.process_trading_outcome(final_state, returns, description)