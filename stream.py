# File: langsmith_streaming_wrapper.py

import os
import time
from rich.console import Console
from rich.markdown import Markdown
import json
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# LangSmith configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "complete-trading-workflow"

console = Console()

class LangSmithStreamingWrapper:
    """Wrapper class to add LangSmith streaming capabilities to any LangGraph workflow"""
    
    def __init__(self, workflow_instance):
        """
        Initialize the streaming wrapper
        
        Args:
            workflow_instance: Any class instance that has a compiled LangGraph (.graph attribute)
        """
        self.workflow = workflow_instance
        self.execution_stats = {}
        self.session_id = None
    
    def stream_workflow(self, 
                       initial_state: Dict, 
                       config: Optional[Dict] = None,
                       save_results: bool = True,
                       filename_prefix: str = "streamed_results") -> Any:
        """
        Stream the workflow execution with LangSmith tracing
        
        Args:
            initial_state: Initial state for the workflow
            config: Configuration for the workflow execution
            save_results: Whether to save results to JSON
            filename_prefix: Prefix for saved files
            
        Returns:
            Final state from workflow execution
        """
        
        # Set up configuration
        if config is None:
            config = {
                "recursion_limit": 50,
                "configurable": {
                    "session_id": f"workflow-session-{int(time.time())}",
                    "user_id": "streaming-system",
                }
            }
        
        self.session_id = config.get("configurable", {}).get("session_id", "unknown")
        
        console.print("[bold blue]Starting LangSmith Streaming Execution[/bold blue]")
        console.print(f"[cyan]Session ID:[/cyan] {self.session_id}")
        console.print("=" * 80)
        
        final_state = None
        node_execution_order = []
        node_timings = {}
        
        print("\n--- Invoking Graph Stream ---")
        
        try:
            workflow_start_time = time.time()
            
            for chunk in self.workflow.graph.stream(initial_state, config=config):
                # Extract node information
                node_name = list(chunk.keys())[0]
                node_start_time = time.time()
                
                # Display node execution
                print(f"Executing Node: {node_name}")
                console.print(f"[bold yellow]Node:[/bold yellow] {node_name}")
                
                # Track execution
                node_execution_order.append(node_name)
                final_state = chunk[node_name]
                
                # Calculate timing
                node_end_time = time.time()
                node_duration = node_end_time - node_start_time
                node_timings[node_name] = node_duration
                
                console.print(f"[green]Node completed in {node_duration:.3f}s[/green]")
                
                # Optional: Brief pause for visual streaming effect
                time.sleep(0.05)
            
            total_execution_time = time.time() - workflow_start_time
            
            print("\n--- Graph Stream Finished ---")
            console.print("[bold green]Workflow streaming completed successfully[/bold green]")
            
            # Store execution statistics
            self.execution_stats = {
                "total_time": total_execution_time,
                "node_count": len(node_execution_order),
                "node_order": node_execution_order,
                "node_timings": node_timings,
                "session_id": self.session_id,
                "average_node_time": total_execution_time / len(node_execution_order) if node_execution_order else 0
            }
            
            # Display execution summary
            self._display_execution_summary()
            
            # Save results if requested
            if save_results and final_state:
                self._save_results(final_state, filename_prefix)
            
            return final_state
            
        except Exception as e:
            console.print(f"[red]Error during streaming execution: {str(e)}[/red]")
            print(f"Streaming Error: {e}")
            return None
    
    def _display_execution_summary(self):
        """Display detailed execution summary"""
        stats = self.execution_stats
        
        console.print("\n" + "=" * 80)
        console.print("[bold blue]EXECUTION SUMMARY[/bold blue]")
        console.print("=" * 80)
        
        console.print(f"[cyan]Session ID:[/cyan] {stats['session_id']}")
        console.print(f"[cyan]Total Execution Time:[/cyan] {stats['total_time']:.2f}s")
        console.print(f"[cyan]Nodes Executed:[/cyan] {stats['node_count']}")
        console.print(f"[cyan]Average Time per Node:[/cyan] {stats['average_node_time']:.2f}s")
        
        # Node execution order
        console.print(f"\n[yellow]Execution Order:[/yellow]")
        for i, node in enumerate(stats['node_order'], 1):
            timing = stats['node_timings'].get(node, 0)
            console.print(f"  {i:2d}. {node:<20} ({timing:.3f}s)")
        
        # Performance analysis
        if stats['node_timings']:
            sorted_timings = sorted(stats['node_timings'].items(), key=lambda x: x[1], reverse=True)
            
            console.print(f"\n[red]Slowest Nodes (Top 3):[/red]")
            for node, timing in sorted_timings[:3]:
                percentage = (timing / stats['total_time']) * 100
                console.print(f"  {node:<20} {timing:.3f}s ({percentage:.1f}%)")
            
            console.print(f"\n[green]Fastest Nodes (Top 3):[/green]")
            for node, timing in sorted_timings[-3:]:
                percentage = (timing / stats['total_time']) * 100
                console.print(f"  {node:<20} {timing:.3f}s ({percentage:.1f}%)")
    
    def _save_results(self, final_state: Any, filename_prefix: str):
        """Save execution results and statistics"""
        try:
            # Save final state
            serializable_state = final_state.copy() if hasattr(final_state, 'copy') else dict(final_state)
            
            # Handle message serialization if present
            if 'messages' in serializable_state and serializable_state['messages']:
                serializable_state['messages'] = [
                    {
                        'type': msg.__class__.__name__,
                        'content': getattr(msg, 'content', str(msg))
                    } for msg in serializable_state['messages']
                ]
            
            # Save final state
            state_filename = f"{filename_prefix}_final_state.json"
            with open(state_filename, "w") as f:
                json.dump(serializable_state, f, indent=2, default=str)
            
            # Save execution statistics
            stats_filename = f"{filename_prefix}_execution_stats.json"
            with open(stats_filename, "w") as f:
                json.dump(self.execution_stats, f, indent=2)
            
            console.print(f"[blue]Results saved to:[/blue]")
            console.print(f"  - State: {state_filename}")
            console.print(f"  - Stats: {stats_filename}")
            
        except Exception as e:
            console.print(f"[red]Error saving results: {str(e)}[/red]")
    
    def get_langsmith_url(self) -> str:
        """Get LangSmith dashboard URL for this session"""
        base_url = "https://smith.langchain.com/projects/complete-trading-workflow"
        if self.session_id and self.session_id != "unknown":
            return f"{base_url}/sessions/{self.session_id}"
        return base_url
    
    def display_langsmith_info(self):
        """Display LangSmith dashboard information"""
        console.print(f"\n[bold blue]LangSmith Dashboard:[/bold blue]")
        console.print(f"[blue]Project:[/blue] complete-trading-workflow")
        console.print(f"[blue]Session:[/blue] {self.session_id}")
        console.print(f"[blue]URL:[/blue] {self.get_langsmith_url()}")
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        return self.execution_stats.copy()

# Convenience function for quick usage
def stream_langraph_workflow(workflow_instance, 
                           initial_state: Dict,
                           config: Optional[Dict] = None,
                           **kwargs) -> Any:
    """
    Convenience function to stream any LangGraph workflow
    
    Args:
        workflow_instance: Instance with .graph attribute
        initial_state: Initial state for execution
        config: Optional configuration
        **kwargs: Additional arguments for streaming
        
    Returns:
        Final state from execution
    """
    wrapper = LangSmithStreamingWrapper(workflow_instance)
    return wrapper.stream_workflow(initial_state, config, **kwargs)