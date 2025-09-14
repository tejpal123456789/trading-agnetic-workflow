from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
import datetime
from rich.console import Console
from rich.markdown import Markdown

from dotenv import load_dotenv

load_dotenv()

def create_analyst_node(llm, toolkit, system_message, tools, output_field):
    """
    Creates a node for an analyst agent.
    
    Args:
        llm: The language model instance to be used by the agent.
        toolkit: The collection of tools available to the agent.
        system_message: The specific instructions defining the agent's role and goals.
        tools: A list of specific tools from the toolkit that this agent is allowed to use.
        output_field: The key in the AgentState where this agent's final report will be stored.
    
    Returns:
        Function: A node function that can be used in a LangGraph workflow.
    """
    # Define the prompt template for the analyst agent
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful AI assistant, collaborating with other assistants. "
            "Use the provided tools to progress towards answering the question. "
            "If you are unable to fully answer, that's OK; another assistant with different tools "
            "will help where you left off. Execute what you can to make progress. "
            "You have access to the following tools: {tool_names}.\n{system_message} "
            "For your reference, the current date is {current_date}. The company we want to look at is {ticker}"
        ),
        # MessagesPlaceholder allows us to pass in the conversation history
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Partially fill the prompt with the specific system message and tool names for this analyst
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    
    # Bind the specified tools to the LLM. This tells the LLM which functions it can call
    chain = prompt | llm.bind_tools(tools)
    
    def analyst_node(state):
        """
        The actual function that will be executed as a node in the graph.
        
        Args:
            state: The current state of the agent workflow.
            
        Returns:
            dict: Updated state with messages and report.
        """
        # Invoke the chain with all required variables from the state
        result = chain.invoke({
            "messages": state["messages"],
            "current_date": state["trade_date"],
            "ticker": state["company_of_interest"]
        })
        
        report = ""
        
        # If the LLM didn't call a tool, it means it has generated the final report
        if not hasattr(result, 'tool_calls') or not result.tool_calls:
            report = result.content
        
        # Return the LLM's response and the final report to update the state
        return {"messages": [result], output_field: report}
    
    return analyst_node


def run_analyst(analyst_node, initial_state, toolkit):
    """
    Helper function to run a single analyst's ReAct loop with proper message handling.
    
    Args:
        analyst_node: The analyst node function to execute
        initial_state: The initial state for the workflow
        toolkit: The toolkit containing available tools
        
    Returns:
        dict: Final state after the analyst completes its work
    """
    state = initial_state.copy()  # Make a copy to avoid modifying original
    
    # Get all available tools from our toolkit instance
    all_tools_in_toolkit = [
        getattr(toolkit, name) 
        for name in dir(toolkit) 
        if callable(getattr(toolkit, name)) and not name.startswith("__")
    ]
    
    # The ToolNode is a special LangGraph node that executes tool calls
    tool_node = ToolNode(all_tools_in_toolkit)
    
    # The ReAct loop can have up to 5 steps of reasoning and tool calls
    for step in range(5):
        print(f"Step {step + 1}: Running analyst...")
        
        # Run the analyst node
        result = analyst_node(state)
        
        # Check if the result contains tool calls
        if tools_condition(result) == "tools":
            print(f"Step {step + 1}: Tool calls detected, executing tools...")
            
            # Update state with the assistant's message containing tool calls
            state["messages"].extend(result["messages"])
            
            # Execute the tools and get tool results
            tool_result = tool_node.invoke(result)
            
            # Add tool results to messages
            state["messages"].extend(tool_result["messages"])
            
        else:
            print(f"Step {step + 1}: No tool calls, analyst finished.")
            # No tool calls, analyst is done
            # Update state with final result
            state["messages"].extend(result["messages"])
            
            # Copy any output fields from result to state
            for key, value in result.items():
                if key != "messages":
                    state[key] = value
            break
    
    return state