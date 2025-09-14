# from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_core.messages import HumanMessage
# import datetime
# from rich.console import Console
# from rich.markdown import Markdown
# from tools.toolkit import toolkit
# # Initialize a console for rich printing.
# console = Console()
# # Helper function to run a single analyst's ReAct loop.
# def run_analyst(analyst_node, initial_state):
#     state = initial_state
#     # Get all available tools from our toolkit instance.
#     all_tools_in_toolkit = [getattr(toolkit, name) for name in dir(toolkit) if callable(getattr(toolkit, name)) and not name.startswith("__")]
#     # The ToolNode is a special LangGraph node that executes tool calls.
#     tool_node = ToolNode(all_tools_in_toolkit)
#     # The ReAct loop can have up to 5 steps of reasoning and tool calls.
#     for _ in range(5):
#         result = analyst_node(state)
#         # The tools_condition checks if the LLM's last message was a tool call.
#         if tools_condition(result) == "tools":
#             # If so, execute the tools and update the state.
#             tool_result = tool_node.invoke(result)
#             # Merge the tool results with the existing state to preserve all keys
#             state = {**state, **tool_result}
#         else:
#             # If not, the agent is done, so we break the loop.
#             state = result
#             break
#     return state
