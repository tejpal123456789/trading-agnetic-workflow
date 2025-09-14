# This function is a factory that creates a LangGraph node for a researcher agent (Bull or Bear).
def create_researcher_node(llm, memory, role_prompt, agent_name):
    """
    Creates a node for a researcher agent.
Args:
        llm: The language model instance to be used by the agent.
        memory: The long-term memory instance for this agent to learn from past experiences.
        role_prompt: The specific system prompt defining the agent's persona (Bull or Bear).
        agent_name: The name of the agent, used for logging and identifying arguments.
    """
    def researcher_node(state):
        # First, combine all analyst reports into a single summary for context.
        situation_summary = f"""
        Market Report: {state['market_report']}
        Sentiment Report: {state['sentiment_report']}
        News Report: {state['news_report']}
        Fundamentals Report: {state['fundamentals_report']}
        """
        # Retrieve relevant memories from past, similar situations.
        past_memories = memory.get_memories(situation_summary)
        past_memory_str = "\n".join([mem['recommendation'] for mem in past_memories])
        
        # Construct the full prompt for the LLM.
        prompt = f"""{role_prompt}
        Here is the current state of the analysis:
        {situation_summary}
        Conversation history: {state['investment_debate_state']['history']}
        Your opponent's last argument: {state['investment_debate_state']['current_response']}
        Reflections from similar past situations: {past_memory_str or 'No past memories found.'}
        Based on all this information, present your argument conversationally."""
        
        # Invoke the LLM to generate the argument.
        response = llm.invoke(prompt)
        argument = f"{agent_name}: {response.content}"
        
        # Update the debate state with the new argument.
        debate_state = state['investment_debate_state'].copy()
        debate_state['history'] += "\n" + argument
        # Update the specific history for this agent (Bull or Bear).
        if agent_name == 'Bull Analyst':
            debate_state['bull_history'] += "\n" + argument
        else:
            debate_state['bear_history'] += "\n" + argument
        debate_state['current_response'] = argument
        debate_state['count'] += 1
        return {"investment_debate_state": debate_state}
    return researcher_node