from pprint import pprint
import os
# Define our central configuration for this notebook run.
config = {
    "results_dir": "./results",
    # LLM settings specify which models to use for different cognitive tasks.
    "llm_provider": "openai",
    "deep_think_llm": "gpt-4o",       # A powerful model for complex reasoning and final decisions.
    "quick_think_llm": "gpt-4o-mini", # A fast, cheaper model for data processing and initial analysis.
     "backend_url": "https://api.openai.com/v1",
    # Debate and discussion settings control the flow of collaborative agents.
    "max_debate_rounds": 2,          # The Bull vs. Bear debate will have 2 rounds.
    "max_risk_discuss_rounds": 1,    # The Risk team has 1 round of debate.
    "max_recur_limit": 100,          # Safety limit for agent loops.
    # Tool settings control data fetching behavior.
    "online_tools": True,            # Use live APIs; set to False to use cached data for faster, cheaper runs.
    "data_cache_dir": "./data_cache" # Directory for caching online data.
}
# Create the cache directory if it doesn't already exist.
os.makedirs(config["data_cache_dir"], exist_ok=True)
print("Configuration dictionary created:")
pprint(config)
