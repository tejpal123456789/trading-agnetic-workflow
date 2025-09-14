from langchain_openai import ChatOpenAI
from config import config
from dotenv import load_dotenv

load_dotenv()

# Initialize the powerful LLM for high-stakes reasoning tasks.
deep_thinking_llm = ChatOpenAI(
    model=config["deep_think_llm"],
    base_url=config["backend_url"],
    temperature=0.1
)
# Initialize the faster, cost-effective LLM for routine data processing.
quick_thinking_llm = ChatOpenAI(
    model=config["quick_think_llm"],
    base_url=config["backend_url"],
    temperature=0.1
)