import chromadb
from openai import OpenAI  
from dotenv import load_dotenv

load_dotenv()
import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config



# The FinancialSituationMemory class provides long-term memory 
# for storing and retrieving financial situations + recommendations.
class FinancialSituationMemory:
    def __init__(self, name, config):
        # Use OpenAI’s small embedding model for vectorizing text
        self.embedding_model = "text-embedding-3-small"
        
        # Initialize OpenAI client (pointing to your configured backend)
        self.client = OpenAI(base_url=config["backend_url"])
        
        # Create a ChromaDB client (with reset allowed for testing)
        self.chroma_client = chromadb.Client(chromadb.config.Settings(allow_reset=True))
        
        # Create a collection (like a table) to store situations + advice
        self.situation_collection = self.chroma_client.create_collection(name=name)

    def get_embedding(self, text):
        # Generate an embedding (vector) for the given text
        response = self.client.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        # Add new situations and recommendations to memory
        if not situations_and_advice:
            return
        
        # Offset ensures unique IDs (in case new data is added later)
        offset = self.situation_collection.count()
        ids = [str(offset + i) for i, _ in enumerate(situations_and_advice)]
        
        # Separate situations and their corresponding advice
        situations = [s for s, r in situations_and_advice]
        recommendations = [r for s, r in situations_and_advice]
        
        # Generate embeddings for all situations
        embeddings = [self.get_embedding(s) for s in situations]
        
        # Store everything in Chroma (vector DB)
        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in recommendations],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        # Retrieve the most similar past situations for a given query
        if self.situation_collection.count() == 0:
            return []
        
        # Embed the new/current situation
        query_embedding = self.get_embedding(current_situation)
        
        # Query the collection for similar embeddings
        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_matches, self.situation_collection.count()),
            include=["metadatas"],  # Only return recommendations
        )
        
        # Return extracted recommendations from the matches
        return [{'recommendation': meta['recommendation']} for meta in results['metadatas'][0]]
    

# Create a dedicated memory instance for each agent that learns.
bull_memory = FinancialSituationMemory("bull_memory", config)
bear_memory = FinancialSituationMemory("bear_memory", config)
trader_memory = FinancialSituationMemory("trader_memory", config)
invest_judge_memory = FinancialSituationMemory("invest_judge_memory", config)
risk_manager_memory = FinancialSituationMemory("risk_manager_memory", config)

