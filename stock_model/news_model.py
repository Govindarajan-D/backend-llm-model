"""
The CosmicWorksAIAgent class encapsulates a LangChain 
agent that can be used to answer questions about Cosmic Works
products, customers, and sales.
"""
import os
import json
from typing import List
import pymongo
from dotenv import load_dotenv, find_dotenv
import os
import urllib.parse
import pymongo
import time
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
import urllib 

import urllib

load_dotenv()
CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
USER_NAME = os.environ.get("USER_NAME")
PASSWORD = os.environ.get("PASSWORD")
EMBEDDINGS_DEPLOYMENT_NAME = "text-embedding-3-small"
COMPLETIONS_DEPLOYMENT_NAME = "gpt-35-turbo-16k"
AOAI_ENDPOINT = os.environ.get("AOAI_ENDPOINT")
AOAI_KEY = os.environ.get("AOAI_KEY")
AOAI_API_VERSION = "2023-05-15"

db_client = pymongo.MongoClient("mongodb+srv://" + urllib.parse.quote(USER_NAME) + ":" + urllib.parse.quote(PASSWORD) + "@" + CONNECTION_STRING)
# Create database to hold cosmic works data
# MongoDB will create the database if it does not exist
db = db_client["news-data"]



class StockNewsAIAgent:
    """
    The StockNewsAIAgent class creates StockNews, an AI agent
    that can be used to answer questions about latest Stock Market
    news
    """
    def __init__(self, session_id: str):
        self.ai_client = AzureOpenAI(
            azure_endpoint = AOAI_ENDPOINT,
            api_version = AOAI_API_VERSION,
            api_key = AOAI_KEY
            )
        self.system_message = """
                You are a helpful, fun and friendly stock market assistant for customers 
                Your name is NewsTrader.
                You are designed to answer questions about the share market stocks will be impacted by the relevant news

                Only answer questions related to the information provided in the news content that is given. Categorize those news as positive and negative.
                Tell how it will impact the stocks.

                If you are asked a question that is not in the list, respond with "I only answer about stock market."

                """
    def run(self, prompt: str) -> str:
        """
        Run the AI agent.
        """
        result = self.rag_with_vector_search(prompt)
        return result
    def generate_embeddings(self, text: str):
        '''
        Generate embeddings from string of text using the deployed Azure OpenAI API embeddings model.
        This will be used to vectorize document data and incoming user messages for a similarity search with
        the vector index.
        '''
        response = self.ai_client.embeddings.create(input=text, model=EMBEDDINGS_DEPLOYMENT_NAME)
        embeddings = response.data[0].embedding
        time.sleep(0.5) # rest period to avoid rate limiting on AOAI
        return embeddings
    
    def vector_search(self, collection_name, query, num_results=3):
        """
        Perform a vector search on the specified collection by vectorizing
        the query and searching the vector index for the most similar documents.

        returns a list of the top num_results most similar documents
        """
        collection = db[collection_name]
        query_embedding = self.generate_embeddings(query)    
        pipeline = [
            {
                '$search': {
                    "cosmosSearch": {
                        "vector": query_embedding,
                        "path": "contentVector",
                        "k": num_results
                    },
                    "returnStoredSource": True }},
            {'$project': { 'similarityScore': { '$meta': 'searchScore' }, 'document' : '$$ROOT' } }
        ]
        results = collection.aggregate(pipeline)
        return results

    def rag_with_vector_search(self, question: str, num_results: int = 5):
        """
        Use the RAG model to generate a prompt using vector search results based on the
        incoming question.  
        """
        # perform the vector search and build news list
        results = self.vector_search("news-api-data", question, num_results=num_results)
        news_list = ""
        for result in results:
            if "contentVector" in result["document"]:
                del result["document"]["contentVector"]
            news_list += json.dumps(result["document"], indent=4, default=str) + "\n\n"

        # generate prompt for the LLM with vector results
        formatted_prompt = self.system_message + news_list

        # prepare the LLM request
        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": question}
        ]

        completion = self.ai_client.chat.completions.create(messages=messages, model=COMPLETIONS_DEPLOYMENT_NAME)
        return completion.choices[0].message.content