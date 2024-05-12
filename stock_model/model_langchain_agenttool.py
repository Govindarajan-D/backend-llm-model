import os
import json
from typing import List
import pymongo
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch
from langchain.schema.document import Document
from langchain.agents import Tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.tools import StructuredTool
from langchain_core.messages import SystemMessage
import urllib

load_dotenv()
CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
USER_NAME = os.getenv("USER_NAME")
PASSWORD = os.getenv("PASSWORD")

EMBEDDINGS_DEPLOYMENT = "text-embedding-3-small"
COMPLETIONS_DEPLOYMENT = "gpt-35-turbo-16k"
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_KEY = os.getenv("AOAI_KEY")
AOAI_API_VERSION = "2023-05-15"
DB_CONNECTION_STRING = "mongodb+srv://" + urllib.parse.quote(str(USER_NAME)) + ":" + urllib.parse.quote(str(PASSWORD)) + "@" + CONNECTION_STRING
db_client = pymongo.MongoClient("mongodb+srv://" + urllib.parse.quote(str(USER_NAME)) + ":" + urllib.parse.quote(str(PASSWORD)) + "@" + CONNECTION_STRING)

db = db_client["news-data"]

class StockNewsAIAgent:
    """
    The StockNewsAIAgent class creates StockNews, an AI agent
    that can be used to answer questions about latest Stock Market
    news
    """
    def __init__(self, session_id: str):
        llm = AzureChatOpenAI(
            temperature = 1,
            openai_api_version = AOAI_API_VERSION,
            azure_endpoint = AOAI_ENDPOINT,
            openai_api_key = AOAI_KEY,
            azure_deployment = COMPLETIONS_DEPLOYMENT
        )
        self.embedding_model = AzureOpenAIEmbeddings(
            openai_api_version = AOAI_API_VERSION,
            azure_endpoint = AOAI_ENDPOINT,
            openai_api_key = AOAI_KEY,
            azure_deployment = EMBEDDINGS_DEPLOYMENT,
            chunk_size=10
        )
        system_message = SystemMessage(
            content = """
                You are a helpful, fun and friendly stock market assistant for customers 
                Your name is NewsTrader.
                You are designed to answer questions about the share market stocks will be impacted by the relevant news

                Only answer questions related to the information provided in the news content that is given. Categorize those news as positive and negative.
                Tell how it will impact the stocks.

                If you are asked a question that is not in relation to stock market, respond with "I only answer about Stock Market News"
            """
        )
        self.agent_executor = create_conversational_retrieval_agent(
                llm,
                self.__create_agent_tools(),
                system_message = system_message,
                memory_key=session_id,
                verbose=True
        )

    def run(self, prompt: str) -> str:
        """
        Run the AI agent.
        """
        result = self.agent_executor({"input": prompt})
        return result["output"]

    def __create_stock_news_vector_store_retriever(
            self,
            collection_name: str,
            top_k: int = 3
        ):
        """
        Returns a vector store retriever for the given collection.
        """
        vector_store =  AzureCosmosDBVectorSearch.from_connection_string(
            connection_string = DB_CONNECTION_STRING,
            namespace = f"news-data.{collection_name}",
            embedding = self.embedding_model,
            index_name = "VectorSearchIndex",
            embedding_key = "contentVector",
            text_key = "_id"
        )
        return vector_store.as_retriever(search_kwargs={"k": top_k})

    def __create_agent_tools(self) -> List[Tool]:
        """
        Returns a list of agent tools.
        """
        news_retriever = self.__create_stock_news_vector_store_retriever("news-api-data")

        # create a chain on the retriever to format the documents as JSON
        news_retriever_chain = news_retriever | format_docs

        tools = [
            Tool(
                name = "vector_search_category",
                func = news_retriever_chain.invoke,
                description = """
                    Searches MongoDB document information for similar category based 
                    on the question. Returns the news in JSON format.
                    """
            ),
            StructuredTool.from_function(get_news)
        ]
        return tools

def format_docs(docs:List[Document]) -> str:
    """
    Prepares the news list for the system prompt.
    """
    str_docs = []
    for doc in docs:
        # Build the document without the contentVector
        doc_dict = doc
        if "contentVector" in doc:
            del doc_dict["contentVector"]
        str_docs.append(json.dumps(doc_dict, default=str))
    # Return a single string containing each product JSON representation
    # separated by two newlines
    return "\n\n".join(str_docs)

# def get_news_by_category(category: str) -> str:
#     """
#     Retrieves a Category by its name
#     """
#     doc = db["news-api-data"].find_one({"category": category})
#     if "contentVector" in doc:
#         del doc["contentVector"]
#     return json.dumps(doc, default=str)

def get_news() -> str:
    """
    Retrieves a Category by its name
    """
    doc = db["news-api-data"].find({})
    if "contentVector" in doc:
        del doc["contentVector"]
    return json.dumps(doc, default=str)