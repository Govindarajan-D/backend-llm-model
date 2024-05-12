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
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import SystemMessage
import urllib
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableLambda


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
        system_message = """
                You are a helpful, fun and friendly stock market assistant for customers 
                Your name is NewsTrader.
                You are designed to answer questions about the share market stocks will be impacted by the relevant news

                Only answer questions related to the information provided in the news content that is given. Categorize those news as positive and negative.
                Tell how it will impact the stocks.

                If you are asked a question that is not in relation to stock market, respond with "I only answer about Stock Market News"
                News:
                {news}

                Question:
                {question}
            """
        self.system_message = system_message
        self.llm = llm
        self.chat = self.__form_rag_chain(system_prompt=system_message, llm=llm)
    def run(self, prompt: str) -> str:
        """
        Run the AI agent.
        """
        question = str(prompt)
        try:
            result = self.chat.invoke("question")
            return result["output"]
        except Exception as e:
            print(e)

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
        query = "What is the news on Tesla?"
        print(vector_store.similarity_search(query, k=3))
        return vector_store.as_retriever(search_kwargs={"k": top_k})
    def __form_rag_chain(self, system_prompt, llm):
        # Create a retriever from the vector store
            retriever = self.__create_stock_news_vector_store_retriever("news-api-data")

            # Create the prompt template from the system_prompt text
            llm_prompt = ChatPromptTemplate.from_template(system_prompt)
            news_retriever = retriever
            print({ "news": news_retriever, "question": RunnablePassthrough()})
            rag_chain = (
                { "news": news_retriever}
                | RunnableLambda(inspect)
            )
            return rag_chain
def inspect(state):
    """Print the state passed between Runnables in a langchain and pass it on"""
    print(state)
    return state

def format_print(data_string):
    print(data_string)

def format_docs(docs:List[Document]) -> str:
    """
    Prepares the news list for the system prompt.
    """
    str_docs = []
    for doc in docs:
        # Build the document without the contentVector
        doc_dict = {"_id": doc.page_content}
        doc_dict.update(doc.metadata)
        if "contentVector" in doc:
            del doc_dict["contentVector"]
        str_docs.append(json.dumps(doc_dict, default=str))
    # Return a single string containing each product JSON representation
    # separated by two newlines
    print("\n\n".join(str_docs))
    return "\n\n".join(str_docs)
