
import os
import json
from typing import List
import pymongo
from dotenv import load_dotenv, find_dotenv
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
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

EMBEDDINGS_DEPLOYMENT = "embeddings"
COMPLETIONS_DEPLOYMENT = "completions"
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_KEY = os.getenv("AOAI_KEY")
AOAI_API_VERSION = "2023-09-01-preview"
DB_CONNECTION_STRING = "mongodb+srv://" + urllib.parse.quote(USER_NAME) + ":" + urllib.parse.quote(PASSWORD) + "@" + CONNECTION_STRING
db_client = pymongo.MongoClient("mongodb+srv://" + urllib.parse.quote(USER_NAME) + ":" + urllib.parse.quote(PASSWORD) + "@" + CONNECTION_STRING)
collection = db_client["news-data"]["news-api-data"]

llm = AzureChatOpenAI(
    temperature = 1,
    openai_api_version = AOAI_API_VERSION,
    azure_endpoint = AOAI_ENDPOINT,
    openai_api_key = AOAI_KEY,
    azure_deployment = COMPLETIONS_DEPLOYMENT
)
# embedding_model = AzureOpenAIEmbeddings(
#     openai_api_version = AOAI_API_VERSION,
#     azure_endpoint = AOAI_ENDPOINT,
#     openai_api_key = AOAI_KEY,   
#     azure_deployment = "embeddings",
#     chunk_size=10
# )
embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint= AOAI_ENDPOINT,
    openai_api_key = AOAI_KEY
)
print(embedding_model._tokenize(["Test","Message"],2048))
print(embedding_model.embed_query("Test Message"))


collection_name = "news-api-data"
vector_store =  AzureCosmosDBVectorSearch.from_connection_string(
            connection_string = DB_CONNECTION_STRING,
            namespace="news-data.news-api-data",
            embedding = embedding_model,
            index_name = "VectorSearchIndex",
            embedding_key = "contentVector",
            text_key = "_id"
        )
print(embedding_model._tokenize(["how","are","you"],2048))
print(embedding_model.embed_documents(["Test", "Message"]))
# query = "What is the news on Tesla?"
# print(vector_store.similarity_search(query, k=3))