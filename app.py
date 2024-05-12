from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_models.ai_request import AIRequest
from stock_model.news_model import StockNewsAIAgent

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

agent_pool = {}

@app.get("/")
def root():
    return {"status":"ready"}

@app.post("/ai")
def run_stock_news_ai_agent(request: AIRequest):
    if request.session_id not in agent_pool:
        agent_pool[request.session_id] = StockNewsAIAgent(request.session_id)
    return {"message": agent_pool[request.session_id].run(request.prompt)}