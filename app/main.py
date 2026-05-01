from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_pipeline import create_rag_pipeline

app = FastAPI()

# 🚀 TRAINER STYLE → pipeline created on startup
qa_chain = create_rag_pipeline()


class Query(BaseModel):
    query: str


@app.get("/")
def home():
    return {"status": "RAG API running"}


@app.post("/ask")
def ask(q: Query):
    result = qa_chain({"question": q.query})
    return {"answer": result["answer"]}
