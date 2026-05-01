from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag_pipeline import create_rag_pipeline

app = FastAPI()
qa_chain = None


class Query(BaseModel):
    query: str


@app.on_event("startup")
def startup_event():
    global qa_chain
    qa_chain = create_rag_pipeline()


@app.get("/")
def home():
    return {"status": "RAG API running"}


@app.post("/ask")
def ask(q: Query):
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    result = qa_chain({"question": q.query})
    return {"answer": result["answer"]}