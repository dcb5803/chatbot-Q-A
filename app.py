# app.py
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from transformers import pipeline
import os

app = FastAPI()

API_KEY = os.environ.get("API_KEY", "deepika143")

# Load model once
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

class QAPayload(BaseModel):
    context: str
    question: str

@app.post("/answer")
def get_answer(payload: QAPayload, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    if not payload.context.strip() or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Context and question required")

    result = qa_pipeline(question=payload.question, context=payload.context)
    return {"answer": result['answer']}
