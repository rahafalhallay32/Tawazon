# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from model import generate_answer

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Receives a Shariah-related question and returns an answer using GPT.
    """
    answer = generate_answer(request.question)
    return {"answer": answer}
