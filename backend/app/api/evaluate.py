# app/api/evaluate.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.evaluator import evaluate_single

router = APIRouter()

class EvalRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]

class EvalResponse(BaseModel):
    metrics: dict

@router.post("/", response_model=EvalResponse)
def evaluate(req: EvalRequest):
    if not req.question or not req.answer or not req.contexts:
        raise HTTPException(status_code=400, detail="question, answer, and contexts are required.")
    metrics = evaluate_single(req.question, req.answer, req.contexts)
    return {"metrics": metrics}
