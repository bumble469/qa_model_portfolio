from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import os
# from dotenv import load_dotenv
from app.model import load_or_train_model, get_answer
from app.admin import router as admin_router

# load_dotenv()
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_or_train_model()
    yield

app = FastAPI(
    title="Portfolio Q&A API",
    description="Domain-specific Q&A assistant powered by scikit-learn",
    version="1.0.0",
    lifespan=lifespan
)

origins_env = os.getenv("ALLOWED_ORIGINS")
if not origins_env:
    raise RuntimeError("ALLOWED_ORIGINS environment variable is not set")

allowed_origins = [origin.strip() for origin in origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type", "x-api-key"],
)

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)

class AnswerResponse(BaseModel):
    answer: str | None
    confidence: float

@app.post("/api/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    answer, confidence = get_answer(payload.question)
    return AnswerResponse(answer=answer, confidence=confidence)


app.include_router(admin_router)
