import json
from fastapi import APIRouter, Depends
from app.auth import verify_api_key
from app.model import train_model
from pathlib import Path

router = APIRouter(prefix="/admin", dependencies=[Depends(verify_api_key)])

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "qa_data.json"

@router.get("/qa")
def get_qa_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@router.post("/qa")
def update_qa_data(data: list):
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return {"status": 200, "message": "Q&A data updated"}


@router.post("/retrain")
def retrain_model():
    train_model()
    return {"status": 200, "message": "Model retrained successfully"}
