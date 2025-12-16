from fastapi import APIRouter, Depends
from app.auth import verify_api_key
from app.model import train_model
from pathlib import Path

router = APIRouter(prefix="/admin", dependencies=[Depends(verify_api_key)])

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "qa_data.json"

@router.post("/retrain")
def retrain_model():
    train_model()
    return {"status": 200, "message": "Model retrained successfully"}
