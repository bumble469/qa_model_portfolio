
from fastapi import Header, HTTPException
from app.config import secret_key


def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key missing")

    if x_api_key != secret_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True