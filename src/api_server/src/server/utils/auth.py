from fastapi.security import APIKeyHeader
from fastapi import HTTPException, status, Depends
from pydantic_settings import BaseSettings
from config.logging_config import logger

class Settings(BaseSettings):
    api_key: str
    class Config:
        env_file = ".env"

settings = Settings()

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != settings.api_key:
        logger.warning(f"Failed API key attempt: {api_key}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    logger.info("API key validated successfully")
    return api_key