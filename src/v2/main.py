# src/server/main.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import io
import os
import shutil
import sqlite3
from time import time
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
from functools import lru_cache
import asyncio
from collections import Counter
from contextlib import asynccontextmanager
import base64

import uvicorn
import aiohttp
import bleach
from databases import Database
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, Form, Response, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, field_validator, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from pybreaker import CircuitBreaker
from PIL import Image

from src.server.utils.auth import (
    get_current_user, get_current_user_with_admin, login, refresh_token, register,
    app_register, register_bulk_users, seed_initial_data, TokenResponse, Settings,
    LoginRequest, RegisterRequest, bearer_scheme
)
from src.server.utils.crypto import decrypt_data
from config.tts_config import SPEED, ResponseFormat, config as tts_config
from config.logging_config import logger
from src.server.db import database

settings = Settings()

runtime_config = {
    "chat_rate_limit": settings.chat_rate_limit,
    "speech_rate_limit": settings.speech_rate_limit,
}
metrics = {
    "request_count": Counter(),
    "response_times": {}
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    await database.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            is_admin BOOLEAN NOT NULL DEFAULT 0,
            session_key TEXT
        )
    """)
    await database.execute("""
        CREATE TABLE IF NOT EXISTS app_users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            session_key TEXT
        )
    """)
    await database.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            result TEXT,
            created_at REAL NOT NULL,
            completed_at REAL
        )
    """)
    await seed_initial_data()
    yield
    await database.disconnect()

app = FastAPI(
    title="Dhwani API",
    description="A multilingual AI-powered API supporting Indian languages for chat, text-to-speech, audio processing, and transcription. "
                "**Authentication Guide:** \n"
                "1. Obtain an access token by sending a POST request to `/v1/token` with `username` and `password`. \n"
                "2. Click the 'Authorize' button (top-right), enter your access token (e.g., `your_access_token`) in the 'bearerAuth' field, and click 'Authorize'. \n"
                "All protected endpoints require this token for access. \n",
    version="1.0.0",
    redirect_slashes=False,
    openapi_tags=[
        {"name": "Chat", "description": "Chat-related endpoints"},
        {"name": "Audio", "description": "Audio processing and TTS endpoints"},
        {"name": "Translation", "description": "Text translation endpoints"},
        {"name": "Authentication", "description": "User authentication and registration"},
        {"name": "Utility", "description": "General utility endpoints"},
    ],
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    start_time = time()
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    endpoint = request.url.path
    duration = time() - start_time
    metrics["request_count"][endpoint] += 1
    metrics["response_times"].setdefault(endpoint, []).append(duration)
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True, extra={
        "endpoint": request.url.path,
        "method": request.method,
        "client_ip": get_remote_address(request)
    })
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

async def get_user_id_for_rate_limit(request: Request):
    try:
        credentials = bearer_scheme(request)
        user_id = await get_current_user(credentials)
        return user_id
    except Exception:
        return get_remote_address(request)

limiter = Limiter(key_func=get_user_id_for_rate_limit)

class SpeechRequest(BaseModel):
    input: str = Field(..., description="Base64-encoded encrypted text to convert to speech (max 1000 characters after decryption)")
    voice: str = Field(..., description="Base64-encoded encrypted voice identifier")
    model: str = Field(..., description="Base64-encoded encrypted TTS model")
    response_format: ResponseFormat = Field(tts_config.response_format, description="Audio format: mp3, flac, or wav")
    speed: float = Field(SPEED, description="Speech speed (default: 1.0)")

    @field_validator("input", "voice", "model")
    def must_be_valid_base64(cls, v):
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError("Field must be valid base64-encoded data")
        return v

    @field_validator("response_format")
    def validate_response_format(cls, v):
        supported_formats = [ResponseFormat.MP3, ResponseFormat.FLAC, ResponseFormat.WAV]
        if v not in supported_formats:
            raise ValueError(f"Response format must be one of {[fmt.value for fmt in supported_formats]}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "input": "base64_encoded_encrypted_hello",
                "voice": "base64_encoded_encrypted_female-1",
                "model": "base64_encoded_encrypted_tts-model-1",
                "response_format": "mp3",
                "speed": 1.0
            }
        }

class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="Transcribed text from the audio")

    class Config:
        json_schema_extra = {"example": {"text": "Hello, how are you?"}} 

class TextGenerationResponse(BaseModel):
    text: str = Field(..., description="Generated text response")

    class Config:
        json_schema_extra = {"example": {"text": "Hi there, I'm doing great!"}} 

class AudioProcessingResponse(BaseModel):
    result: str = Field(..., description="Processed audio result")

    class Config:
        json_schema_extra = {"example": {"result": "Processed audio output"}} 

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="Base64-encoded encrypted prompt (max 1000 characters after decryption)")
    src_lang: str = Field(..., description="Base64-encoded encrypted source language code")
    tgt_lang: str = Field(..., description="Base64-encoded encrypted target language code")

    @field_validator("prompt", "src_lang", "tgt_lang")
    def must_be_valid_base64(cls, v):
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError("Field must be valid base64-encoded data")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "base64_encoded_encrypted_prompt",
                "src_lang": "base64_encoded_encrypted_kan_Knda",
                "tgt_lang": "base64_encoded_encrypted_kan_Knda"
            }
        }

class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated chat response")

    class Config:
        json_schema_extra = {"example": {"response": "Hi there, I'm doing great!"}} 

class TranslationRequest(BaseModel):
    sentences: List[str] = Field(..., description="List of base64-encoded encrypted sentences")
    src_lang: str = Field(..., description="Base64-encoded encrypted source language code")
    tgt_lang: str = Field(..., description="Base64-encoded encrypted target language code")

    @field_validator("sentences", "src_lang", "tgt_lang")
    def must_be_valid_base64(cls, v):
        if isinstance(v, list):
            for item in v:
                try:
                    base64.b64decode(item)
                except Exception:
                    raise ValueError("Each sentence must be valid base64-encoded data")
        else:
            try:
                base64.b64decode(v)
            except Exception:
                raise ValueError("Field must be valid base64-encoded data")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "sentences": ["base64_encoded_encrypted_hello", "base64_encoded_encrypted_how_are_you"],
                "src_lang": "base64_encoded_encrypted_en",
                "tgt_lang": "base64_encoded_encrypted_kan_Knda"
            }
        }

class TranslationResponse(BaseModel):
    translations: List[str] = Field(..., description="Translated sentences")

    class Config:
        json_schema_extra = {"example": {"translations": ["ನಮಸ್ಕಾರ", "ನೀವು ಹೇಗಿದ್ದೀರಿ?"]}} 

class VisualQueryRequest(BaseModel):
    query: str = Field(..., description="Base64-encoded encrypted text query")
    src_lang: str = Field(..., description="Base64-encoded encrypted source language code")
    tgt_lang: str = Field(..., description="Base64-encoded encrypted target language code")

    @field_validator("query", "src_lang", "tgt_lang")
    def must_be_valid_base64(cls, v):
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError("Field must be valid base64-encoded data")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "query": "base64_encoded_encrypted_describe_image",
                "src_lang": "base64_encoded_encrypted_kan_Knda",
                "tgt_lang": "base64_encoded_encrypted_kan_Knda"
            }
        }

class VisualQueryResponse(BaseModel):
    answer: str

class BulkRegisterResponse(BaseModel):
    successful: List[str] = Field(..., description="List of successfully registered usernames")
    failed: List[dict] = Field(..., description="List of failed registrations with reasons")

    class Config:
        json_schema_extra = {
            "example": {
                "successful": ["user1", "user2"],
                "failed": [{"username": "user3", "reason": "Username already exists"}]
            }
        }

class ConfigUpdateRequest(BaseModel):
    chat_rate_limit: Optional[str] = Field(None, description="Chat endpoint rate limit (e.g., '100/minute')")
    speech_rate_limit: Optional[str] = Field(None, description="Speech endpoint rate limit (e.g., '5/minute')")

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict] = None
    created_at: float
    completed_at: Optional[float] = None

tts_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)
chat_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)
audio_proc_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)
translate_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)
visual_query_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)
speech_to_speech_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

class TTSService(ABC):
    @abstractmethod
    async def generate_speech(self, payload: dict) -> aiohttp.ClientResponse:
        pass

class ExternalTTSService(TTSService):
    @tts_breaker
    async def generate_speech(self, payload: dict) -> aiohttp.ClientResponse:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    settings.external_tts_url,
                    json=payload,
                    headers={"accept": "application/json", "Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status >= 400:
                        raise HTTPException(status_code=response.status, detail=await response.text())
                    return response
            except asyncio.TimeoutError:
                logger.error("External TTS API timeout")
                raise HTTPException(status_code=504, detail="External TTS API timeout")
            except aiohttp.ClientError as e:
                logger.error(f"External TTS API error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"External TTS API error: {str(e)}")

def get_tts_service() -> TTSService:
    return ExternalTTSService()

@app.get("/v1/health", 
         summary="Check API Health",
         description="Returns detailed health status of the API, including database and external service connectivity.",
         tags=["Utility"],
         response_model=dict)
async def health_check():
    health_status = {"status": "healthy", "model": settings.llm_model_name}
    
    try:
        await database.fetch_one("SELECT 1")
        health_status["database"] = "connected"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["database"] = f"error: {str(e)}"
        logger.error(f"Database health check failed: {str(e)}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(settings.external_tts_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                health_status["tts_service"] = "reachable" if resp.status < 400 else f"error: {resp.status}"
    except Exception as e:
        health_status["tts_service"] = f"error: {str(e)}"
        logger.error(f"TTS service health check failed: {str(e)}")
    
    return health_status

@app.get("/v1/metrics",
         summary="Get API Metrics",
         description="Returns request counts and average response times per endpoint. Requires admin access.",
         tags=["Utility"],
         response_model=dict)
async def get_metrics(current_user: str = Depends(get_current_user_with_admin)):
    metrics_summary = {
        "request_count": dict(metrics["request_count"]),
        "average_response_times": {}
    }
    for endpoint, times in metrics["response_times"].items():
        avg_time = sum(times) / len(times) if times else 0
        metrics_summary["average_response_times"][endpoint] = f"{avg_time:.3f}s"
    return metrics_summary

@app.get("/",
         summary="Redirect to Docs",
         description="Redirects to the Swagger UI documentation.",
         tags=["Utility"])
async def home():
    return RedirectResponse(url="/docs")

@app.post("/v1/token", 
          response_model=TokenResponse,
          summary="User Login",
          description="Authenticate a user with encrypted email and device token to obtain an access token and refresh token. Requires X-Session-Key header.",
          tags=["Authentication"],
          responses={
              200: {"description": "Successful login", "model": TokenResponse},
              400: {"description": "Invalid encrypted data"},
              401: {"description": "Invalid email or device token"},
              429: {"description": "Too many login attempts"}
          })
@limiter.limit("5/minute")
async def token(
    login_request: LoginRequest,
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    login_request.username = bleach.clean(login_request.username)
    login_request.password = bleach.clean(login_request.password)
    return await login(login_request, x_session_key)

@app.post("/v1/refresh", 
          response_model=TokenResponse,
          summary="Refresh Access Token",
          description="Generate a new access token and refresh token using an existing valid refresh token.",
          tags=["Authentication"],
          responses={
              200: {"description": "New tokens issued", "model": TokenResponse},
              401: {"description": "Invalid or expired refresh token"}
          })
async def refresh(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    return await refresh_token(credentials)

@app.post("/v1/register", 
          response_model=TokenResponse,
          summary="Register New User (Admin Only)",
          description="Create a new user account in the `users` table. Only admin accounts can register new users.",
          tags=["Authentication"],
          responses={
              200: {"description": "User registered successfully", "model": TokenResponse},
              400: {"description": "Username already exists"},
              403: {"description": "Admin access required"}
          })
async def register_user(
    register_request: RegisterRequest,
    current_user: str = Depends(get_current_user_with_admin)
):
    register_request.username = bleach.clean(register_request.username)
    register_request.password = bleach.clean(register_request.password)
    return await register(register_request, current_user)

@app.post("/v1/app/register",
          response_model=TokenResponse,
          summary="Register New App User",
          description="Create a new user account for the mobile app in the `app_users` table using an encrypted email and device token. Requires X-Session-Key header.",
          tags=["Authentication"],
          responses={
              200: {"description": "User registered successfully", "model": TokenResponse},
              400: {"description": "Email already registered or invalid encrypted data"},
              429: {"description": "Rate limit exceeded"}
          })
@limiter.limit(settings.speech_rate_limit)
async def app_register_user(
    request: Request,
    register_request: RegisterRequest,
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    register_request.username = bleach.clean(register_request.username)
    register_request.password = bleach.clean(register_request.password)
    logger.info(f"App registration attempt")
    return await app_register(register_request, x_session_key)

async def process_bulk_users(csv_content: str, current_user: str, task_id: str):
    await database.execute(
        "INSERT INTO tasks (task_id, status, created_at) VALUES (:task_id, :status, :created_at)",
        {"task_id": task_id, "status": "running", "created_at": time()}
    )
    try:
        result = await register_bulk_users(csv_content, current_user)
        await database.execute(
            "UPDATE tasks SET status = :status, result = :result, completed_at = :completed_at WHERE task_id = :task_id",
            {
                "task_id": task_id,
                "status": "completed",
                "result": str(result),
                "completed_at": time()
            }
        )
        logger.info(f"Background bulk registration completed for task {task_id}: {len(result['successful'])} succeeded, {len(result['failed'])} failed")
    except Exception as e:
        await database.execute(
            "UPDATE tasks SET status = :status, result = :result, completed_at = :completed_at WHERE task_id = :task_id",
            {
                "task_id": task_id,
                "status": "failed",
                "result": f"Error: {str(e)}",
                "completed_at": time()
            }
        )
        logger.error(f"Background bulk registration failed for task {task_id}: {str(e)}")

@app.post("/v1/register_bulk", 
          response_model=dict,
          summary="Register Multiple Users via CSV",
          description="Upload a CSV file with 'username' and 'password' columns to register multiple users in the background. Returns a task ID to track progress. Requires admin access.",
          tags=["Authentication"],
          responses={
              202: {"description": "Bulk registration started", "content": {"application/json": {"example": {"message": "Bulk registration started", "task_id": "unique-id"}}}},
              400: {"description": "Invalid CSV format or data"},
              403: {"description": "Admin access required"},
              429: {"description": "Rate limit exceeded"}
          })
@limiter.limit("10/minute")
async def register_bulk(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file with 'username' and 'password' columns"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    current_user = await get_current_user_with_admin(credentials)
    
    if not file.filename.endswith('.csv') or file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Invalid file type; only CSV allowed")
    
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large; max 10MB")
    
    try:
        csv_content = content.decode("utf-8")
        task_id = str(time())
        background_tasks.add_task(process_bulk_users, csv_content, current_user, task_id)
        return {"message": "Bulk registration started", "task_id": task_id}
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid CSV encoding; must be UTF-8")

@app.get("/v1/task_status/{task_id}",
         response_model=TaskStatusResponse,
         summary="Check Task Status",
         description="Retrieve the status and result of a background task (e.g., bulk registration). Requires admin access.",
         tags=["Authentication"],
         responses={
             200: {"description": "Task status", "model": TaskStatusResponse},
             404: {"description": "Task not found"},
             403: {"description": "Admin access required"}
         })
async def get_task_status(
    task_id: str,
    current_user: str = Depends(get_current_user_with_admin)
):
    task = await database.fetch_one(
        "SELECT task_id, status, result, created_at, completed_at FROM tasks WHERE task_id = :task_id",
        {"task_id": task_id}
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    result = eval(task["result"]) if task["result"] and task["status"] == "completed" else task["result"]
    return TaskStatusResponse(
        task_id=task["task_id"],
        status=task["status"],
        result=result,
        created_at=task["created_at"],
        completed_at=task["completed_at"]
    )

@app.get("/v1/export_db",
         summary="Export User Database",
         description="Download the current user database as a SQLite file. Requires admin access.",
         tags=["Authentication"],
         responses={
             200: {"description": "SQLite database file", "content": {"application/octet-stream": {}}},
             403: {"description": "Admin access required"},
             500: {"description": "Error exporting database"}
         })
async def export_db(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    current_user = await get_current_user_with_admin(credentials)
    db_path = settings.database_path
    
    if not os.path.exists(db_path):
        raise HTTPException(status_code=500, detail="Database file not found")
    
    logger.info(f"Database export requested by admin: {current_user}")
    return FileResponse(
        db_path,
        filename="users.db",
        media_type="application/octet-stream"
    )

@app.post("/v1/import_db",
          summary="Import User Database",
          description="Upload a SQLite database file to replace the current user database. Requires admin access.",
          tags=["Authentication"],
          responses={
              200: {"description": "Database imported successfully"},
              400: {"description": "Invalid database file"},
              403: {"description": "Admin access required"},
              500: {"description": "Error importing database"}
          })
async def import_db(
    request: Request,
    file: UploadFile = File(..., description="SQLite database file to import"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    current_user = await get_current_user_with_admin(credentials)
    db_path = settings.database_path
    temp_path = "users_temp.db"
    
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large; max 10MB")
    
    try:
        with open(temp_path, "wb") as f:
            f.write(content)
        
        async with database.transaction():
            conn = sqlite3.connect(temp_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('users', 'app_users');")
            tables = [row[0] for row in cursor.fetchall()]
            if 'users' not in tables or 'app_users' not in tables:
                conn.close()
                os.remove(temp_path)
                raise HTTPException(status_code=400, detail="Uploaded file is not a valid user database")
            
            cursor.execute("PRAGMA table_info(users);")
            columns = [col[1] for col in cursor.fetchall()]
            expected_columns = ["username", "password", "is_admin", "session_key"]
            if not all(col in columns for col in expected_columns):
                conn.close()
                os.remove(temp_path)
                raise HTTPException(status_code=400, detail="Uploaded database has an incompatible schema")
            
            conn.close()
        
        shutil.move(temp_path, db_path)
        logger.info(f"Database imported successfully by admin: {current_user}")
        return {"message": "Database imported successfully"}
    
    except sqlite3.Error as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"SQLite error during import: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid SQLite database: {str(e)}")
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Error importing database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error importing database: {str(e)}")

@app.post("/v1/update_config",
          summary="Update Runtime Configuration",
          description="Update rate limits dynamically. Requires admin access. Changes are in-memory and reset on restart.",
          tags=["Utility"],
          responses={
              200: {"description": "Configuration updated successfully"},
              400: {"description": "Invalid configuration values"},
              403: {"description": "Admin access required"}
          })
async def update_config(
    config_request: ConfigUpdateRequest,
    current_user: str = Depends(get_current_user_with_admin)
):
    if config_request.chat_rate_limit:
        try:
            limiter._check_rate(config_request.chat_rate_limit)
            runtime_config["chat_rate_limit"] = config_request.chat_rate_limit
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid chat_rate_limit format; use 'X/minute'")
    
    if config_request.speech_rate_limit:
        try:
            limiter._check_rate(config_request.speech_rate_limit)
            runtime_config["speech_rate_limit"] = config_request.speech_rate_limit
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid speech_rate_limit format; use 'X/minute'")
    
    logger.info(f"Runtime config updated by {current_user}: {runtime_config}")
    return {"message": "Configuration updated successfully", "current_config": runtime_config}

@app.post("/v1/audio/speech",
          summary="Generate Speech from Text",
          description="Convert encrypted text to speech using an external TTS service. Requires authentication and X-Session-Key header.",
          tags=["Audio"],
          responses={
              200: {"description": "Audio stream", "content": {"audio/mp3": {}}},
              400: {"description": "Invalid or empty input"},
              401: {"description": "Unauthorized - Token required"},
              429: {"description": "Rate limit exceeded"},
              503: {"description": "Service unavailable due to repeated failures"},
              504: {"description": "TTS service timeout"}
          })
@limiter.limit(lambda: runtime_config["speech_rate_limit"])
async def generate_audio(
    request: Request,
    speech_request: SpeechRequest = Depends(),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key"),
    tts_service: TTSService = Depends(get_tts_service)
):
    user_id = await get_current_user(credentials)
    session_key = base64.b64decode(x_session_key)
    
    try:
        encrypted_input = base64.b64decode(speech_request.input)
        decrypted_input = decrypt_data(encrypted_input, session_key).decode("utf-8")
        decrypted_input = bleach.clean(decrypted_input)
        encrypted_voice = base64.b64decode(speech_request.voice)
        decrypted_voice = decrypt_data(encrypted_voice, session_key).decode("utf-8")
        decrypted_voice = bleach.clean(decrypted_voice)
        encrypted_model = base64.b64decode(speech_request.model)
        decrypted_model = decrypt_data(encrypted_model, session_key).decode("utf-8")
        decrypted_model = bleach.clean(decrypted_model)
    except Exception as e:
        logger.error(f"Input decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted input")
    
    if not decrypted_input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")
    if len(decrypted_input) > 1000:
        raise HTTPException(status_code=400, detail="Decrypted input cannot exceed 1000 characters")
    
    logger.info("Processing speech request", extra={
        "endpoint": "/v1/audio/speech",
        "input_length": len(decrypted_input),
        "client_ip": get_remote_address(request),
        "user_id": user_id
    })
    
    payload = {
        "input": decrypted_input,
        "voice": decrypted_voice,
        "model": decrypted_model,
        "response_format": speech_request.response_format.value,
        "speed": speech_request.speed
    }
    
    response = await tts_service.generate_speech(payload)
    audio_content = await response.read()
    
    headers = {
        "Content-Disposition": f"inline; filename=\"speech.{speech_request.response_format.value}\"",
        "Cache-Control": "no-cache",
        "Content-Type": f"audio/{speech_request.response_format.value}"
    }
    
    async def stream_response():
        yield audio_content
    
    return StreamingResponse(
        stream_response(),
        media_type=f"audio/{speech_request.response_format.value}",
        headers=headers
    )

@lru_cache(maxsize=100)
def cached_chat_response(prompt: str, src_lang: str, tgt_lang: str) -> str:
    return None

@app.post("/v1/chat", 
          response_model=ChatResponse,
          summary="Chat with AI",
          description="Generate a chat response from an encrypted prompt and encrypted language code. Requires authentication and X-Session-Key header.",
          tags=["Chat"],
          responses={
              200: {"description": "Chat response", "model": ChatResponse},
              400: {"description": "Invalid prompt, encrypted data, or language code"},
              401: {"description": "Unauthorized - Token required"},
              429: {"description": "Rate limit exceeded"},
              503: {"description": "Service unavailable due to repeated failures"},
              504: {"description": "Chat service timeout"}
          })
@limiter.limit(lambda: runtime_config["chat_rate_limit"])
async def chat(
    request: Request,
    chat_request: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    user_id = await get_current_user(credentials)
    session_key = base64.b64decode(x_session_key)
    
    try:
        encrypted_prompt = base64.b64decode(chat_request.prompt)
        decrypted_prompt = decrypt_data(encrypted_prompt, session_key).decode("utf-8")
        decrypted_prompt = bleach.clean(decrypted_prompt)
        encrypted_src_lang = base64.b64decode(chat_request.src_lang)
        decrypted_src_lang = decrypt_data(encrypted_src_lang, session_key).decode("utf-8")
        decrypted_src_lang = bleach.clean(decrypted_src_lang)
        encrypted_tgt_lang = base64.b64decode(chat_request.tgt_lang)
        decrypted_tgt_lang = decrypt_data(encrypted_tgt_lang, session_key).decode("utf-8")
        decrypted_tgt_lang = bleach.clean(decrypted_tgt_lang)
    except Exception as e:
        logger.error(f"Decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted data")
    
    if not decrypted_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if len(decrypted_prompt) > 1000:
        raise HTTPException(status_code=400, detail="Decrypted prompt cannot exceed 1000 characters")
    
    cache_key = f"{decrypted_prompt}:{decrypted_src_lang}:{decrypted_tgt_lang}"
    cached_response = cached_chat_response(decrypted_prompt, decrypted_src_lang, decrypted_tgt_lang)
    if cached_response:
        logger.info(f"Cache hit for chat request: {cache_key}")
        return ChatResponse(response=cached_response)
    
    logger.info(f"Received prompt: {decrypted_prompt}, src_lang: {decrypted_src_lang}, user_id: {user_id}")
    
    @chat_breaker
    async def call_chat_api():
        async with aiohttp.ClientSession() as session:
            external_url = "https://slabstech-dhwani-internal-api-server.hf.space/v1/chat"
            payload = {
                "prompt": decrypted_prompt,
                "src_lang": decrypted_src_lang,
                "tgt_lang": decrypted_tgt_lang
            }
            async with session.post(
                external_url,
                json=payload,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json"
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status >= 400:
                    raise HTTPException(status_code=response.status, detail=await response.text())
                return await response.json()
    
    try:
        response_data = await call_chat_api()
        response_text = response_data.get("response", "")
        cached_chat_response(decrypted_prompt, decrypted_src_lang, decrypted_tgt_lang)
        logger.info(f"Generated Chat response from external API: {response_text}")
        return ChatResponse(response=response_text)
    except asyncio.TimeoutError:
        logger.error("External chat API request timed out")
        raise HTTPException(status_code=504, detail="Chat service timeout")
    except aiohttp.ClientError as e:
        logger.error(f"Error calling external chat API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/v1/process_audio/", 
          response_model=AudioProcessingResponse,
          summary="Process Audio File",
          description="Process an uploaded audio file in the specified language. Requires authentication and X-Session-Key header.",
          tags=["Audio"],
          responses={
              200: {"description": "Processed result", "model": AudioProcessingResponse},
              400: {"description": "Invalid encrypted audio or language"},
              401: {"description": "Unauthorized - Token required"},
              429: {"description": "Rate limit exceeded"},
              503: {"description": "Service unavailable due to repeated failures"},
              504: {"description": "Audio processing timeout"}
          })
@limiter.limit(lambda: runtime_config["chat_rate_limit"])
async def process_audio(
    request: Request,
    file: UploadFile = File(..., description="Encrypted audio file to process"),
    language: str = Query(..., description="Base64-encoded encrypted language of the audio (kannada, hindi, tamil after decryption)"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    user_id = await get_current_user(credentials)
    session_key = base64.b64decode(x_session_key)
    
    try:
        encrypted_language = base64.b64decode(language)
        decrypted_language = decrypt_data(encrypted_language, session_key).decode("utf-8")
        decrypted_language = bleach.clean(decrypted_language)
    except Exception as e:
        logger.error(f"Language decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted language")
    
    allowed_languages = ["kannada", "hindi", "tamil"]
    if decrypted_language not in allowed_languages:
        raise HTTPException(status_code=400, detail=f"Language must be one of {allowed_languages}")
    
    allowed_types = ["audio/mpeg", "audio/wav", "audio/flac"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type; allowed: {allowed_types}")
    
    encrypted_content = await file.read()
    if len(encrypted_content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large; max 10MB")
    
    try:
        file_content = decrypt_data(encrypted_content, session_key)
    except Exception as e:
        logger.error(f"Audio decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted audio")
    
    logger.info("Processing audio processing request", extra={
        "endpoint": "/v1/process_audio",
        "filename": file.filename,
        "language": decrypted_language,
        "client_ip": get_remote_address(request),
        "user_id": user_id
    })
    
    start_time = time()
    
    @audio_proc_breaker
    async def call_audio_proc_api():
        async with aiohttp.ClientSession() as session:
            form_data = aiohttp.FormData()
            form_data.add_field('file', file_content, filename=file.filename, content_type=file.content_type)
            external_url = f"{settings.external_audio_proc_url}/process_audio/?language={decrypted_language}"
            async with session.post(
                external_url,
                data=form_data,
                headers={"accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status >= 400:
                    raise HTTPException(status_code=response.status, detail=await response.text())
                return await response.json()
    
    try:
        processed_result = (await call_audio_proc_api()).get("result", "")
        logger.info(f"Audio processing completed in {time() - start_time:.2f} seconds")
        return AudioProcessingResponse(result=processed_result)
    except asyncio.TimeoutError:
        logger.error("Audio processing service timed out")
        raise HTTPException(status_code=504, detail="Audio processing service timeout")
    except aiohttp.ClientError as e:
        logger.error(f"Audio processing request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

@app.post("/v1/transcribe/", 
          response_model=TranscriptionResponse,
          summary="Transcribe Audio File",
          description="Transcribe an encrypted audio file into text in the specified encrypted language. Requires authentication and X-Session-Key header.",
          tags=["Audio"],
          responses={
              200: {"description": "Transcription result", "model": TranscriptionResponse},
              400: {"description": "Invalid encrypted audio or language"},
              401: {"description": "Unauthorized - Token required"},
              504: {"description": "Transcription service timeout"}
          })
async def transcribe_audio(
    file: UploadFile = File(..., description="Encrypted audio file to transcribe"),
    language: str = Query(..., description="Base64-encoded encrypted language of the audio (kannada, hindi, tamil after decryption)"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    user_id = await get_current_user(credentials)
    session_key = base64.b64decode(x_session_key)
    
    try:
        encrypted_language = base64.b64decode(language)
        decrypted_language = decrypt_data(encrypted_language, session_key).decode("utf-8")
        decrypted_language = bleach.clean(decrypted_language)
    except Exception as e:
        logger.error(f"Language decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted language")
    
    allowed_languages = ["kannada", "hindi", "tamil"]
    if decrypted_language not in allowed_languages:
        raise HTTPException(status_code=400, detail=f"Language must be one of {allowed_languages}")
    
    allowed_types = ["audio/mpeg", "audio/wav", "audio/flac"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type; allowed: {allowed_types}")
    
    encrypted_content = await file.read()
    if len(encrypted_content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large; max 10MB")
    
    try:
        file_content = decrypt_data(encrypted_content, session_key)
    except Exception as e:
        logger.error(f"Audio decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted audio")
    
    start_time = time()
    
    async with aiohttp.ClientSession() as session:
        form_data = aiohttp.FormData()
        form_data.add_field('file', file_content, filename=file.filename, content_type=file.content_type)
        external_url = f"{settings.external_asr_url}/transcribe/?language={decrypted_language}"
        try:
            async with session.post(
                external_url,
                data=form_data,
                headers={"accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status >= 400:
                    raise HTTPException(status_code=response.status, detail=await response.text())
                transcription = (await response.json()).get("text", "")
                logger.info(f"Transcription completed in {time() - start_time:.2f} seconds")
                return TranscriptionResponse(text=transcription)
        except asyncio.TimeoutError:
            logger.error("Transcription service timed out")
            raise HTTPException(status_code=504, detail="Transcription service timeout")
        except aiohttp.ClientError as e:
            logger.error(f"Transcription request failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/v1/chat_v2", 
          response_model=TranscriptionResponse,
          summary="Chat with Image (V2)",
          description="Generate a response from an encrypted text prompt and optional image. Requires authentication and X-Session-Key header.",
          tags=["Chat"],
          responses={
              200: {"description": "Chat response", "model": TranscriptionResponse},
              400: {"description": "Invalid prompt"},
              401: {"description": "Unauthorized - Token required"},
              429: {"description": "Rate limit exceeded"}
          })
@limiter.limit(lambda: runtime_config["chat_rate_limit"])
async def chat_v2(
    request: Request,
    prompt: str = Form(..., description="Base64-encoded encrypted text prompt for chat"),
    image: UploadFile = File(default=None, description="Optional encrypted image to accompany the prompt"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    user_id = await get_current_user(credentials)
    session_key = base64.b64decode(x_session_key)
    
    try:
        encrypted_prompt = base64.b64decode(prompt)
        decrypted_prompt = decrypt_data(encrypted_prompt, session_key).decode("utf-8")
        decrypted_prompt = bleach.clean(decrypted_prompt)
    except Exception as e:
        logger.error(f"Prompt decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted prompt")
    
    if not decrypted_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    image_data = None
    if image:
        if image.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid image type; allowed: jpeg, png")
        encrypted_image = await image.read()
        if len(encrypted_image) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large; max 10MB")
        try:
            decrypted_image = decrypt_data(encrypted_image, session_key)
            image_data = Image.open(io.BytesIO(decrypted_image))
        except Exception as e:
            logger.error(f"Image decryption failed: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid encrypted image")
    
    logger.info("Processing chat_v2 request", extra={
        "endpoint": "/v1/chat_v2",
        "prompt_length": len(decrypted_prompt),
        "has_image": bool(image),
        "client_ip": get_remote_address(request),
        "user_id": user_id
    })
    
    try:
        response_text = f"Processed: {decrypted_prompt}" + (" with image" if image_data else "")
        return TranscriptionResponse(text=response_text)
    except Exception as e:
        logger.error(f"Chat_v2 processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/v1/translate", 
          response_model=TranslationResponse,
          summary="Translate Text",
          description="Translate a list of base64-encoded encrypted sentences from an encrypted source to an encrypted target language. Requires authentication and X-Session-Key header.",
          tags=["Translation"],
          responses={
              200: {"description": "Translation result", "model": TranslationResponse},
              400: {"description": "Invalid encrypted sentences or languages"},
              401: {"description": "Unauthorized - Token required"},
              503: {"description": "Service unavailable due to repeated failures"},
              504: {"description": "Translation service timeout"}
          })
@limiter.limit(lambda: runtime_config["chat_rate_limit"])
async def translate(
    request: TranslationRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    user_id = await get_current_user(credentials)
    session_key = base64.b64decode(x_session_key)
    
    decrypted_sentences = []
    for sentence in request.sentences:
        try:
            encrypted_sentence = base64.b64decode(sentence)
            decrypted_sentence = decrypt_data(encrypted_sentence, session_key).decode("utf-8")
            decrypted_sentence = bleach.clean(decrypted_sentence)
            if not decrypted_sentence.strip():
                raise ValueError("Decrypted sentence is empty")
            decrypted_sentences.append(decrypted_sentence)
        except Exception as e:
            logger.error(f"Sentence decryption failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid encrypted sentence: {str(e)}")
    
    try:
        encrypted_src_lang = base64.b64decode(request.src_lang)
        decrypted_src_lang = decrypt_data(encrypted_src_lang, session_key).decode("utf-8")
        decrypted_src_lang = bleach.clean(decrypted_src_lang)
        if not decrypted_src_lang.strip():
            raise ValueError("Decrypted source language is empty")
        encrypted_tgt_lang = base64.b64decode(request.tgt_lang)
        decrypted_tgt_lang = decrypt_data(encrypted_tgt_lang, session_key).decode("utf-8")
        decrypted_tgt_lang = bleach.clean(decrypted_tgt_lang)
        if not decrypted_tgt_lang.strip():
            raise ValueError("Decrypted target language is empty")
    except Exception as e:
        logger.error(f"Language decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid encrypted language: {str(e)}")
    
    supported_languages = [
        "eng_Latn", "hin_Deva", "kan_Knda", "tam_Taml", "mal_Mlym", "tel_Telu",
        "deu_Latn", "fra_Latn", "nld_Latn", "spa_Latn", "ita_Latn", "por_Latn",
        "rus_Cyrl", "pol_Latn"
    ]
    if decrypted_src_lang not in supported_languages or decrypted_tgt_lang not in supported_languages:
        logger.error(f"Unsupported language codes: src={decrypted_src_lang}, tgt={decrypted_tgt_lang}")
        raise HTTPException(status_code=400, detail=f"Unsupported language codes: src={decrypted_src_lang}, tgt={decrypted_tgt_lang}")
    
    logger.info(f"Received translation request: {len(decrypted_sentences)} sentences, src_lang: {decrypted_src_lang}, tgt_lang: {decrypted_tgt_lang}, user_id: {user_id}")
    
    @translate_breaker
    async def call_translate_api():
        async with aiohttp.ClientSession() as session:
            external_url = "https://slabstech-dhwani-internal-api-server.hf.space/v1/translate"
            payload = {
                "sentences": decrypted_sentences,
                "src_lang": decrypted_src_lang,
                "tgt_lang": decrypted_tgt_lang
            }
            async with session.post(
                external_url,
                json=payload,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json"
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status >= 400:
                    raise HTTPException(status_code=response.status, detail=await response.text())
                return await response.json()
    
    try:
        response_data = await call_translate_api()
        translations = response_data.get("translations", [])
        if not translations or len(translations) != len(decrypted_sentences):
            logger.warning(f"Unexpected response format: {response_data}")
            raise HTTPException(status_code=500, detail="Invalid response from translation service")
        logger.info(f"Translation successful: {translations}")
        return TranslationResponse(translations=translations)
    except asyncio.TimeoutError:
        logger.error("Translation request timed out")
        raise HTTPException(status_code=504, detail="Translation service timeout")
    except aiohttp.ClientError as e:
        logger.error(f"Error during translation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/v1/visual_query", 
          response_model=VisualQueryResponse,
          summary="Visual Query with Image",
          description="Process a visual query with an encrypted text query, encrypted image, and encrypted language codes provided in a JSON body named 'data'. Requires authentication and X-Session-Key header.",
          tags=["Chat"],
          responses={
              200: {"description": "Query response", "model": VisualQueryResponse},
              400: {"description": "Invalid query, encrypted data, or language codes"},
              401: {"description": "Unauthorized - Token required"},
              429: {"description": "Rate limit exceeded"},
              503: {"description": "Service unavailable due to repeated failures"},
              504: {"description": "Visual query service timeout"}
          })
@limiter.limit(lambda: runtime_config["chat_rate_limit"])
async def visual_query(
    request: Request,
    data: str = Form(..., description="JSON string containing encrypted query, src_lang, and tgt_lang"),
    file: UploadFile = File(..., description="Encrypted image file to analyze"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    user_id = await get_current_user(credentials)
    session_key = base64.b64decode(x_session_key)
    
    try:
        import json
        visual_query_request = VisualQueryRequest.parse_raw(data)
        logger.info(f"Received visual query JSON: {data}")
    except Exception as e:
        logger.error(f"Failed to parse JSON data: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Invalid JSON data: {str(e)}")
    
    try:
        encrypted_query = base64.b64decode(visual_query_request.query)
        decrypted_query = decrypt_data(encrypted_query, session_key).decode("utf-8")
        decrypted_query = bleach.clean(decrypted_query)
        encrypted_src_lang = base64.b64decode(visual_query_request.src_lang)
        decrypted_src_lang = decrypt_data(encrypted_src_lang, session_key).decode("utf-8")
        decrypted_src_lang = bleach.clean(decrypted_src_lang)
        encrypted_tgt_lang = base64.b64decode(visual_query_request.tgt_lang)
        decrypted_tgt_lang = decrypt_data(encrypted_tgt_lang, session_key).decode("utf-8")
        decrypted_tgt_lang = bleach.clean(decrypted_tgt_lang)
    except Exception as e:
        logger.error(f"Decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted data")
    
    if not decrypted_query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(decrypted_query) > 1000:
        raise HTTPException(status_code=400, detail="Decrypted query cannot exceed 1000 characters")
    
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type; allowed: jpeg, png")
    
    encrypted_content = await file.read()
    if len(encrypted_content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large; max 10MB")
    
    try:
        decrypted_content = decrypt_data(encrypted_content, session_key)
    except Exception as e:
        logger.error(f"Image decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted image")
    
    logger.info("Processing visual query request", extra={
        "endpoint": "/v1/visual_query",
        "query_length": len(decrypted_query),
        "file_name": file.filename,
        "client_ip": get_remote_address(request),
        "user_id": user_id,
        "src_lang": decrypted_src_lang,
        "tgt_lang": decrypted_tgt_lang
    })
    
    @visual_query_breaker
    async def call_visual_query_api():
        async with aiohttp.ClientSession() as session:
            external_url = f"https://slabstech-dhwani-internal-api-server.hf.space/v1/visual_query/?src_lang={decrypted_src_lang}&tgt_lang={decrypted_tgt_lang}"
            form_data = aiohttp.FormData()
            form_data.add_field('file', decrypted_content, filename=file.filename, content_type=file.content_type)
            form_data.add_field('query', decrypted_query)
            async with session.post(
                external_url,
                data=form_data,
                headers={"accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status >= 400:
                    raise HTTPException(status_code=response.status, detail=await response.text())
                return await response.json()
    
    try:
        response_data = await call_visual_query_api()
        answer = response_data.get("answer", "")
        if not answer:
            logger.warning(f"Empty answer received from external API: {response_data}")
            raise HTTPException(status_code=500, detail="No answer provided by visual query service")
        logger.info(f"Visual query successful: {answer}")
        return VisualQueryResponse(answer=answer)
    except asyncio.TimeoutError:
        logger.error("Visual query request timed out")
        raise HTTPException(status_code=504, detail="Visual query service timeout")
    except aiohttp.ClientError as e:
        logger.error(f"Error during visual query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visual query failed: {str(e)}")

from enum import Enum

class SupportedLanguage(str, Enum):
    kannada = "kannada"
    hindi = "hindi"
    tamil = "tamil"

@app.post("/v1/speech_to_speech",
          summary="Speech-to-Speech Conversion",
          description="Convert input encrypted speech to processed speech in the specified encrypted language. Requires authentication and X-Session-Key header.",
          tags=["Audio"],
          responses={
              200: {"description": "Audio stream", "content": {"audio/mp3": {}}},
              400: {"description": "Invalid input, encrypted audio, or language"},
              401: {"description": "Unauthorized - Token required"},
              429: {"description": "Rate limit exceeded"},
              503: {"description": "Service unavailable due to repeated failures"},
              504: {"description": "External API timeout"}
          })
@limiter.limit(lambda: runtime_config["speech_rate_limit"])
async def speech_to_speech(
    request: Request,
    file: UploadFile = File(..., description="Encrypted audio file to process"),
    language: str = Query(..., description="Base64-encoded encrypted language of the audio (kannada, hindi, tamil after decryption)"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key")
) -> StreamingResponse:
    user_id = await get_current_user(credentials)
    session_key = base64.b64decode(x_session_key)
    
    try:
        encrypted_language = base64.b64decode(language)
        decrypted_language = decrypt_data(encrypted_language, session_key).decode("utf-8")
        decrypted_language = bleach.clean(decrypted_language)
    except Exception as e:
        logger.error(f"Language decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted language")
    
    allowed_languages = [lang.value for lang in SupportedLanguage]
    if decrypted_language not in allowed_languages:
        raise HTTPException(status_code=400, detail=f"Language must be one of {allowed_languages}")
    
    allowed_types = ["audio/mpeg", "audio/wav", "audio/flac"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type; allowed: {allowed_types}")
    
    encrypted_content = await file.read()
    if len(encrypted_content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large; max 10MB")
    
    try:
        file_content = decrypt_data(encrypted_content, session_key)
    except Exception as e:
        logger.error(f"Audio decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted audio")
    
    logger.info("Processing speech-to-speech request", extra={
        "endpoint": "/v1/speech_to_speech",
        "audio_filename": file.filename,
        "language": decrypted_language,
        "client_ip": get_remote_address(request),
        "user_id": user_id
    })
    
    @speech_to_speech_breaker
    async def call_speech_to_speech_api():
        async with aiohttp.ClientSession() as session:
            form_data = aiohttp.FormData()
            form_data.add_field('file', file_content, filename=file.filename, content_type=file.content_type)
            external_url = f"https://slabstech-dhwani-internal-api-server.hf.space/v1/speech_to_speech?language={decrypted_language}"
            async with session.post(
                external_url,
                data=form_data,
                headers={"accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status >= 400:
                    raise HTTPException(status_code=response.status, detail=await response.text())
                return response
    
    try:
        response = await call_speech_to_speech_api()
        headers = {
            "Content-Disposition": f"inline; filename=\"speech.mp3\"",
            "Cache-Control": "no-cache",
            "Content-Type": "audio/mp3"
        }
        return StreamingResponse(
            response.content.iter_any(8192),
            media_type="audio/mp3",
            headers=headers
        )
    except asyncio.TimeoutError:
        logger.error("External speech-to-speech API timed out", extra={"user_id": user_id})
        raise HTTPException(status_code=504, detail="External API timeout")
    except aiohttp.ClientError as e:
        logger.error(f"External speech-to-speech API error: {str(e)}", extra={"user_id": user_id})
        raise HTTPException(status_code=500, detail=f"External API error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default=settings.host, help="Host to run the server on.")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)