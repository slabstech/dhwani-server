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

import uvicorn
import aiohttp
import bleach
from databases import Database
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, Form, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, field_validator, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from pybreaker import CircuitBreaker
from PIL import Image

from utils.auth import get_current_user, get_current_user_with_admin, login, refresh_token, register, TokenResponse, Settings, LoginRequest, RegisterRequest, bearer_scheme, register_bulk_users, seed_initial_data
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
    # Create the tasks table
    await database.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            result TEXT,
            created_at REAL NOT NULL,
            completed_at REAL
        )
    """)
    # Create the users table
    await database.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            is_admin BOOLEAN NOT NULL DEFAULT 0
        )
    """)
    # Seed initial data
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
    #response.headers["X-Content-Type-Options"] = "nosniff"
    # TODO - check this for security
    #response.headers["X-Frame-Options"] = "DENY"
    #response.headers["Content-Security-Policy"] = "default-src 'self'"
    
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

limiter = Limiter(key_func=lambda request: get_remote_address(request))

class SpeechRequest(BaseModel):
    input: str = Field(..., description="Text to convert to speech (max 1000 characters)")
    voice: str = Field(..., description="Voice identifier for the TTS service")
    model: str = Field(..., description="TTS model to use")
    response_format: ResponseFormat = Field(tts_config.response_format, description="Audio format: mp3, flac, or wav")
    speed: float = Field(SPEED, description="Speech speed (default: 1.0)")

    @field_validator("input")
    def input_must_be_valid(cls, v):
        v = bleach.clean(v)
        if len(v) > 1000:
            raise ValueError("Input cannot exceed 1000 characters")
        return v.strip()

    @field_validator("response_format")
    def validate_response_format(cls, v):
        supported_formats = [ResponseFormat.MP3, ResponseFormat.FLAC, ResponseFormat.WAV]
        if v not in supported_formats:
            raise ValueError(f"Response format must be one of {[fmt.value for fmt in supported_formats]}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "input": "Hello, how are you?",
                "voice": "female-1",
                "model": "tts-model-1",
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

class TTSService(ABC):
    @abstractmethod
    async def generate_speech(self, payload: dict) -> aiohttp.ClientResponse:
        pass

class ExternalTTSService(TTSService):
    async def generate_speech(self, payload: dict) -> aiohttp.ClientResponse:
        async with aiohttp.ClientSession() as session:
            try:
                print(settings.external_tts_url)
                async with session.post(
                    settings.external_tts_url,
                    json=payload,
                    headers={"accept": "application/json", "Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status >= 400:
                        raise HTTPException(status_code=response.status, detail=await response.text())
                    return response
            except asyncio.TimeoutError:  # Fixed from aiohttp.ClientTimeout
                raise HTTPException(status_code=504, detail="External TTS API timeout")
            except aiohttp.ClientError as e:
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
         description="Returns basic request metrics (count and average response time per endpoint). Requires admin access.",
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
          description="Authenticate a user with username and password to obtain an access token and refresh token. Copy the access token and use it in the 'Authorize' button above.",
          tags=["Authentication"],
          responses={
              200: {"description": "Successful login", "model": TokenResponse},
              401: {"description": "Invalid username or password"},
              429: {"description": "Too many login attempts"}
          })
@limiter.limit("5/minute")
async def token(request: Request, login_request: LoginRequest):
    login_request.username = bleach.clean(login_request.username)
    login_request.password = bleach.clean(login_request.password)
    return await login(login_request)

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
          summary="Register New User",
          description="Create a new user account and return an access token and refresh token. Requires admin access (use 'admin' user with password 'adminpass' initially).",
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
          description="Upload a CSV file with 'username' and 'password' columns to register multiple users in the background. Returns a task ID to track progress. Requires admin access. Rate limited to 10 requests per minute per user.",
          tags=["Authentication"],
          responses={
              202: {"description": "Bulk registration started", "content": {"application/json": {"example": {"message": "Bulk registration started", "task_id": "unique-id"}}}},
              400: {"description": "Invalid CSV format or data"},
              401: {"description": "Unauthorized - Token required"},
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
             401: {"description": "Unauthorized - Token required"},
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
             200: {"description": "SQLite database file", "content": {"application/octet-stream": {"example": "Binary SQLite file"}}},
             401: {"description": "Unauthorized - Token required"},
             403: {"description": "Admin access required"},
             500: {"description": "Error exporting database"}
         })
async def export_db(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    current_user = await get_current_user_with_admin(credentials)
    db_path = "users.db"
    
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
          description="Upload a SQLite database file to replace the current user database. Requires admin access. The uploaded file must be a valid SQLite database with the expected schema.",
          tags=["Authentication"],
          responses={
              200: {"description": "Database imported successfully"},
              400: {"description": "Invalid database file"},
              401: {"description": "Unauthorized - Token required"},
              403: {"description": "Admin access required"},
              500: {"description": "Error importing database"}
          })
async def import_db(
    request: Request,
    file: UploadFile = File(..., description="SQLite database file to import"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    current_user = await get_current_user_with_admin(credentials)
    db_path = "users.db"
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
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
            if not cursor.fetchone():
                conn.close()
                os.remove(temp_path)
                raise HTTPException(status_code=400, detail="Uploaded file is not a valid user database")
            
            cursor.execute("PRAGMA table_info(users);")
            columns = [col[1] for col in cursor.fetchall()]
            expected_columns = ["username", "password", "is_admin"]
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
              401: {"description": "Unauthorized - Token required"},
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
          description="Convert text to speech in the specified format using an external TTS service. Rate limited to 5 requests per minute per user. Requires authentication.",
          tags=["Audio"],
          responses={
              200: {"description": "Audio stream", "content": {"audio/mp3": {"example": "Binary audio data"}}},
              400: {"description": "Invalid input"},
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
    tts_service: TTSService = Depends(get_tts_service)
):
    user_id = await get_current_user(credentials)
    if not speech_request.input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")
    
    logger.info("Processing speech request", extra={
        "endpoint": "/v1/audio/speech",
        "input_length": len(speech_request.input),
        "client_ip": get_remote_address(request),
        "user_id": user_id
    })
    
    payload = {
        "input": speech_request.input,
        "voice": bleach.clean(speech_request.voice),
        "model": bleach.clean(speech_request.model),
        "response_format": speech_request.response_format.value,
        "speed": speech_request.speed
    }
    
    # Fetch the full response
    response = await tts_service.generate_speech(payload)
    audio_content = await response.read()  # Buffer the entire audio content
    
    headers = {
        "Content-Disposition": f"inline; filename=\"speech.{speech_request.response_format.value}\"",
        "Cache-Control": "no-cache",
        "Content-Type": f"audio/{speech_request.response_format.value}"
    }
    
    async def stream_response():
        yield audio_content  # Yield the full content in one chunk
    
    return StreamingResponse(
        stream_response(),
        media_type=f"audio/{speech_request.response_format.value}",
        headers=headers
    )

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for chat (max 1000 characters)")
    src_lang: str = Field("kan_Knda", description="Source language code (default: Kannada)")

    @field_validator("prompt")
    def prompt_must_be_valid(cls, v):
        v = bleach.clean(v)
        if len(v) > 1000:
            raise ValueError("Prompt cannot exceed 1000 characters")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Hello, how are you?",
                "src_lang": "kan_Knda"
            }
        }

class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated chat response")

    class Config:
        json_schema_extra = {"example": {"response": "Hi there, I'm doing great!"}} 

chat_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

@lru_cache(maxsize=100)
def cached_chat_response(prompt: str, src_lang: str) -> str:
    return None

@app.post("/v1/chat", 
          response_model=ChatResponse,
          summary="Chat with AI",
          description="Generate a chat response from a prompt in the specified language. Rate limited to 100 requests per minute per user. Requires authentication.",
          tags=["Chat"],
          responses={
              200: {"description": "Chat response", "model": ChatResponse},
              400: {"description": "Invalid prompt"},
              401: {"description": "Unauthorized - Token required"},
              429: {"description": "Rate limit exceeded"},
              503: {"description": "Service unavailable due to repeated failures"},
              504: {"description": "Chat service timeout"}
          })
@limiter.limit(lambda: runtime_config["chat_rate_limit"])
async def chat(
    request: Request,
    chat_request: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    user_id = await get_current_user(credentials)
    if not chat_request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    cache_key = f"{chat_request.prompt}:{chat_request.src_lang}"
    cached_response = cached_chat_response(chat_request.prompt, chat_request.src_lang)
    if cached_response:
        logger.info(f"Cache hit for chat request: {cache_key}")
        return ChatResponse(response=cached_response)
    
    logger.info(f"Received prompt: {chat_request.prompt}, src_lang: {chat_request.src_lang}, user_id: {user_id}")
    
    external_url = "https://slabstech-dhwani-internal-api-server.hf.space/v1/chat"
    payload = {
        "prompt": chat_request.prompt,
        "src_lang": bleach.clean(chat_request.src_lang),
        "tgt_lang": bleach.clean(chat_request.src_lang)
    }
    
    async with aiohttp.ClientSession() as session:
        try:
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
                response_data = await response.json()
                response_text = response_data.get("response", "")
                cached_chat_response(chat_request.prompt, chat_request.src_lang)
                logger.info(f"Generated Chat response from external API: {response_text}")
                return ChatResponse(response=response_text)
        except asyncio.TimeoutError:  # Fixed from aiohttp.ClientTimeout
            logger.error("External chat API request timed out")
            raise HTTPException(status_code=504, detail="Chat service timeout")
        except aiohttp.ClientError as e:
            logger.error(f"Error calling external chat API: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

audio_proc_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

@app.post("/v1/process_audio/", 
          response_model=AudioProcessingResponse,
          summary="Process Audio File",
          description="Process an uploaded audio file in the specified language. Rate limited to 100 requests per minute per user. Requires authentication.",
          tags=["Audio"],
          responses={
              200: {"description": "Processed result", "model": AudioProcessingResponse},
              401: {"description": "Unauthorized - Token required"},
              429: {"description": "Rate limit exceeded"},
              503: {"description": "Service unavailable due to repeated failures"},
              504: {"description": "Audio processing timeout"}
          })
@limiter.limit(settings.chat_rate_limit)
async def process_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file to process"),
    language: str = Query(..., enum=["kannada", "hindi", "tamil"], description="Language of the audio"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    user_id = await get_current_user(credentials)
    
    allowed_types = ["audio/mpeg", "audio/wav", "audio/flac"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type; allowed: {allowed_types}")
    
    file_content = await file.read()
    if len(file_content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large; max 10MB")
    
    logger.info("Processing audio processing request", extra={
        "endpoint": "/v1/process_audio",
        "filename": file.filename,
        "client_ip": get_remote_address(request),
        "user_id": user_id
    })
    
    start_time = time()
    async with aiohttp.ClientSession() as session:
        try:
            form_data = aiohttp.FormData()
            form_data.add_field('file', file_content, filename=file.filename, content_type=file.content_type)
            
            external_url = f"{settings.external_audio_proc_url}/process_audio/?language={bleach.clean(language)}"
            #external_url = f"https://slabstech-asr-indic-server-cpu.hf.space/transcribe?language={bleach.clean(language)}"

            async with session.post(
                external_url,
                data=form_data,
                headers={"accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status >= 400:
                    raise HTTPException(status_code=response.status, detail=await response.text())
                processed_result = (await response.json()).get("result", "")
                logger.info(f"Audio processing completed in {time() - start_time:.2f} seconds")
                return AudioProcessingResponse(result=processed_result)
        except asyncio.TimeoutError:  # Fixed from aiohttp.ClientTimeout
            raise HTTPException(status_code=504, detail="Audio processing service timeout")
        except aiohttp.ClientError as e:
            logger.error(f"Audio processing request failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
@app.post("/v1/transcribe/", 
          response_model=TranscriptionResponse,
          summary="Transcribe Audio File",
          description="Transcribe an uploaded audio file into text in the specified language. Requires authentication.",
          tags=["Audio"])
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Query(..., enum=["kannada", "hindi", "tamil"], description="Language of the audio"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    user_id = await get_current_user(credentials)
    
    allowed_types = ["audio/mpeg", "audio/wav", "audio/flac", "audio/x-wav"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type; allowed: {allowed_types}")
    
    if file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large; max 10MB")
    
    file_content = await file.read()
    
    async with aiohttp.ClientSession() as session:
        form_data = aiohttp.FormData()
        form_data.add_field('file', file_content, filename=file.filename, content_type=file.content_type)
        
        external_url = f"https://slabstech-asr-indic-server-cpu.hf.space/transcribe/?language={bleach.clean(language)}"
        async with session.post(
            external_url,
            data=form_data,
            headers={"accept": "application/json"},
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status >= 400:
                raise HTTPException(status_code=response.status, detail=await response.text())
            transcription = (await response.json()).get("text", "")
            return TranscriptionResponse(text=transcription)
        
        
@app.post("/v1/chat_v2", 
          response_model=TranscriptionResponse,
          summary="Chat with Image (V2)",
          description="Generate a response from a text prompt and optional image. Rate limited to 100 requests per minute per user. Requires authentication.",
          tags=["Chat"],
          responses={
              200: {"description": "Chat response", "model": TranscriptionResponse},
              400: {"description": "Invalid prompt"},
              401: {"description": "Unauthorized - Token required"},
              429: {"description": "Rate limit exceeded"}
          })
@limiter.limit(settings.chat_rate_limit)
async def chat_v2(
    request: Request,
    prompt: str = Form(..., description="Text prompt for chat"),
    image: UploadFile = File(default=None, description="Optional image to accompany the prompt"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    user_id = await get_current_user(credentials)
    prompt = bleach.clean(prompt)
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    if image:
        if image.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid image type; allowed: jpeg, png")
        image_content = await image.read()
        if len(image_content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large; max 10MB")
    
    logger.info("Processing chat_v2 request", extra={
        "endpoint": "/v1/chat_v2",
        "prompt_length": len(prompt),
        "has_image": bool(image),
        "client_ip": get_remote_address(request),
        "user_id": user_id
    })
    
    try:
        image_data = Image.open(io.BytesIO(image_content)) if image else None
        response_text = f"Processed: {prompt}" + (" with image" if image_data else "")
        return TranscriptionResponse(text=response_text)
    except Exception as e:
        logger.error(f"Chat_v2 processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

class TranslationRequest(BaseModel):
    sentences: List[str] = Field(..., description="List of sentences to translate")
    src_lang: str = Field(..., description="Source language code")
    tgt_lang: str = Field(..., description="Target language code")

    @field_validator("sentences")
    def sanitize_sentences(cls, v):
        return [bleach.clean(sentence) for sentence in v]

    class Config:
        json_schema_extra = {
            "example": {
                "sentences": ["Hello", "How are you?"],
                "src_lang": "en",
                "tgt_lang": "kan_Knda"
            }
        }

class TranslationResponse(BaseModel):
    translations: List[str] = Field(..., description="Translated sentences")

    class Config:
        json_schema_extra = {"example": {"translations": ["ನಮಸ್ಕಾರ", "ನೀವು ಹೇಗಿದ್ದೀರಿ?"]}} 

translate_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

@app.post("/v1/translate", 
          response_model=TranslationResponse,
          summary="Translate Text",
          description="Translate a list of sentences from source to target language. Requires authentication.",
          tags=["Translation"],
          responses={
              200: {"description": "Translation result", "model": TranslationResponse},
              401: {"description": "Unauthorized - Token required"},
              500: {"description": "Translation service error"},
              503: {"description": "Service unavailable due to repeated failures"},
              504: {"description": "Translation service timeout"}
          })
async def translate(
    request: TranslationRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    user_id = await get_current_user(credentials)
    logger.info(f"Received translation request: {request.dict()}, user_id: {user_id}")
    
    external_url = f"https://slabstech-dhwani-internal-api-server.hf.space/translate?src_lang={bleach.clean(request.src_lang)}&tgt_lang={bleach.clean(request.tgt_lang)}"
    
    payload = {
        "sentences": request.sentences,
        "src_lang": request.src_lang,
        "tgt_lang": request.tgt_lang
    }
    
    async with aiohttp.ClientSession() as session:
        try:
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
                response_data = await response.json()
                translations = response_data.get("translations", [])
                
                if not translations or len(translations) != len(request.sentences):
                    logger.warning(f"Unexpected response format: {response_data}")
                    raise HTTPException(status_code=500, detail="Invalid response from translation service")
                
                logger.info(f"Translation successful: {translations}")
                return TranslationResponse(translations=translations)
        except asyncio.TimeoutError:  # Fixed from aiohttp.ClientTimeout
            logger.error("Translation request timed out")
            raise HTTPException(status_code=504, detail="Translation service timeout")
        except aiohttp.ClientError as e:
            logger.error(f"Error during translation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

class VisualQueryRequest(BaseModel):
    query: str
    src_lang: str = "kan_Knda"
    tgt_lang: str = "kan_Knda"

    @field_validator("query")
    def query_must_be_valid(cls, v):
        v = bleach.clean(v)
        if len(v) > 1000:
            raise ValueError("Query cannot exceed 1000 characters")
        return v.strip()

class VisualQueryResponse(BaseModel):
    answer: str

visual_query_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

@app.post("/v1/visual_query", 
          response_model=VisualQueryResponse,
          summary="Visual Query with Image",
          description="Process a visual query with an image and text question. Rate limited to 100 requests per minute per user. Requires authentication.",
          tags=["Chat"],
          responses={
              200: {"description": "Query response", "model": VisualQueryResponse},
              400: {"description": "Invalid query"},
              401: {"description": "Unauthorized - Token required"},
              429: {"description": "Rate limit exceeded"},
              503: {"description": "Service unavailable due to repeated failures"},
              504: {"description": "Visual query service timeout"}
          })
@limiter.limit(settings.chat_rate_limit)
async def visual_query(
    request: Request,
    query: str = Form(..., description="Text query for the visual content"),
    file: UploadFile = File(..., description="Image file to analyze"),
    src_lang: str = Query(default="kan_Knda", description="Source language code"),
    tgt_lang: str = Query(default="kan_Knda", description="Target language code"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    user_id = await get_current_user(credentials)
    query = bleach.clean(query)
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type; allowed: jpeg, png")
    
    file_content = await file.read()
    if len(file_content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large; max 10MB")
    
    logger.info("Processing visual query request", extra={
        "endpoint": "/v1/visual_query",
        "query_length": len(query),
        "file_name": file.filename,
        "client_ip": get_remote_address(request),
        "user_id": user_id,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang
    })
    
    external_url = f"https://slabstech-dhwani-internal-api-server.hf.space/v1/visual_query/?src_lang={bleach.clean(src_lang)}&tgt_lang={bleach.clean(tgt_lang)}"
    
    async with aiohttp.ClientSession() as session:
        try:
            form_data = aiohttp.FormData()
            form_data.add_field('file', file_content, filename=file.filename, content_type=file.content_type)
            form_data.add_field('query', query)
            
            async with session.post(
                external_url,
                data=form_data,
                headers={"accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status >= 400:
                    raise HTTPException(status_code=response.status, detail=await response.text())
                response_data = await response.json()
                answer = response_data.get("answer", "")
                
                if not answer:
                    logger.warning(f"Empty answer received from external API: {response_data}")
                    raise HTTPException(status_code=500, detail="No answer provided by visual query service")
                
                logger.info(f"Visual query successful: {answer}")
                return VisualQueryResponse(answer=answer)
        except asyncio.TimeoutError:  # Fixed from aiohttp.ClientTimeout
            logger.error("Visual query request timed out")
            raise HTTPException(status_code=504, detail="Visual query service timeout")
        except aiohttp.ClientError as e:
            logger.error(f"Error during visual query: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Visual query failed: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default=settings.host, help="Host to run the server on.")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)