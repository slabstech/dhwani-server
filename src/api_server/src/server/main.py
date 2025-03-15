import argparse
import io
from time import time
from typing import List, Optional
from abc import ABC, abstractmethod

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from slowapi import Limiter
from slowapi.util import get_remote_address
import requests
from PIL import Image

# Assuming these are in your project structure
from config.tts_config import SPEED, ResponseFormat, config as tts_config
from config.logging_config import logger
from utils.auth import get_api_key

# Configuration settings
class Settings(BaseSettings):
    llm_model_name: str = "google/gemma-3-4b-it"
    max_tokens: int = 512
    host: str = "0.0.0.0"
    port: int = 7860
    chat_rate_limit: str = "100/minute"
    speech_rate_limit: str = "5/minute"
    external_tts_url: str = Field(..., env="EXTERNAL_TTS_URL")
    external_asr_url: str = Field(..., env="EXTERNAL_ASR_URL")
    external_text_gen_url: str = Field(..., env="EXTERNAL_TEXT_GEN_URL")
    external_audio_proc_url: str = Field(..., env="EXTERNAL_AUDIO_PROC_URL")
    api_key_secret: str = Field(..., env="API_KEY_SECRET")

    @field_validator("chat_rate_limit", "speech_rate_limit")
    def validate_rate_limit(cls, v):
        if not v.count("/") == 1 or not v.split("/")[0].isdigit():
            raise ValueError("Rate limit must be in format 'number/period' (e.g., '5/minute')")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# FastAPI app setup
app = FastAPI(
    title="Dhwani API",
    description="AI Chat API supporting Indian languages",
    version="1.0.0",
    redirect_slashes=False,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Request/Response Models
class SpeechRequest(BaseModel):
    input: str
    voice: str
    model: str
    response_format: ResponseFormat = tts_config.response_format
    speed: float = SPEED

    @field_validator("input")
    def input_must_be_valid(cls, v):
        if len(v) > 1000:
            raise ValueError("Input cannot exceed 1000 characters")
        return v.strip()

    @field_validator("response_format")
    def validate_response_format(cls, v):
        supported_formats = [ResponseFormat.MP3, ResponseFormat.FLAC, ResponseFormat.WAV]
        if v not in supported_formats:
            raise ValueError(f"Response format must be one of {[fmt.value for fmt in supported_formats]}")
        return v

class TranscriptionResponse(BaseModel):
    text: str

class TextGenerationResponse(BaseModel):
    text: str

class AudioProcessingResponse(BaseModel):
    result: str

# TTS Service Interface
class TTSService(ABC):
    @abstractmethod
    async def generate_speech(self, payload: dict) -> requests.Response:
        pass

class ExternalTTSService(TTSService):
    async def generate_speech(self, payload: dict) -> requests.Response:
        try:
            return requests.post(
                settings.external_tts_url,
                json=payload,
                headers={"accept": "application/json", "Content-Type": "application/json"},
                stream=True,
                timeout=10
            )
        except requests.Timeout:
            raise HTTPException(status_code=504, detail="External TTS API timeout")
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"External TTS API error: {str(e)}")

def get_tts_service() -> TTSService:
    return ExternalTTSService()

# Endpoints
@app.get("/v1/health")
async def health_check():
    return {"status": "healthy", "model": settings.llm_model_name}

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

@app.post("/v1/audio/speech")
@limiter.limit(settings.speech_rate_limit)
async def generate_audio(
    request: Request,
    speech_request: SpeechRequest = Depends(),
    api_key: str = Depends(get_api_key),
    tts_service: TTSService = Depends(get_tts_service)
):
    if not speech_request.input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")
    
    logger.info("Processing speech request", extra={
        "endpoint": "/v1/audio/speech",
        "input_length": len(speech_request.input),
        "client_ip": get_remote_address(request)
    })
    
    payload = {
        "input": speech_request.input,
        "voice": speech_request.voice,
        "model": speech_request.model,
        "response_format": speech_request.response_format.value,
        "speed": speech_request.speed
    }
    
    response = await tts_service.generate_speech(payload)
    response.raise_for_status()
    
    headers = {
        "Content-Disposition": f"inline; filename=\"speech.{speech_request.response_format.value}\"",
        "Cache-Control": "no-cache",
        "Content-Type": f"audio/{speech_request.response_format.value}"
    }
    
    return StreamingResponse(
        response.iter_content(chunk_size=8192),
        media_type=f"audio/{speech_request.response_format.value}",
        headers=headers
    )

@app.post("/v1/generate_text/", response_model=TextGenerationResponse)
@limiter.limit(settings.chat_rate_limit)
async def generate_text(
    file: UploadFile = File(...),
    language: str = Query(..., enum=["kannada", "hindi", "tamil"]),
    api_key: str = Depends(get_api_key),
    request: Request = None,
):
    logger.info("Processing text generation request", extra={
        "endpoint": "/v1/generate_text",
        "filename": file.filename,
        "client_ip": get_remote_address(request)
    })
    
    start_time = time()
    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        
        external_url = f"{settings.external_text_gen_url}/generate_text/?language={language}"
        response = requests.post(
            external_url,
            files=files,
            headers={"accept": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        
        generated_text = response.json().get("text", "")
        logger.info(f"Text generation completed in {time() - start_time:.2f} seconds")
        return TextGenerationResponse(text=generated_text)
    
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Text generation service timeout")
    except requests.RequestException as e:
        logger.error(f"Text generation request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

@app.post("/v1/process_audio/", response_model=AudioProcessingResponse)
@limiter.limit(settings.chat_rate_limit)
async def process_audio(
    file: UploadFile = File(...),
    language: str = Query(..., enum=["kannada", "hindi", "tamil"]),
    api_key: str = Depends(get_api_key),
    request: Request = None,
):
    logger.info("Processing audio processing request", extra={
        "endpoint": "/v1/process_audio",
        "filename": file.filename,
        "client_ip": get_remote_address(request)
    })
    
    start_time = time()
    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        
        external_url = f"{settings.external_audio_proc_url}/process_audio/?language={language}"
        response = requests.post(
            external_url,
            files=files,
            headers={"accept": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        
        processed_result = response.json().get("result", "")
        logger.info(f"Audio processing completed in {time() - start_time:.2f} seconds")
        return AudioProcessingResponse(result=processed_result)
    
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Audio processing service timeout")
    except requests.RequestException as e:
        logger.error(f"Audio processing request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

@app.post("/v1/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Query(..., enum=["kannada", "hindi", "tamil"]),
    #api_key: str = Depends(get_api_key),
    request: Request = None,
):
    '''
    logger.info("Processing transcription request", extra={
        "endpoint": "/v1/transcribe",
        "filename": file.filename,
        "client_ip": get_remote_address(request)
    })
    '''
    start_time = time()
    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        
        external_url = f"{settings.external_asr_url}/transcribe/?language={language}"
        response = requests.post(
            external_url,
            files=files,
            headers={"accept": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        
        transcription = response.json().get("text", "")
        #logger.info(f"Transcription completed in {time() - start_time:.2f} seconds")
        return TranscriptionResponse(text=transcription)
    
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Transcription service timeout")
    except requests.RequestException as e:
        #logger.error(f"Transcription request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/v1/chat_v2", response_model=TranscriptionResponse)
@limiter.limit(settings.chat_rate_limit)
async def chat_v2(
    request: Request,
    prompt: str = Form(...),
    image: UploadFile = File(default=None),
    api_key: str = Depends(get_api_key)
):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    logger.info("Processing chat_v2 request", extra={
        "endpoint": "/v1/chat_v2",
        "prompt_length": len(prompt),
        "has_image": bool(image),
        "client_ip": get_remote_address(request)
    })
    
    try:
        # For demonstration, we'll just return the prompt as text
        image_data = Image.open(await image.read()) if image else None
        response_text = f"Processed: {prompt}" + (" with image" if image_data else "")
        return TranscriptionResponse(text=response_text)
    except Exception as e:
        logger.error(f"Chat_v2 processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default=settings.host, help="Host to run the server on.")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)