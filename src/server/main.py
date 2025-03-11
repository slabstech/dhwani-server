from fastapi import FastAPI, Request, HTTPException, Body, UploadFile, File, Depends, Query
from fastapi.responses import RedirectResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
from slowapi import Limiter
from slowapi.util import get_remote_address
import io
import soundfile as sf
import tempfile
from time import time
import argparse
import uvicorn
from PIL import Image
from typing import List
from pydub import AudioSegment

from config.logging_config import logger
from config.tts_config import SPEED, ResponseFormat, config as tts_config
from models.llm import LLMManager
from models.translation import TranslationManager
from models.tts import TTSManager
from models.vlm import VLMManager
from models.asr import ASRManager
from utils.auth import get_api_key, settings as auth_settings

class Settings(BaseSettings):
    llm_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    max_tokens: int = 512
    host: str = "0.0.0.0"
    port: int = 7860
    chat_rate_limit: str = "100/minute"
    speech_rate_limit: str = "5/minute"
    @field_validator("chat_rate_limit", "speech_rate_limit")
    def validate_rate_limit(cls, v):
        if not v.count("/") == 1 or not v.split("/")[0].isdigit():
            raise ValueError("Rate limit must be in format 'number/period' (e.g., '5/minute')")
        return v
    class Config:
        env_file = ".env"

settings = Settings()

app = FastAPI(title="Dhwani API", description="AI Chat API supporting Indian languages", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type", "Accept"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Initialize model managers with lazy loading
llm_manager = LLMManager(settings.llm_model_name)
trans_to_en = TranslationManager("kan_Knda", "eng_Latn")
trans_to_kn = TranslationManager("eng_Latn", "kan_Knda")
tts_manager = TTSManager()
vlm_manager = VLMManager()
asr_manager = ASRManager()

# Request/Response Models
class ChatRequest(BaseModel):
    prompt: str
    @field_validator('prompt')
    def prompt_must_be_valid(cls, v):
        if len(v) > 1000:
            raise ValueError("Prompt cannot exceed 1000 characters")
        return v.strip()

class ChatResponse(BaseModel):
    response: str

class SpeechRequest(BaseModel):
    input: str
    voice: str
    model: str
    response_format: ResponseFormat = tts_config.response_format
    speed: float = SPEED
    @field_validator('input')
    def input_must_be_valid(cls, v):
        if len(v) > 1000:
            raise ValueError("Input cannot exceed 1000 characters")
        return v.strip()
    @field_validator('response_format')
    def validate_response_format(cls, v):
        supported_formats = [ResponseFormat.MP3, ResponseFormat.FLAC, ResponseFormat.WAV]
        if v not in supported_formats:
            raise ValueError(f"Response format must be one of {[fmt.value for fmt in supported_formats]}")
        return v

class TranscriptionResponse(BaseModel):
    text: str

class BatchTranscriptionResponse(BaseModel):
    transcriptions: List[str]

# Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": settings.llm_model_name}

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

@app.post("/load_all_models")
async def load_all_models(api_key: str = Depends(get_api_key)):
    try:
        logger.info("Starting to load all models...")
        llm_manager.load()
        trans_to_en.load()
        trans_to_kn.load()
        tts_manager.load_model(tts_config.model)
        vlm_manager.load()
        asr_manager.load()  # Load default ASR model (kn)
        logger.info("All models loaded successfully")
        return {"status": "success", "message": "All models loaded"}
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")

@app.post("/v1/audio/speech")
@limiter.limit(settings.speech_rate_limit)
async def generate_audio(request: Request, speech_request: SpeechRequest = Body(...), api_key: str = Depends(get_api_key)):
    if not speech_request.input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")
    logger.info(f"Speech request: input={speech_request.input[:50]}..., voice={speech_request.voice}")
    try:
        audio_arr = tts_manager.generate_audio(speech_request.input, speech_request.voice, speech_request.model, speech_request.speed)
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_arr, 24000, format=speech_request.response_format.value)
        audio_buffer.seek(0)
        headers = {
            "Content-Disposition": f"inline; filename=\"speech.{speech_request.response_format.value}\"",
            "Cache-Control": "no-cache",
        }
        return StreamingResponse(audio_buffer, media_type=f"audio/{speech_request.response_format.value}", headers=headers)
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.chat_rate_limit)
async def chat(request: Request, chat_request: ChatRequest, api_key: str = Depends(get_api_key)):
    if not chat_request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    logger.info(f"Received prompt: {chat_request.prompt}")
    try:
        translated_prompt = trans_to_en.translate(chat_request.prompt)
        logger.info(f"Translated prompt to English: {translated_prompt}")
        response = llm_manager.generate(translated_prompt, settings.max_tokens)
        logger.info(f"Generated English response: {response}")
        translated_response = trans_to_kn.translate(response)
        logger.info(f"Translated response to Kannada: {translated_response}")
        return ChatResponse(response=translated_response)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/caption/")
async def caption_image(file: UploadFile = File(...), length: str = "normal"):
    image = Image.open(file.file)
    caption = vlm_manager.caption(image, length)
    return {"caption": caption}

@app.post("/visual_query/")
async def visual_query(file: UploadFile = File(...), query: str = Body(...)):
    image = Image.open(file.file)
    answer = vlm_manager.query(image, query)
    return {"answer": answer}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...), object_type: str = "face"):
    image = Image.open(file.file)
    objects = vlm_manager.detect(image, object_type)
    return {"objects": objects}

@app.post("/point/")
async def point_objects(file: UploadFile = File(...), object_type: str = "person"):
    image = Image.open(file.file)
    points = vlm_manager.point(image, object_type)
    return {"points": points}

@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...), language: str = Query(..., enum=list(asr_manager.model_language.keys())), api_key: str = Depends(get_api_key)):
    start_time = time()
    try:
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["wav", "mp3"]:
            logger.warning(f"Unsupported file format: {file_extension}")
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a WAV or MP3 file.")

        file_content = await file.read()
        audio = AudioSegment.from_mp3(io.BytesIO(file_content)) if file_extension == "mp3" else AudioSegment.from_wav(io.BytesIO(file_content))
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000).set_channels(1)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            audio.export(tmp_file.name, format="wav")
            tmp_file_path = tmp_file.name

        chunk_file_paths = asr_manager.split_audio(tmp_file_path)
        try:
            language_id = asr_manager.model_language.get(language, asr_manager.default_language)
            transcription = asr_manager.transcribe(chunk_file_paths, language_id)
            logger.info(f"Transcription completed in {time() - start_time:.2f} seconds")
            return JSONResponse(content={"text": transcription})
        finally:
            for chunk_file_path in chunk_file_paths:
                if os.path.exists(chunk_file_path):
                    os.remove(chunk_file_path)
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            asr_manager.cleanup()
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/transcribe_batch/", response_model=BatchTranscriptionResponse)
async def transcribe_audio_batch(files: List[UploadFile] = File(...), language: str = Query(..., enum=list(asr_manager.model_language.keys())), api_key: str = Depends(get_api_key)):
    start_time = time()
    all_transcriptions = []
    try:
        for file in files:
            file_extension = file.filename.split(".")[-1].lower()
            if file_extension not in ["wav", "mp3"]:
                logger.warning(f"Unsupported file format: {file_extension}")
                raise HTTPException(status_code=400, detail="Unsupported file format. Please upload WAV or MP3 files.")

            file_content = await file.read()
            audio = AudioSegment.from_mp3(io.BytesIO(file_content)) if file_extension == "mp3" else AudioSegment.from_wav(io.BytesIO(file_content))
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000).set_channels(1)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                audio.export(tmp_file.name, format="wav")
                tmp_file_path = tmp_file.name

            chunk_file_paths = asr_manager.split_audio(tmp_file_path)
            try:
                language_id = asr_manager.model_language.get(language, asr_manager.default_language)
                transcription = asr_manager.transcribe(chunk_file_paths, language_id)
                all_transcriptions.append(transcription)
            finally:
                for chunk_file_path in chunk_file_paths:
                    if os.path.exists(chunk_file_path):
                        os.remove(chunk_file_path)
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
                asr_manager.cleanup()

        logger.info(f"Batch transcription completed in {time() - start_time:.2f} seconds")
        return JSONResponse(content={"transcriptions": all_transcriptions})
    except Exception as e:
        logger.error(f"Error during batch transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch transcription failed: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default=settings.host, help="Host to run the server on.")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)