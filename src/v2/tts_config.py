# src/server/config/tts_config.py
from enum import Enum

SPEED = 1.0

class ResponseFormat(str, Enum):
    MP3 = "mp3"
    FLAC = "flac"
    WAV = "wav"

config = {
    "response_format": ResponseFormat.MP3,
    "supported_formats": [ResponseFormat.MP3, ResponseFormat.FLAC, ResponseFormat.WAV],
    "default_speed": SPEED
}