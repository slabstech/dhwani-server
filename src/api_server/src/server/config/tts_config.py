import enum
from pydantic_settings import BaseSettings

SPEED = 1.0

class StrEnum(str, enum.Enum):
    def __str__(self):
        return str(self.value)

class ResponseFormat(StrEnum):
    MP3 = "mp3"
    FLAC = "flac"
    WAV = "wav"

class Config(BaseSettings):
    log_level: str = "info"
    model: str = "ai4bharat/indic-parler-tts"
    max_models: int = 1
    lazy_load_model: bool = False  # Unused now, as all models are lazy-loaded
    input: str = "ನಿಮ್ಮ ಇನ್‌ಪುಟ್ ಪಠ್ಯವನ್ನು ಇಲ್ಲಿ ಸೇರಿಸಿ"
    voice: str = (
        "Female speaks with a high pitch at a normal pace in a clear, close-sounding environment. "
        "Her neutral tone is captured with excellent audio quality."
    )
    response_format: ResponseFormat = ResponseFormat.MP3

config = Config()