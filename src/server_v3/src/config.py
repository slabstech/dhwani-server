from pydantic_settings import BaseSettings
from typing import List

SUPPORTED_LANGUAGES = {
    "asm_Beng", "kas_Arab", "pan_Guru", "ben_Beng", "kas_Deva", "san_Deva",
    "brx_Deva", "mai_Deva", "sat_Olck", "doi_Deva", "mal_Mlym", "snd_Arab",
    "eng_Latn", "mar_Deva", "snd_Deva", "gom_Deva", "mni_Beng", "tam_Taml",
    "guj_Gujr", "mni_Mtei", "tel_Telu", "hin_Deva", "npi_Deva", "urd_Arab",
    "kan_Knda", "ory_Orya"
}

class Settings(BaseSettings):
    llm_model_name: str = "google/gemma-3-12b-it"
    max_tokens: int = 512
    host: str = "0.0.0.0"
    port: int = 7860
    chat_rate_limit: str = "100/minute"

    class Config:
        env_file = ".env"

settings = Settings()