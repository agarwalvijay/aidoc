from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    app_name: str = "AI Doctor"
    app_env: str = "dev"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # LLM provider selection
    llm_enabled: bool = True
    llm_provider: str = "openai"  # openai | deepseek | groq | google

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    # DeepSeek (OpenAI compatible)
    deepseek_api_key: Optional[str] = None
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"

    # Groq (OpenAI compatible)
    groq_api_key: Optional[str] = None
    groq_model: str = "openai/gpt-oss-20b"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Google Gemini
    google_api_key: Optional[str] = None
    google_model: str = "gemini-1.5-flash"

    # Safety posture: conservative means uncertain cases escalate.
    conservative_mode: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
