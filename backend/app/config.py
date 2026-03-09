from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    app_name: str = "AI Doctor"
    app_env: str = "dev"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # LLM provider selection
    llm_enabled: bool = True
    llm_provider: str = "anthropic"  # anthropic | openai | deepseek | groq | google

    # Anthropic Claude (recommended)
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-sonnet-4-6"

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"

    # DeepSeek (OpenAI compatible)
    deepseek_api_key: Optional[str] = None
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"

    # Groq (OpenAI compatible)
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Google Gemini
    google_api_key: Optional[str] = None
    google_model: str = "gemini-1.5-pro"

    # Safety posture: conservative floors uncertain cases at specialist_soon
    conservative_mode: bool = True
    # When True, the LLM may recommend specific medications (OTC + limited Rx).
    # When False (default), all medication output is framed as "ask your clinician about X".
    prescribing_enabled: bool = False
    min_turns_before_assessment: int = 3
    # 12 gives mental health (PHQ-9 + suicidality) enough turns without
    # affecting simpler presentations where the LLM self-terminates earlier.
    max_turns_before_assessment: int = 12

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
