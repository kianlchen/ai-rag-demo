import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    llm_provider: str = os.getenv("LLM_PROVIDER", "dummy").lower()
    retry_threshold: float = float(os.getenv("RETRY_CONFIDENCE_THRESHOLD", "0.6"))
    model: str = os.getenv("MODEL", "gpt-4o-mini")

settings = Settings()