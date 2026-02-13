"""Конфигурация проекта через Pydantic Settings."""

from pathlib import Path
from typing import Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Централизованные настройки приложения.

    Все параметры могут быть переопределены через переменные окружения
    или файл .env в корне проекта.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    CATALOG_PATH: Path = Path("data/synthetic_catalog.json")
    BM25_PATH: Path = Path("indices/bm25_index.pkl")
    CHROMA_PATH: Path = Path("indices/chroma")

    EMBEDDER_PATH: Path = Path("models/multilingual-e5-base")
    RERANKER_PATH: Path = Path("models/ms-marco-MiniLM-L-6-v2")

    GIGACHAT_CREDENTIALS: str = ""
    GIGACHAT_SCOPE: str = "GIGACHAT_API_PERS"
    GIGACHAT_VERIFY_SSL: bool = True
    GIGACHAT_CA_CERT_PATH: str | None = None
    GIGACHAT_CERT_FILE: str | None = None
    GIGACHAT_KEY_FILE: str | None = None
    GIGACHAT_MODEL: str = "GigaChat-Pro"

    GIGACHAT_RETRY_ATTEMPTS: int = 3
    GIGACHAT_RETRY_MIN_WAIT: int = 1
    GIGACHAT_RETRY_MAX_WAIT: int = 8

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 120
    CHROMA_BATCH_SIZE: int = 100

    BM25_TOP_K: int = 20
    VECTOR_TOP_K: int = 20
    RRF_K: int = 60
    RRF_WEIGHTS: Tuple[float, float] = (0.5, 0.5)
    FINAL_CANDIDATES_COUNT: int = 10
    RERANK_TOP_K: int = 5

    MAX_EDA_CHUNKS_IN_CONTEXT: int = 3
    LLM_TEMPERATURE: float = 0.4

    @field_validator("RRF_WEIGHTS", mode="before")
    @classmethod
    def parse_rrf_weights(cls, value: str | Tuple[float, float]) -> Tuple[float, float]:
        """Преобразует строку вида '0.5,0.5' в tuple(float, float)."""
        if isinstance(value, tuple):
            return value
        parts = [float(part.strip()) for part in str(value).split(",")]
        if len(parts) != 2:
            raise ValueError("RRF_WEIGHTS must contain exactly two numbers")
        return parts[0], parts[1]

    def resolve_path(self, path: Path) -> Path:
        """Возвращает абсолютный путь относительно корня проекта."""
        return path if path.is_absolute() else self.PROJECT_ROOT / path


settings = Settings()
